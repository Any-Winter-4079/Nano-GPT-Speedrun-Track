import os
import math
import time
import torch
import inspect
import tiktoken
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# pip install tiktoken

# torchrun --standalone --nproc_per_node=8 7-gpt-3-small-with-ddp.py
# Note: torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

##################################################
# 8 x A100 SXM | 256 vCPU 1006 GB RAM | $13.92/h # 
##################################################

# step: 9 | bs: 4 | grad acc mini steps: 16 | train loss:  10.93424797 | t: 415.86ms | norm: 7.9850 | lr: 0.000000 | tok/s: 1,260,727.11

# Batch size can be increased but we need a larger dataset

# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB
# 6495MiB /  81920MiB

@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    up_proj_factor: int = 4

class DataLoader:
    def __init__(self, gpu_batch_size, seq_len, ddp_world_size, ddp_rank, epoch):
        # batch size each gpu can fit in
        self.gpu_batch_size = gpu_batch_size
        # seq len for each gpu
        self.seq_len = seq_len
        # total gpu count
        self.ddp_world_size = ddp_world_size
        # 'id' of gpu
        self.ddp_global_rank = ddp_rank
        # current epoch, to shuffle data once epoch is complete
        self.epoch = epoch

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        # there are ddp_world_size (e.g., 2) gpus,
        # each to get gpu_batch_size (e.g., 2) sequences,
        # each of seq_len (e.g., 1024) tokens
        # we discard tokens that wouldn't fit into the final grad accumulation mini-step,
        # e.g., tokens in positions 12289-16000 if there were 16000 tokens total in 
        # the shard, keeping 0-12288 (12289 tokens, with 12288 fitting nicely into 3 
        # grad accum mini steps of 2 * 2 * 1024 = 4096 tokens each plus a final token
        # for y, since it is always one token ahead of x)
        grad_accum_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
        # needing grad_accum_mini_steps_per_epoch (e.g., 3) to exhaust tokens
        self.grad_accum_mini_steps_per_epoch = len(self.tokens) // grad_accum_tokens_per_mini_step
        # e.g., take up to 3 * 4096 = 12288 and a final token since y is one ahead
        self.tokens = self.tokens[:grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_epoch + 1]

        # precompute all possible x sequence start indices (stride = seq_len)
        # e.g., [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264]
        # to get tokens 0-1023, 1024-2047, 2048-3071, 3072-4095, 4096-5119, 5120-6143,
        # 6144-7167, 7168-8191, 8192-9215, 9216-10239, 10240-11263, 11264-12287
        self.x_seq_starts = list(range(0, len(self.tokens) - seq_len, seq_len))

        if master_process:
            print(f"total dataset tokens: {len(self.tokens):,}")
            print(f"total gpus: {ddp_world_size}")
            print(f"gpu batch size: {gpu_batch_size:,} sequences")
            print(f"sequence length: {seq_len:,} tokens")
            print(f"tokens fed to gpu per grad accum mini-step: {(seq_len * gpu_batch_size):,} ({ddp_world_size:,} gpus, {grad_accum_tokens_per_mini_step:,} total tokens)")
            print(f"grad accumulation mini-steps per epoch (each processing {grad_accum_tokens_per_mini_step:,} tokens): {self.grad_accum_mini_steps_per_epoch}")

        # shuffle x chunk starts, e.g.,
        # [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        self._shuffle_for_epoch(self.epoch)

        # after this counter reaches self.grad_accum_mini_steps_per_epoch, an epoch should have passed
        self.grad_accum_mini_steps_per_epoch_counter = 0
    
    def _shuffle_for_epoch(self, epoch):
        g = torch.Generator()
        g.manual_seed(epoch)
        # e.g., [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        shuffled_indices = torch.randperm(len(self.x_seq_starts), generator=g).tolist()
        self.x_seq_starts = [self.x_seq_starts[i] for i in shuffled_indices]
        # e.g., 12 * 0 / 2 = 0 for gpu0, 12 * 1 / 2 = 6 for gpu 1
        gpu_x_seq_start_idx = len(self.x_seq_starts) * self.ddp_global_rank // self.ddp_world_size
        # e.g., 0 + 12 / 2 = 6 for gpu0, 6 + 12 / 2 = 12 for gpu 1
        gpu_x_seq_end_idx = gpu_x_seq_start_idx + len(self.x_seq_starts) // self.ddp_world_size
        # e.g., [0, 2048, 11264, 10240, 4096, 9216] for gpu 0
        # [1024, 5120, 6144, 8192, 3072, 7168] for gpu 1
        self.gpu_x_seq_starts = self.x_seq_starts[gpu_x_seq_start_idx:gpu_x_seq_end_idx]
    
    def next_batch(self):
        if self.grad_accum_mini_steps_per_epoch_counter == self.grad_accum_mini_steps_per_epoch:
            self.epoch += 1
            self.grad_accum_mini_steps_per_epoch_counter = 0
            self._shuffle_for_epoch(self.epoch)
        # e.g., 0 for the 1st grad accum mini-step, gpu_batch_size for the 2nd grad accum mini-step, etc.
        i = self.grad_accum_mini_steps_per_epoch_counter * self.gpu_batch_size
        # e.g., for gpu 0: [0, 2048] for the 1st grad accum mini-step, [11264, 10240] for the 2nd grad accum mini-step, etc.
        #       for gpu 1: [1024, 5120] for the 1st grad accum mini-step, [6144, 8192] for the 2nd grad accum mini-step, etc.
        batch_starts = self.gpu_x_seq_starts[i:i + self.gpu_batch_size]
        # e.g., for gpu 0: [0-1023, 2048-3071] for the 1st grad accum mini-step
        #       for gpu 1: [1024-2047, 5120-6143] for the 1st grad accum mini-step
        x = torch.stack([self.tokens[start:start + self.seq_len] for start in batch_starts])
        y = torch.stack([self.tokens[start + 1:start + self.seq_len + 1] for start in batch_starts])
        # a full mini-step is to be done after this
        self.grad_accum_mini_steps_per_epoch_counter += 1
        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.d_model % config.n_heads == 0
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads

    def forward(self, x):
        gpu_batch_size, seq_len, d_model = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(gpu_batch_size, seq_len, d_model)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.d_model, config.d_model * config.up_proj_factor)
        self.activ = nn.GELU()
        self.c_proj = nn.Linear(config.d_model * config.up_proj_factor, config.d_model)

    def forward(self, x):
        x = self.activ(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.residual_scale = (2 * config.n_layers) ** -0.5

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.residual_scale * self.attn(self.ln_1(x))
        x = x + self.residual_scale * self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, lr, betas, eps, weight_decay, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        return optimizer

    def forward(self, indices, targets=None):
        gpu_batch_size, seq_len = indices.size()
        te = self.transformer.wte(indices)
        pe = self.transformer.wpe(torch.arange(0, seq_len, dtype=torch.long, device=device))
        x = te + pe
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # (gpu_batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) * max_lr / warmup_steps
    elif step >= warmup_and_cosine_steps:
        return min_lr_after_warmup
    else:
        coeff = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (warmup_and_cosine_steps - warmup_steps)))
        return min_lr_after_warmup + coeff * (max_lr - min_lr_after_warmup)

init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

total_batch_size_in_tokens = 2**19
gpu_batch_size = 4 # micro-batch size in samples
seq_len = 1024
assert total_batch_size_in_tokens % (gpu_batch_size * seq_len) == 0
grad_accum_steps = total_batch_size_in_tokens // (ddp_world_size * gpu_batch_size * seq_len)
if master_process:
    print(f"gradient accumulation steps: {grad_accum_steps}")
betas = (0.9,0.95)
eps = 1e-8
max_lr = 6e-4
min_lr_after_warmup_ratio = 0.1
warmup_tokens = 375*10**6
warmup_and_cosine_tokens = 260*10**9
max_tokens = 300*10**9
weight_decay = 0.1

max_steps = max_tokens // (gpu_batch_size * seq_len)
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // (gpu_batch_size * seq_len)
warmup_and_cosine_steps = warmup_and_cosine_tokens // (gpu_batch_size * seq_len)
if master_process:
    print(f"warmup steps: {warmup_steps:,}")
    print(f"warmup and cosine steps: {warmup_and_cosine_steps:,}")
    print(f"max steps: {max_steps:,}")

seed = 1337 + ddp_rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = "cuda"

torch.set_float32_matmul_precision('high')

data_loader = DataLoader(gpu_batch_size, seq_len, ddp_world_size, ddp_rank, 0)

gpt_model = GPT(GPTConfig())
gpt_model.to(device)
gpt_model = torch.compile(gpt_model)
gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
raw_gpt_model = gpt_model.module

# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
if master_process:
    print(f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters")
optimizer = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)

for step in range(max_steps):
    start_t = time.time()
    optimizer.zero_grad()
    accum_loss = 0.0

    # grad accumulation mini-steps
    for micro_step in range(grad_accum_steps):
        x, y = data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        gpt_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = gpt_model(x, y)
        loss /= grad_accum_steps
        accum_loss += loss.detach()
        loss.backward()

    dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_t = time.time()
    if master_process:
        print(f"step: {step:,} | bs: {gpu_batch_size:,} | grad acc mini steps: {grad_accum_steps:,} | train loss: {accum_loss.item():.8f} | t: {1000*(end_t - start_t):,.2f}ms | grad norm: {norm:.4f} | lr: {lr:.6f} | tok/s: {ddp_world_size * grad_accum_steps * gpu_batch_size * seq_len / (end_t - start_t):,.2f}")

destroy_process_group()
