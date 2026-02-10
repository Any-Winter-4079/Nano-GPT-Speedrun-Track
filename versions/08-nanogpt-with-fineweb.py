import os
import math
import time
import torch
import inspect
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# torchrun --standalone --nproc_per_node=8 8-gpt-3-small-with-fineweb.py
# Note: torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

##################################################
# 8 x A100 SXM | 256 vCPU 1006 GB RAM | $13.92/h # 
##################################################

# step: 9 | train loss:  10.38876629 | time: 327.57 ms | norm: 3.0004 | lr: 0.00000839 | tok/s: 1,600,518.84 | total toks: 5,242,880 | total time: 0.19 min

# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB
# 49059MiB /  81920MiB

@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    up_proj_factor: int = 4

class DataLoader:
    def __init__(self, gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_folder, split="train"):
        # batch size each gpu can fit in
        self.gpu_batch_size = gpu_batch_size
        # seq len for each gpu
        self.seq_len = seq_len
        # total gpu count
        self.ddp_world_size = ddp_world_size
        # 'id' of gpu
        self.ddp_global_rank = ddp_rank
        self.split = split
        self.epoch = 0

        shards = os.listdir(data_folder)
        shards = [shard for shard in shards if split in shard]
        sorted_shards = sorted(shards)
        sorted_shards = [os.path.join(data_folder, shard) for shard in sorted_shards]
        assert len(sorted_shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(sorted_shards)} shards for split {split}")
        self.sorted_shards = sorted_shards
        self.current_shard_idx = 0
        self.tokens = self.load_shard_tokens(sorted_shards[self.current_shard_idx])

        # there are ddp_world_size (e.g., 2) gpus,
        # each to get gpu_batch_size (e.g., 2) sequences,
        # each of seq_len (e.g., 1024) tokens
        # then for the current shard (loading the next shard upon token exhaustion):
        # we discard tokens that wouldn't fit into the final grad accumulation mini-step,
        # e.g., tokens in positions 12289-16000 if there were 16000 tokens total in 
        # the shard, keeping 0-12288 (12289 tokens, with 12288 fitting nicely into 3 
        # grad accum mini steps of 2 * 2 * 1024 = 4096 tokens each plus a final token
        # for y, since it is always one token ahead of x)
        self.grad_accum_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
        # needing grad_accum_mini_steps_per_shard (e.g., 3) to exhaust tokens
        self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
        # e.g., take up to 3 * 4096 = 12288 and a final token since y is one ahead
        self.tokens = self.tokens[:self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]

        # precompute all possible x sequence start indices (stride = seq_len)
        # e.g., [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264]
        # to get tokens 0-1023, 1024-2047, 2048-3071, 3072-4095, 4096-5119, 5120-6143,
        # 6144-7167, 7168-8191, 8192-9215, 9216-10239, 10240-11263, 11264-12287
        self.x_seq_starts = list(range(0, len(self.tokens) - seq_len, seq_len))

        if master_process:
            self.show_shard_info()

        # shuffle x chunk starts, e.g.,
        # [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        self._shuffle_shard()

        # after this counter reaches self.grad_accum_mini_steps_per_shard, the next shard should replace self.tokens
        self.grad_accum_mini_steps_per_shard_counter = 0
    
    def show_shard_info(self):
        print(f"total {self.split} shard {self.current_shard_idx} tokens: {len(self.tokens):,}")
        print(f"total gpus: {self.ddp_world_size}")
        print(f"gpu batch size: {self.gpu_batch_size:,} sequences")
        print(f"sequence length: {self.seq_len:,} tokens")
        print(f"tokens fed to gpu per grad accum mini-step: {(self.seq_len * self.gpu_batch_size):,} ({self.ddp_world_size:,} gpus, {self.grad_accum_tokens_per_mini_step:,} total tokens)")
        print(f"grad accumulation mini-steps for {self.split} shard {self.current_shard_idx} (each mini-step processing {self.grad_accum_tokens_per_mini_step:,} tokens): {(len(self.tokens) // self.grad_accum_tokens_per_mini_step):,}")
    
    def _shuffle_shard(self):
        g = torch.Generator()
        g.manual_seed(self.current_shard_idx + self.epoch)
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

    def load_shard_tokens(self, shard_path):
        return torch.tensor(np.load(shard_path).astype(np.int32), dtype=torch.long)
    
    def next_batch(self):
        if self.grad_accum_mini_steps_per_shard_counter == self.grad_accum_mini_steps_per_shard:
            self.current_shard_idx += 1
            self.grad_accum_mini_steps_per_shard_counter = 0
            if self.current_shard_idx >= len(self.sorted_shards):
                self.current_shard_idx = 0
                self.epoch += 1
                if master_process:
                    print(f"--- starting epoch {self.epoch} ---")
                    self.show_shard_info()
            self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])
            self._shuffle_shard()
        # e.g., 0 for the 1st grad accum mini-step, gpu_batch_size for the 2nd grad accum mini-step, etc.
        i = self.grad_accum_mini_steps_per_shard_counter * self.gpu_batch_size
        # e.g., for gpu 0: [0, 2048] for the 1st grad accum mini-step, [11264, 10240] for the 2nd grad accum mini-step, etc.
        #       for gpu 1: [1024, 5120] for the 1st grad accum mini-step, [6144, 8192] for the 2nd grad accum mini-step, etc.
        batch_starts = self.gpu_x_seq_starts[i:i + self.gpu_batch_size]
        # e.g., for gpu 0: [0-1023, 2048-3071] for the 1st grad accum mini-step
        #       for gpu 1: [1024-2047, 5120-6143] for the 1st grad accum mini-step
        x = torch.stack([self.tokens[start:start + self.seq_len] for start in batch_starts])
        y = torch.stack([self.tokens[start + 1:start + self.seq_len + 1] for start in batch_starts])
        # a full mini-step is to be done after this
        self.grad_accum_mini_steps_per_shard_counter += 1
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

total_tokens_per_step = 2**19
gpu_batch_size = 64
seq_len = 1024
total_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
grad_accum_mini_steps = total_tokens_per_step // total_tokens_per_mini_step
assert total_tokens_per_step % total_tokens_per_mini_step == 0
if master_process:
    print(f"gradient accumulation mini-steps: {grad_accum_mini_steps}")

betas = (0.9,0.95)
eps = 1e-8
max_lr = 6e-4
min_lr_after_warmup_ratio = 0.1
warmup_tokens = 375*10**6
warmup_and_cosine_tokens = 3*10**9
max_tokens = 5*10**9
weight_decay = 0.1

max_steps = max_tokens // total_tokens_per_step
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // total_tokens_per_step
warmup_and_cosine_steps = warmup_and_cosine_tokens // total_tokens_per_step
if master_process:
    print(f"warmup steps: {warmup_steps:,}")
    print(f"warmup and cosine steps: {warmup_and_cosine_steps:,}")
    print(f"max steps: {max_steps:,}")

seed = 1337 + ddp_rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = "cuda"

torch.set_float32_matmul_precision('high')

data_path = "./data/edu_fineweb10B"
train_data_loader = DataLoader(gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_path, "train")

gpt_model = GPT(GPTConfig())
gpt_model.to(device)
gpt_model = torch.compile(gpt_model)
gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
raw_gpt_model = gpt_model.module

# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
if master_process:
    print(f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters")

optimizer = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)

tokens_processed = 0
total_t = 0
for step in range(max_steps):
    start_t = time.time()
    optimizer.zero_grad()
    accum_loss = 0.0

    # grad accumulation mini-steps
    for mini_step in range(grad_accum_mini_steps):
        x, y = train_data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = gpt_model(x, y)
        loss /= grad_accum_mini_steps
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
    step_t = end_t - start_t
    total_t += step_t
    if master_process:
        tokens_processed += (grad_accum_mini_steps * total_tokens_per_mini_step)
        print(f"step: {step:,} | train loss: {accum_loss.item():.8f} | time: {1000*(end_t - start_t):,.2f} ms | grad norm: {norm:.4f} | lr: {lr:.8f} | tok/s: {ddp_world_size * grad_accum_mini_steps * gpu_batch_size * seq_len / step_t:,.2f} | total toks: {tokens_processed:,} | total time: {total_t/60:,.2f} min")

destroy_process_group()