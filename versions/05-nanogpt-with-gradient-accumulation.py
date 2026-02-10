import math
import time
import torch
import random
import inspect
import tiktoken
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
# pip install tiktoken
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    up_proj_factor: int = 4

class DataLoader:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        # precompute all possible chunk start indices (stride = seq_len)
        self.chunk_starts = list(range(0, len(self.tokens) - seq_len - 1, seq_len))
        print(f"total dataset tokens: {len(tokens):,}")
        print(f"batch size: {batch_size:,} samples")
        print(f"sequence length: {seq_len:,} tokens")
        print(f"dataset possible samples of length {seq_len:,}: {len(self.chunk_starts):,}")
        print(f"1 step: {batch_size:,} random samples of {seq_len:,} tokens each (total tokens: {batch_size * seq_len:,})")
        print(f"1 epoch: {len(self.chunk_starts) // batch_size} steps")
        #print(f"1 epoch: {len(self.tokens) // (batch_size * seq_len):,} steps")

        self.current_pos = 0
        self._shuffle_indices()

    def _shuffle_indices(self):
        random.shuffle(self.chunk_starts)

    def next_batch(self):
        if self.current_pos + self.batch_size > len(self.chunk_starts):
            # end of epoch; shuffle and restart
            self._shuffle_indices()
            self.current_pos = 0

        batch_starts = self.chunk_starts[self.current_pos:self.current_pos + self.batch_size]
        x = torch.stack([self.tokens[i:i+self.seq_len] for i in batch_starts])
        y = torch.stack([self.tokens[i+1:i+self.seq_len+1] for i in batch_starts])
        self.current_pos += self.batch_size
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
        batch_size, seq_len, d_model = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

        attn = q @ k.transpose(2,3) / math.sqrt(self.head_size)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=3)

        y = attn @ v
        y = y.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        return optimizer
    
    def forward(self, indices, targets=None):
        batch_size, seq_len = indices.size()
        te = self.transformer.wte(indices)
        pe = self.transformer.wpe(torch.arange(0, seq_len, dtype=torch.long, device=device))
        x = te + pe
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # (batch_size, seq_len, vocab_size)
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

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"using {device}")

total_batch_size_in_tokens = 2**14
batch_size = 4 # micro-batch size in samples
seq_len = 1024
assert total_batch_size_in_tokens % (batch_size * seq_len) == 0
grad_accum_steps = total_batch_size_in_tokens // (batch_size * seq_len)
print(f"gradient accumulation steps: {grad_accum_steps}")
betas = (0.9,0.95)
eps = 1e-8
max_lr = 6e-4
min_lr_after_warmup_ratio = 0.1
warmup_tokens = 1024 * 64
warmup_and_cosine_tokens = 1024 * 192
max_tokens = 1024 * 256
weight_decay = 0.1

max_steps = max_tokens // (batch_size * seq_len)
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // (batch_size * seq_len)
warmup_and_cosine_steps = warmup_and_cosine_tokens // (batch_size * seq_len)
print(f"warmup steps: {warmup_steps:,}")
print(f"warmup and cosine steps: {warmup_and_cosine_steps:,}")
print(f"max steps: {max_steps:,}")

seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)

device_type = "cuda"

data_loader = DataLoader(batch_size, seq_len)

gpt_model = GPT(GPTConfig())
gpt_model.to(device)
# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
print(f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters")
optimizer = gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)

for step in range(max_steps):
    start_t = time.time()
    optimizer.zero_grad()
    accum_loss = 0.0
    for grad_accum_step in range(grad_accum_steps):
        x, y = data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        logits, loss = gpt_model(x, y)
        loss /= grad_accum_steps
        accum_loss += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    end_t = time.time()
    print(f"step: {step:,} | bs: {batch_size:,} | grad acc mini steps: {grad_accum_steps:,} | train loss: {accum_loss.item():.8f} | t: {1000*(end_t - start_t):,.2f}ms | grad norm: {norm:.4f} | lr: {lr:.6f} | tok/s: {grad_accum_steps * batch_size * seq_len / (end_t - start_t):,.2f}")

