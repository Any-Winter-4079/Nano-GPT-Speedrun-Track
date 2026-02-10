import math
import time
import torch
import random
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

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"using {device}")

batch_size = 4
seq_len = 1024
n_steps = 165
lr = 3e-4
betas = (0.9,0.95)
eps = 1e-8

seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)

data_loader = DataLoader(batch_size, seq_len)

gpt_model = GPT(GPTConfig())
gpt_model.to(device)
# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
print(f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters")
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=lr, betas=betas, eps=eps)

for step in range(n_steps):
    start_t = time.time()
    optimizer.zero_grad()
    x, y = data_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = gpt_model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),1.0)
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    end_t = time.time()
    print(f"step: {step:,} | train loss: {loss.item():.8f} | time: {1000*(end_t - start_t):,.2f}ms | grad norm: {norm:.4f} | tok/s: {batch_size * seq_len / (end_t - start_t):,.2f}")

