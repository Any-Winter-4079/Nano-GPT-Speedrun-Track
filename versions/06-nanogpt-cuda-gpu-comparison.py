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

# Pod template
# Runpod Pytorch 2.8.0 | runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Latest Gen

###########################################################################
# 1 x B200 | 28 vCPU 283 GB RAM | $5.99/hr | ==> 477M tok/$ | 2.86B tok/h #
###########################################################################

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77634907 | t: 667.35ms | norm: 7.7872 | lr: 0.000001 | tok/s: 785,629.48
# 48182MiB / 183359MiB

# step: 9 | bs: 128 | grad acc mini steps: 4 | train loss: 10.61479473 | t: 660.40ms | norm: 7.3937 | lr: 0.000002 | tok/s: 793,890.99
# 94436MiB / 183359MiB

# tok/h: 2,858,007,564
# tok/$: 477,129,810.35

###############################################################################
# 1 x H200 SXM | 24 vCPU 251 GB RAM | $3.99/hr | ==> 449M tok/$ | 1.79B tok/h #
###############################################################################

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77556229 | t: 1,058.85ms | norm: 7.7994 | lr: 0.000001 | tok/s: 495,147.37
# 47944MiB / 143771MiB

# step: 9 | bs: 128 | grad acc mini steps: 4 | train loss: 10.61416531 | t: 1,052.49ms | norm: 7.3798 | lr: 0.000002 | tok/s: 498,140.83
# 94382MiB / 143771MiB

# tok/h: 1,793,306,988
# tok/$: 449,450,372.93

###################################################################################
# 1 x RTX PRO 6000 | 16 vCPU 282 GB RAM | $1.79/hr | ==> 579M tok/$ | 1.04B tok/h #
###################################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.86120796 | t: 1,846.57ms | norm: 7.9141 | lr: 0.000001 | tok/s: 283,925.18
# 24716MiB /  97887MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77552986 | t: 1,821.87ms | norm: 7.7886 | lr: 0.000001 | tok/s: 287,774.03
# 47862MiB /  97887MiB

# tok/h: 1,035,986,508
# tok/$: 578,763,412.29

#############################################################################
# 1 x H100 NVL | 16 vCPU 180 GB RAM | $2.79/h | ==> 338M tok/$ | 944M tok/h #
#############################################################################

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77486420 | t: 2,024.14ms | norm: 7.8247 | lr: 0.000001 | tok/s: 259,017.52
# 47944MiB /  95830MiB

# step: 9 | bs: 128 | grad acc mini steps: 4 | train loss: 10.61607170 | t: 2,000.09ms | norm: 7.3479 | lr: 0.000002 | tok/s: 262,132.66
# 94430MiB /  95830MiB

# tok/h: 943,677,576
# tok/$: 338,235,690.32

###############################################################################
# 1 x H100 SXM | 26 vCPU 125 GB RAM | $2.69/hr | ==> 624M tok/$ | 1.68B tok/h #
###############################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.86010361 | t: 1,147.45ms | norm: 7.9264 | lr: 0.000001 | tok/s: 456,914.17
# 24814MiB /  81559MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77565670 | t: 1,124.02ms | norm: 7.7753 | lr: 0.000001 | tok/s: 466,439.96
# 47954MiB /  81559MiB

# tok/h: 1,679,183,856
# tok/$: 624,231,916.73

################################################################################
# 1 x H100 PCIe | 16 vCPU 251 GB RAM | $2.39/hr | ==> 436M tok/$ | 1.04B tok/h #
################################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85940361 | t: 1,837.20ms | norm: 7.9669 | lr: 0.000001 | tok/s: 285,373.26
# 24731MiB /  81559MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77542210 | t: 1,811.34ms | norm: 7.8184 | lr: 0.000001 | tok/s: 289,446.85
# 47901MiB /  81559MiB

# tok/h: 1,042,008,660
# tok/$: 435,986,887.03

########################################################################
# 1 x L40 | 16 vCPU 250 GB RAM | $0.99/h | ==> 350M tok/$ | 347M tok/h #
########################################################################

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90218735 | t: 5,454.58ms | norm: 8.0031 | lr: 0.000000 | tok/s: 96,118.82
# 12991MiB /  46068MiB

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.86079025 | t: 5,441.74ms | norm: 7.9421 | lr: 0.000001 | tok/s: 96,345.59
# 24555MiB /  46068MiB

# tok/h: 346,844,124
# tok/$: 350,347,600

########################################################################
# 1 x L40S | 16 vCPU 94 GB RAM | $0.86/h | ==> 597M tok/$ | 514M tok/h #
########################################################################

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90320587 | t: 3,704.91ms | norm: 7.9574 | lr: 0.000000 | tok/s: 141,511.57
# 12989MiB /  46068MiB

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85965919 | t: 3,675.58ms | norm: 7.9376 | lr: 0.000001 | tok/s: 142,640.82
# 24553MiB /  46068MiB

# tok/h: 513,506,952
# tok/$: 597,101,106.98

################################################################################
# 1 x RTX 6000 Ada | 16 vCPU 62 GB RAM | $0.77/h | ==> 501M tok/$ | 386M tok/h #
################################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85911083 | t: 4,892.75ms | norm: 7.9486 | lr: 0.000001 | tok/s: 107,156.05
# 24578MiB /  49140MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77541924 | t: 5,002.63ms | norm: 7.7934 | lr: 0.000001 | tok/s: 104,802.56
# 47708MiB /  49140MiB

# tok/h: 385,761,780
# tok/$: 500,989,324.68

#############################################################################
# 1 x RTX 5090 | 16 vCPU 141 GB RAM | $0.94/h | ==> 753M tok/$ | 708M tok/h #
#############################################################################

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90199757 | t: 2,767.73ms | norm: 8.0156 | lr: 0.000000 | tok/s: 189,429.11
# 13070MiB /  32607MiB

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85978222 | t: 2,667.44ms | norm: 7.9799 | lr: 0.000001 | tok/s: 196,551.34
# 24676MiB /  32607MiB

# tok/h: 707,584,824
# tok/$: 752,749,812.77

##################################################################
# 1 x RTX 4090 | 16 vCPU 62 GB RAM | $0.69/h | ==> tok/$ | tok/h #
##################################################################

# container init err

#####################################################################
# 1 x L4 | 4 vCPU 19 GB RAM | $0.43/h | ==> 338M tok/$ | 146M tok/h #
#####################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92448807 | t: 12,997.52ms | norm: 8.0040 | lr: 0.000000 | tok/s: 40,337.53
# 7731MiB /  23034MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90204239 | t: 12,970.13ms | norm: 7.9742 | lr: 0.000000 | tok/s: 40,422.74
# 12739MiB /  23034MiB

# tok/h: 145,521,864
# tok/$: 338,422,939.53

################################################################################
# 1 x RTX 4000 Ada | 16 vCPU 62 GB RAM | $0.26/h | ==> 763M tok/$ | 198M tok/h #
################################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92493153 | t: 9,864.08ms | norm: 8.0172 | lr: 0.000000 | tok/s: 53,151.26
# 7689MiB /  20475MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90263176 | t: 9,514.63ms | norm: 8.0097 | lr: 0.000000 | tok/s: 55,103.34
# 12677MiB /  20475MiB

# tok/h: 198,372,024
# tok/$: 762,969,323.08

###############################################################################
# 1 x RTX 2000 Ada | 6 vCPU 31 GB RAM | $0.23/h | ==> 491M tok/$ | 113M tok/h #
###############################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92374039 | t: 17,248.26ms | norm: 8.0073 | lr: 0.000000 | tok/s: 30,396.58
# 7590MiB /  16380MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90242863 | t: 16,718.44ms | norm: 7.9733 | lr: 0.000000 | tok/s: 31,359.86
# 12574MiB /  16380MiB

# tok/h: 112,895,496
# tok/$: 490,849,982.61

# Previous Gen

##############################################################################
# 1 x A100 PCIe | 31 vCPU 117 GB RAM | $1.64/h | ==> 416M tok/$ | 682M tok/h #
##############################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85941887 | t: 2,823.65ms | norm: 7.9685 | lr: 0.000001 | tok/s: 185,677.35
# 24591MiB /  81920MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77552700 | t: 2,768.25ms | norm: 7.8225 | lr: 0.000001 | tok/s: 189,393.43
# 47705MiB /  81920MiB

# tok/h: 681,816,348
# tok/$: 415,741,675.61

#############################################################################
# 1 x A100 SXM | 32 vCPU 251 GB RAM | $1.74/h | ==> 432M tok/$ | 752M tok/h # 
#############################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85972118 | t: 2,571.18ms | norm: 7.9372 | lr: 0.000001 | tok/s: 203,909.78
# 24551MiB /  81920MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77493858 | t: 2,509.02ms | norm: 7.8116 | lr: 0.000001 | tok/s: 208,961.64
# 47725MiB /  81920MiB

# tok/h: 752,261,904
# tok/$: 432,334,427.59

######################################################################
# 1 x A40 | 9 vCPU 50 GB RAM | $0.40/h | ==> 789M tok/$ | 315M tok/h #
######################################################################

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90247250 | t: 6,206.73ms | norm: 7.9940 | lr: 0.000000 | tok/s: 84,470.95
# 12781MiB /  46068MiB

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85945129 | t: 5,984.21ms | norm: 7.9424 | lr: 0.000001 | tok/s: 87,611.88
# 24365MiB /  46068MiB

# tok/h: 315,402,768
# tok/$: 788,506,920

############################################################################
# 1 x RTX A6000 | 8 vCPU 62 GB RAM | $0.49/h | ==> 679M tok/$ | 333M tok/h #
############################################################################

# step: 9 | bs: 32 | grad acc mini steps: 16 | train loss: 10.85971355 | t: 5,752.56ms | norm: 7.9608 | lr: 0.000001 | tok/s: 91,140.01
# 24366MiB /  49140MiB

# step: 9 | bs: 64 | grad acc mini steps: 8 | train loss: 10.77493382 | t: 5,671.41ms | norm: 7.8061 | lr: 0.000001 | tok/s: 92,444.03
# 47520MiB /  49140MiB

# tok/h: 332,798,508
# tok/$: 679,180,628.57

####################################################################
# 1 x RTX 3090 | 32 vCPU 125 GB RAM | $0.46/h | ==> tok/$ | tok/h  #
####################################################################

# container init err

##############################################################################
# 1 x RTX A5000 | 16 vCPU 62 GB RAM | $0.27/h | ==> 983M tok/$ | 265M tok/h  #
##############################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92375851 | t: 7,360.11ms | norm: 8.0253 | lr: 0.000000 | tok/s: 71,233.75
# 7702MiB /  24564MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90342903 | t: 7,113.12ms | norm: 7.9754 | lr: 0.000000 | tok/s: 73,707.20
# 12714MiB /  24564MiB

# tok/h: 265,345,920
# tok/$: 982,762,666.67

#############################################################################
# 1 x RTX A4500 | 12 vCPU 62 GB RAM | $0.25/h | ==> 899M tok/$ | 225M tok/h #
#############################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92211342 | t: 8,607.60ms | norm: 8.0005 | lr: 0.000000 | tok/s: 60,909.86
# 7673MiB /  20470MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90239429 | t: 8,395.25ms | norm: 7.9641 | lr: 0.000000 | tok/s: 62,450.57
# 12685MiB /  20470MiB

# tok/h: 224,822,052
# tok/$: 899,288,208

#############################################################################
# 1 x RTX A4000 | 16 vCPU 62 GB RAM | $0.25/h | ==> 743M tok/$ | 186M tok/h #
#############################################################################

# step: 9 | bs: 8 | grad acc mini steps: 64 | train loss: 10.92327213 | t: 10,653.83ms | norm: 7.9727 | lr: 0.000000 | tok/s: 49,211.21
# 7647MiB /  16376MiB

# step: 9 | bs: 16 | grad acc mini steps: 32 | train loss: 10.90207291 | t: 10,159.94ms | norm: 7.9959 | lr: 0.000000 | tok/s: 51,603.43
# 12659MiB /  16376MiB

# tok/h: 185,772,348
# tok/$: 743,089,392

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

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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

total_batch_size_in_tokens = 2**19
batch_size = 8 # micro-batch size in samples
seq_len = 1024
assert total_batch_size_in_tokens % (batch_size * seq_len) == 0
grad_accum_steps = total_batch_size_in_tokens // (batch_size * seq_len)
print(f"gradient accumulation steps: {grad_accum_steps}")
betas = (0.9,0.95)
eps = 1e-8
max_lr = 6e-4
min_lr_after_warmup_ratio = 0.1
warmup_tokens = 375*10**6
warmup_and_cosine_tokens = 260*10**9
max_tokens = 300*10**9
weight_decay = 0.1

max_steps = max_tokens // (batch_size * seq_len)
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // (batch_size * seq_len)
warmup_and_cosine_steps = warmup_and_cosine_tokens // (batch_size * seq_len)
print(f"warmup steps: {warmup_steps:,}")
print(f"warmup and cosine steps: {warmup_and_cosine_steps:,}")
print(f"max steps: {max_steps:,}")

seed = 1337
device = "cuda"
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = "cuda"

torch.set_float32_matmul_precision('high')

data_loader = DataLoader(batch_size, seq_len)

gpt_model = GPT(GPTConfig())
gpt_model.to(device)
gpt_model = torch.compile(gpt_model)
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
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = gpt_model(x, y)
        loss /= grad_accum_steps
        accum_loss += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    end_t = time.time()
    print(f"step: {step:,} | bs: {batch_size:,} | grad acc mini steps: {grad_accum_steps:,} | train loss: {accum_loss.item():.8f} | t: {1000*(end_t - start_t):,.2f}ms | grad norm: {norm:.4f} | lr: {lr:.6f} | tok/s: {grad_accum_steps * batch_size * seq_len / (end_t - start_t):,.2f}")
