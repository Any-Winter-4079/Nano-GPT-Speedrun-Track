import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy
import math
import time
import torch
import inspect
import tiktoken
import threading
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch.distributed as dist
from datasets import load_dataset
from dataclasses import dataclass
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_model, load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask, and_masks
# pip install tiktoken huggingface_hub safetensors

# torchrun --standalone --nproc_per_node=4 28-gpt-3-small-with-blockmask-from-kv-blocks.py
# Note: torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

################################################
# 4 x H100 SXM | 56 vCPU 503 GB RAM | $10.86/h # 
################################################

################################################
#                   GPT Config                 #
################################################
@dataclass
class GPTConfig:
    max_seq_len: int = 1024 # overwritten by max(seq_len_train, seq_len_val)
    vocab_size: int = 50304
    n_layers: int = 12
    # n_heads == n_kv_heads    for MHA (Multi-Head Attention)
    # n_heads > n_kv_heads > 1 for GQA (Grouped-Query Attention) 
    # and n_kv_heads == 1      for MQA  (Multi-Query Attention)
    # NOTE: increasing n_heads (fixing n_kv_heads) reduces the parameters for GQA/MQA
    # q_proj: (d_model → n_heads * head_size)
    # k_proj: (d_model → n_kv_heads * head_size)
    # when d_model is split, the more (q) heads, the smaller the head_size, which is reused for k, v
    n_heads: int = 12
    n_kv_heads: int = 4
    d_model: int = 768
    up_proj_factor: int = 4
    use_tied_embeddings: bool = True
    pad_token_id: int = 50257
    eos_token_id: int = 50256
    norm_type: str = "rms" # "rms" or any other name for "layer"
    use_bias: bool = False

    pos_encoding_type: str = "rope" # "rope", "nope", or any other name for "absolute"
    rope_base_theta: int = 500_000

    activation: str = "swiglu" # "gelu", "relu", "relu2" (relu^2), "silu" or "swiglu"
    use_fair_swiglu: bool = True # only if swiglu

    optimizer_type: str = "muon" # "muon" (and adamw) or any other name for "adamw"
    # Muon usually likes a smaller LR than AdamW (e.g., 0.1x)
    muon_lr_scale: float = 0.15
    muon_backend: str = 'newtonschulz5'
    # try 5–10; 8–10 for slightly “tighter” orthogonalization (slower)
    muon_backend_steps: int = 8
    muon_momentum: float = 0.95
    use_nesterov: bool = True

    # <-- Uncomment for gradient norm clipping -->
    # use_grad_norm_clipping: bool = False
    # gradient_clipping_norm: float = 1.0
    # <-- Uncomment for gradient norm clipping -->

    use_qk_norm: bool = True
    qk_norm_type: str = "rms" # "rms" or any other name for "l2"
    qk_scale_init: float = 1.0 # init per-head scale
    qk_scale_max: float = 5.0 # cap scales to keep logits in range
    qk_eps: float = 1e-6
    use_qk_debug_log: bool = True

    # <-- Uncomment for logit soft-capping -->
    # use_lm_head_logit_softcapping: bool = True
    # lm_head_logit_softcap: float = 20.0
    # <-- Uncomment for logit soft-capping -->

    is_causal: bool = True # True for decoders or False for encoders - NOTE: affects sliding_window_size! -
    use_flex_attention: bool = True # True for FlexAttention or False for SDPA
    flex_block_size: int = 128
    # NOTE: for performance reasons, SWA, attention logit soft capping and doc masking require FlexAttention
    # While is_causal, RoPE and sampling are supported by both (SDPA and FlexAttention) branches
    use_doc_masking: bool = True
    use_sliding_window_attention: bool = True # True for Sliding Window (local) Attention (e.g., Mistral-style)
    sliding_window_size: int = 1024 # number of most-recent tokens a query can attend to (only used if use_sliding_window_attention=True)
    use_attn_logit_softcapping: bool = False # True for tanh soft-capping of attention logits (Gemma2/Grok-1 style), applied via score_mod before softmax
    attn_logit_softcap: float = 15.0
    tanh_backend: str = "ptx" # "clamp", "ptx" (faster) or "rational" or "exact"

################################################
#            Fast tanh Soft Capping            #
################################################
# adapted from PyTorch FlexAttention "attention-gym" example
# -- Start of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --
_PTX_TANH_LOWERED = False

try:
    # use inductor internals if present; otherwise we keep ptx unavailable
    from functools import partial
    from torch._inductor.virtualized import ops
    from torch._inductor.lowering import make_pointwise, register_lowering

    # avoid double-registrations when re-running in dev
    try:
        torch.ops.approx.tanh
        _ALREADY_REGISTERED = True
    except (AttributeError, RuntimeError):
        _ALREADY_REGISTERED = False

    if not _ALREADY_REGISTERED:
        @torch.library.custom_op("approx::tanh", mutates_args=())
        def _tanh_approx(inp: torch.Tensor) -> torch.Tensor:
            # eager fallback path
            return torch.tanh(inp)

        @_tanh_approx.register_fake
        def _(inp: torch.Tensor) -> torch.Tensor:
            return torch.tanh(inp)

        def _tanh_approx_lowering(inp):
            # inline ptx (f32)
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)

    _PTX_TANH_LOWERED = True
except Exception:
    # any failure leaves ptx unavailable
    _PTX_TANH_LOWERED = False

class _TanhApproxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ptx op is f32; caller upcasts beforehand
        y = torch.ops.approx.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        # d/dx tanh(x) = 1 - tanh^2(x)
        return grad_out * (1 - y * y)

def _tanh_ptx(x: torch.Tensor) -> torch.Tensor:
    # expects float32
    return _TanhApproxFn.apply(x)

def generate_tanh_softcap(soft_cap: float, backend: str):
    """
    returns a score_mod closure for flexattention that applies tanh soft-capping.

    backend: "ptx" | "rational" | "exact"
      - "ptx": use inline ptx tanh.approx (requires inductor lowering to be available)
      - "rational": use polynomial approx (fast & portable)
      - "exact": use torch.tanh
    """
    sc = float(soft_cap)

    if backend == "clamp":
        def fn(score):
            # clamp to [-sc, sc] with no transcendental ops
            return torch.clamp(score, min=-sc, max=sc)
    elif backend == "exact":
        def fn(score):
            return sc * torch.tanh(score / sc)
    elif backend == "rational":
        def fn(score):
            z = (score / sc).to(torch.float32)
            x2 = z * z
            y = z * (27.0 + x2) / (27.0 + 9.0 * x2)
            return (y * sc).to(dtype=score.dtype)
    elif backend == "ptx":
        if not _PTX_TANH_LOWERED:
            raise RuntimeError(
                "tanh_backend='ptx' requested, but ptx lowering is unavailable. "
                "pick 'rational' or 'exact'."
            )
        def fn(score):
            z32 = (score / sc).to(torch.float32)
            y32 = _tanh_ptx(z32)
            return (y32.to(score.dtype) * sc)
    else:
        raise ValueError(f"unknown tanh backend: {backend}")

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return fn(score)

    tanh_softcap.__name__ = f"tanh_softcap_{backend}_{int(sc)}"
    return tanh_softcap
# -- End of (modified) source: https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/softcapping.py --

################################################
#                      RoPE                    #
################################################
class Rotary(nn.Module):
    def __init__(self, head_size, base_theta, max_seq_len):
        super().__init__()
        assert head_size % 2 == 0, "RoPE needs even head_size"
        # head_size/2,
        # frequencies decay geometrically down to ~1/base_theta with ratio 1/base_theta^(2/head_size)
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        # register_buffer registers as a non-trainable buffer that moves/casts with the model (to device/dtype)
        # persistent=False avoids taking checkpoint space (won't be in state_dict)
        # buffers are broadcast to all ranks (broadcast_buffers=True), so this stays in sync
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        # max_seq_len, head_size/2
        freqs = torch.outer(t, self.inv_freq)
        # 1, 1, max_seq_len, head_size/2
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(self, seq_len, dtype, device):
        # slice to current length and cast on the fly to avoid
        # shape mutation of buffers
        cos = self.cos_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin

def apply_rotation(q, k, cos, sin):
    gpu_batch_size, n_heads, seq_len, head_size = q.shape
    _, n_kv_heads_k, _, _ = k.shape

    # gpu_batch_size, n_heads, seq_len, head_size // 2, 2
    q_pairs = q.contiguous().reshape(gpu_batch_size, n_heads, seq_len, head_size // 2, 2)
    k_pairs = k.contiguous().reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size // 2, 2)

    # gpu_batch_size, n_heads, seq_len, head_size/2 each
    q_x1, q_x2 = q_pairs[..., 0], q_pairs[..., 1]
    k_x1, k_x2 = k_pairs[..., 0], k_pairs[..., 1]

    # gpu_batch_size, n_heads, seq_len, head_size
    q_rot = torch.stack([q_x1 * cos - q_x2 * sin, q_x1 * sin + q_x2 * cos], dim=-1)
    k_rot = torch.stack([k_x1 * cos - k_x2 * sin, k_x1 * sin + k_x2 * cos], dim=-1)

    q = q_rot.reshape(gpu_batch_size, n_heads, seq_len, head_size)
    k = k_rot.reshape(gpu_batch_size, n_kv_heads_k, seq_len, head_size)
    
    return q, k

################################################
#                     Muon                     #
################################################
# In Muon, the goal is:
#   For 2D weight matrices (e.g., Linear weights of shape (out_features, in_features)),
#   build an SGD+momentum update g, then REPLACE it with its nearest orthogonal
#   matrix (orthogonalization), and apply that as the step. This helps keep layers
#   well-conditioned and discourages rank collapse.
 # ---------
# Muon vs AdamW:
#   Muon uses SGD + momentum to form g, then orthogonalizes g (via the backend) and scales it.
#   Use Muon for 2D block weights (e.g., transformer.h.*.weight).
#   Keep embeddings, layer norms, biases, and the final lm_head on AdamW.
#   Train these disjoint  param sets with their respective optimizers in parallel.
#   To use it with 4D convolutional filters, flatten the last 3 dimensions.
 # ---------
# The backend is simply the algorithm used to orthogonalize the 2D update matrix. Two choices are:
#   svd:
#       Exact, via SVD. If G = U S V^T, the projection of G (in Frobenius norm) onto the Stiefel
#       manifold (the set of all orthonormal k-frames, with a k-frame an ordered set of 
#       k linearly independent vectors in an n-dimensional vector space) is U V^T 
#       (i.e., the closest matrix with orthonormal columns/rows).
#   newtonschulz5:
#       Fast, iterative approximation (GPU-friendly) using a quintic Newton-Schulz iteration.
#       It avoids an SVD on every step, runs well in bfloat16, and plays nice with torch.compile.
# ---------
# It is recommended to use Nesterov-style momentum in the internal SGD.
# ---------
# # QKV note:
#   If a fused projection has shape (3d, d), we split it into three (d, d) blocks and
#   orthogonalize each block separately. After orthogonalization we scale updates by
#   sqrt(max(m, n)) so update magnitudes are well-behaved.
# ---------
# Why @torch.compile on newtonschulz5?
#   The kernel is a small loop of matmuls; compiling fuses ops and reduces Python overhead,
#   generating an efficient GPU kernel.
#   SVD is a single library call and doesn’t benefit.
# -----------------------------------------------------------

# (external)            Muon blog: https://kellerjordan.github.io/posts/muon/
# (another external)    Muon blog: https://jeremybernste.in/writing/deriving-muon

# -- Start of (modified) source: https://github.com/tyler-romero/nanogpt-speedrun/blob/main/src/train_gpt2.py --
# -----------------------------------------------------------------------------
# Muon optimizer
# Reference: https://kellerjordan.github.io/posts/muon/
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    In-place & buffer-reusing variant:
      - uses preallocated mm(out=...) to avoid new tensors each iter
      - uses in-place fused linear combination to reduce allocator pressure
      - keeps the 'transpose-if-tall' trick to minimize flops
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # orient to have rows <= cols so A = X X^T is the smaller square
    transposed = G.size(0) > G.size(1)
    X = G.T if transposed else G
    # scale so top singular value <= 1 (Frobenius norm upper-bounds spectral norm)
    denom = X.norm() + eps
    X = (X / denom).to(torch.bfloat16)

    m, n = X.shape  # m <= n
    # reusable buffers (compiled graph will persist them between calls)
    A = torch.empty((m, m), dtype=X.dtype, device=X.device)
    B = torch.empty_like(X)
    AB = torch.empty_like(X)

    for _ in range(steps):
        # A = X @ X^T
        torch.mm(X, X.T, out=A)
        # B = A @ X
        torch.mm(A, X, out=B)
        # AB = A @ B  (quintic NS5 term)
        torch.mm(A, B, out=AB)
        # X = a*X + b*B + c*AB  (in-place fused)
        X.mul_(a).add_(B, alpha=b).add_(AB, alpha=c)

    X = X.T if transposed else X
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g.add_(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1)) ** 0.5  # scale so the update has ≈ unit mean-squared magnitude
                p.data.add_(g, alpha=-lr * scale)
# -- End of (modified) source: https://github.com/tyler-romero/nanogpt-speedrun/blob/main/src/train_gpt2.py --

################################################
#                  DataLoader                  #
################################################
class DataLoader:
    def __init__(self, gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_folder, split="train", 
                 epoch=0, current_shard_idx=0, grad_accum_mini_steps_per_shard_counter=0,
                 pad_token_id=50257, eos_token_id=50256, return_document_ids=False,
                 shuffle_val_tokens=True):
        # batch size each gpu can fit in
        self.gpu_batch_size = gpu_batch_size
        # seq len for each gpu
        self.seq_len = seq_len
        # total gpu count
        self.ddp_world_size = ddp_world_size
        # 'id' of gpu
        self.ddp_global_rank = ddp_rank
        self.split = split
        self.epoch = epoch
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.return_document_ids = return_document_ids
        self.shuffle_val_tokens = shuffle_val_tokens

        shards = os.listdir(data_folder)
        shards = [shard for shard in shards if split in shard]
        sorted_shards = sorted(shards)
        sorted_shards = [os.path.join(data_folder, shard) for shard in sorted_shards]
        assert len(sorted_shards) > 0, f"no shards found for split {split}"
        if master_process:
            message = f"found {len(sorted_shards)} shards for split {split}"
            print(message)
            log_buffer.append(message)
        self.sorted_shards = sorted_shards
        self.current_shard_idx = current_shard_idx

        # load shard tokens (CPU long tensor; materialize now to avoid memmap stalls)
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
        self.grad_accum_mini_steps_per_shard_counter = grad_accum_mini_steps_per_shard_counter

        # prefetch machinery
        self._prefetch_thread = None
        self._next_tokens = None
        self._start_shard_prefetch()
    
    def show_shard_info(self):
        messages = [
            f"total {self.split} shard {self.current_shard_idx} tokens: {len(self.tokens):,}",
            f"total gpus: {self.ddp_world_size}",
            f"gpu batch size: {self.gpu_batch_size:,} sequences",
            f"sequence length: {self.seq_len:,} tokens",
            f"tokens fed to gpu per grad accum mini-step: {(self.seq_len * self.gpu_batch_size):,} ({self.ddp_world_size:,} gpus, {self.grad_accum_tokens_per_mini_step:,} total tokens)",
            f"per-gpu grad accumulation mini-steps for {self.split} shard {self.current_shard_idx} (each mini-step processing {self.grad_accum_tokens_per_mini_step:,} tokens): {(len(self.tokens) // self.grad_accum_tokens_per_mini_step):,}"
        ]
        for message in messages:
            print(message)
            log_buffer.append(message)
    
    def _shuffle_shard(self):
        if self.split == "train" or (self.split == "val" and self.shuffle_val_tokens):
            g = torch.Generator()
            g.manual_seed(self.current_shard_idx + self.epoch)
            # e.g., [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
            shuffled_indices = torch.randperm(len(self.x_seq_starts), generator=g).tolist()
            self.x_seq_starts = [self.x_seq_starts[i] for i in shuffled_indices]
        # else: keep original order for validation (global prefix)

        # e.g., 12 * 0 / 2 = 0 for gpu0, 12 * 1 / 2 = 6 for gpu 1
        gpu_x_seq_start_idx = len(self.x_seq_starts) * self.ddp_global_rank // self.ddp_world_size
        # e.g., 0 + 12 / 2 = 6 for gpu0, 6 + 12 / 2 = 12 for gpu 1
        gpu_x_seq_end_idx = gpu_x_seq_start_idx + len(self.x_seq_starts) // self.ddp_world_size
        # e.g., [0, 2048, 11264, 10240, 4096, 9216] for gpu 0
        # [1024, 5120, 6144, 8192, 3072, 7168] for gpu 1
        self.gpu_x_seq_starts = self.x_seq_starts[gpu_x_seq_start_idx:gpu_x_seq_end_idx]


    def _start_shard_prefetch(self):
        # prefetch the next shard if not already running
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        next_idx = (self.current_shard_idx + 1) % len(self.sorted_shards)

        def _worker():
            # load + convert once on CPU (int32 -> long), avoids memmap stalls
            np_arr = np.load(self.sorted_shards[next_idx])
            tokens = torch.tensor(np_arr.astype(np.int32), dtype=torch.long)
            self._next_tokens = tokens

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def load_shard_tokens(self, shard_path):
        # materialize to CPU long tensor (stable, no memmap)
        return torch.tensor(np.load(shard_path).astype(np.int32), dtype=torch.long)
    
    def reset(self):
        self.current_shard_idx = 0
        self.grad_accum_mini_steps_per_shard_counter = 0
        self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])
        # recompute/all the same work as in __init__ for the current shard
        self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
        self.tokens = self.tokens[: self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]
        self.x_seq_starts = list(range(0, len(self.tokens) - self.seq_len, self.seq_len))
        self._shuffle_shard()
        # restart prefetch after reset
        self._next_tokens = None
        self._start_shard_prefetch()
    
    def next_batch(self):
        if self.grad_accum_mini_steps_per_shard_counter == self.grad_accum_mini_steps_per_shard:
            self.current_shard_idx += 1
            self.grad_accum_mini_steps_per_shard_counter = 0
            if self.current_shard_idx >= len(self.sorted_shards):
                self.current_shard_idx = 0
                self.epoch += 1
                if master_process:
                    message = f"--- starting epoch {self.epoch} ---"
                    print(message); log_buffer.append(message)
                    self.show_shard_info()

            # use prefetched tokens if ready, else fall back to blocking load
            if self._next_tokens is not None:
                self.tokens = self._next_tokens
                self._next_tokens = None
            else:
                self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])

            # recompute shard-dependent fields after swapping shards
            self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
            self.tokens = self.tokens[: self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]
            self.x_seq_starts = list(range(0, len(self.tokens) - self.seq_len, self.seq_len))

            self._shuffle_shard()
            # immediately prefetch the following shard
            self._start_shard_prefetch()
        
        if (self.split == "val") and (not self.shuffle_val_tokens):
            # global, ordered prefix partitioning across ranks
            global_batch_start = self.grad_accum_mini_steps_per_shard_counter * self.ddp_world_size * self.gpu_batch_size
            start = global_batch_start + self.ddp_global_rank * self.gpu_batch_size
            batch_starts = self.x_seq_starts[start:start + self.gpu_batch_size]
        else:
            # e.g., 0 for the 1st grad accum mini-step, gpu_batch_size for the 2nd grad accum mini-step, etc.
            i = self.grad_accum_mini_steps_per_shard_counter * self.gpu_batch_size
            # e.g., for gpu 0: [0, 2048] for the 1st grad accum mini-step, [11264, 10240] for the 2nd grad accum mini-step, etc.
            #       for gpu 1: [1024, 5120] for the 1st grad accum mini-step, [6144, 8192] for the 2nd grad accum mini-step, etc.
            batch_starts = self.gpu_x_seq_starts[i:i + self.gpu_batch_size]
            # e.g., for gpu 0: [0-1023, 2048-3071] for the 1st grad accum mini-step
            #       for gpu 1: [1024-2047, 5120-6143] for the 1st grad accum mini-step

        x = torch.stack([self.tokens[start:start + self.seq_len] for start in batch_starts])
        y = torch.stack([self.tokens[start + 1:start + self.seq_len + 1] for start in batch_starts])

        # optional doc ids (per-token doc index within the window)
        doc_ids = None
        if self.return_document_ids:
            is_eos = (x == self.eos_token_id)
            doc_ids = torch.cumsum(is_eos.to(torch.int32), dim=1)
            doc_ids = doc_ids - is_eos.to(torch.int32)

        # a full mini-step is to be done after this
        self.grad_accum_mini_steps_per_shard_counter += 1
        return x, y, doc_ids

################################################
#                QK Norm Debug                 #
################################################
@torch.no_grad()
def qk_scale_debug_string(model):
    max_q_raw = 0.0
    max_k_raw = 0.0
    max_q_eff = 0.0
    max_k_eff = 0.0

    for blk in model.transformer.h:
        attn = blk.attn
        # if disabled qk_norm, continue
        if not hasattr(attn, "q_scale"):
            continue

        # raw learned params
        q_raw = attn.q_scale.max()
        k_raw = attn.k_scale.max()

        # effective values used in forward (after clamping)
        q_eff = torch.clamp(attn.q_scale, max=attn.qk_scale_max).max()
        k_eff = torch.clamp(attn.k_scale, max=attn.qk_scale_max).max()

        max_q_raw = max(max_q_raw, float(q_raw))
        max_k_raw = max(max_k_raw, float(k_raw))
        max_q_eff = max(max_q_eff, float(q_eff))
        max_k_eff = max(max_k_eff, float(k_eff))

    return (
        f"max q_scale raw/eff: {max_q_raw:.4f}/{max_q_eff:.4f} | "
        f"max k_scale raw/eff: {max_k_raw:.4f}/{max_k_eff:.4f}"
    )

################################################
#                Self Attention                #
################################################
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        assert 1 <= self.n_kv_heads <= self.n_heads, "n_kv_heads must be in [1, n_heads]"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA/MQA"
        self.head_size = config.d_model // config.n_heads
        self.q_heads_per_kv_head = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_size, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_size, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_size, bias=config.use_bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        self.rotary = Rotary(self.head_size, config.rope_base_theta, config.max_seq_len) if config.pos_encoding_type.lower() == "rope" else None
        
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        self.qk_scale_max = config.qk_scale_max
        self.qk_eps = config.qk_eps
        if self.use_qk_norm:
            # q_scale and k_scale are learned per-head scalars, like LayerNorm’s γ, that let 
            # the model adjust magnitude after normalization
            # They end up controlling the temperature of attention logits
            # How this changes the attention logits:
            # SDPA’s default scaling divides by sqrt(head_size)
            # -> L2 variant (unit-norm Q and K)
            #   Due to ||q'||_2 = ||k'||_2 = 1 (via product with q' and k's gains, i.e, 1/their L2 norm),
            #   (attention) logits = (q' · k') / sqrt(head_size) = 
            #   = (q_scale * k_scale * cosθ) / sqrt(head_size)
            #   i.e., the learned product (q_scale * k_scale) is a per-head temperature multiplier
            #   ---
            #   recalling that q' ⋅ k' = ||q'||_2 ||k'||_2 cosθ
            #   so if you normalize Q and K (gain does that), you get 
            #   ||q'||_2 = ||k'||_2 = 1 if L2 (or = sqrt(head_size) if RMS)
            #   Then the attention logit becomes proportional to cosθ
            #   ---
            # -> RMS variant (unit-RMS Q and K)
            #   because RMS(q) = 1 => L2Norm(q) = sqrt(head_size), we get:
            #   (attention) logits = (q' · k') / sqrt(head_size) =
            #   = (q_scale * sqrt(head_size) * k_scale * sqrt(head_size) * cosθ) / sqrt(head_size) =
            #   = (q_scale * k_scale * sqrt(head_size) * cosθ)
            #   i.e., logits are roughly bounded by sqrt(head_size) * the learned product * [-1, 1]
            #   ---
            #   recalling that RMS(q) = sqrt(1/head_size * sum(q_i^2)) =
            #   = sqrt(1/head_size) * sqrt(sum(q_i^2)) = 
            #   = sqrt(1/head_size) * L2 = 1/sqrt(head_size) * L2
            #   ---
            self.q_scale = nn.Parameter(torch.full((1, self.n_heads, 1, 1), config.qk_scale_init))
            self.k_scale = nn.Parameter(torch.full((1, self.n_kv_heads, 1, 1), config.qk_scale_init))
            
        self.is_causal = config.is_causal
        self.use_flex_attention = config.use_flex_attention
        self.use_doc_masking = config.use_doc_masking
        self.use_sliding_window_attention = config.use_sliding_window_attention
        self.sliding_window_size = config.sliding_window_size
        self.use_attn_logit_softcapping = config.use_attn_logit_softcapping
        self.attn_logit_softcap = config.attn_logit_softcap

        if (self.use_doc_masking or self.use_sliding_window_attention or self.use_attn_logit_softcapping):
            assert self.use_flex_attention, "Document masking / SWA / attention soft-capping require use_flex_attention=True"
        if self.use_sliding_window_attention:
            assert self.sliding_window_size > 0, "sliding_window_size must be > 0 with use_sliding_window_attention=True"

        self.tanh_backend = config.tanh_backend
        self._score_mod = generate_tanh_softcap(self.attn_logit_softcap, backend=self.tanh_backend) if self.use_attn_logit_softcapping else None

    def forward(self, x, flex_attn_block_mask=None, sdpa_attn_mask=None):
        _, seq_len, d_model = x.size()

        q = self.q_proj(x).view(x.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(x.size(0), seq_len, self.n_kv_heads, self.head_size).transpose(1, 2)

        if self.rotary is not None:
            # apply RoPE
            cos, sin = self.rotary.get_cos_sin(seq_len, q.dtype, q.device)
            q, k = apply_rotation(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            # expand KV to heads if needed
            k = k.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)
            v = v.unsqueeze(2).expand(x.size(0), self.n_kv_heads, self.q_heads_per_kv_head, seq_len, self.head_size).reshape(x.size(0), self.n_heads, seq_len, self.head_size)

        if self.use_qk_norm:
            if self.qk_norm_type == "rms":
                # 1 / RMS, i.e.,
                # 1 / RMS(q) = 1 / sqrt(mean(q_i^2)) = 1 / (sqrt(1/head_size * sum(q_i^2))) =
                # = sqrt(head_size) / sqrt(mean(q_i^2))
                # multiplying by these will make each query / key vector unit-RMS
                q_gain = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.qk_eps)
            # l2
            else:
                # 1 / L2, i.e.,
                # 1 / L2(q) = 1 / sqrt(sum(q_i^2))
                # multiplying by these will make each query / key vector unit-length
                q_gain = torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)
                k_gain = torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + self.qk_eps)

            # clamp scales
            q_scale = torch.clamp(self.q_scale, max=self.qk_scale_max)
            k_scale = torch.clamp(self.k_scale, max=self.qk_scale_max)

            # if we expanded K/V from n_kv_heads -> n_heads, do the same for k_scale
            if self.n_kv_heads != self.n_heads:
                # replicate per-kv scale across the grouped query heads
                k_scale = k_scale.repeat(1, self.q_heads_per_kv_head, 1, 1)

            # keep numerics tidy under autocast
            q_gain = q_gain.to(dtype=q.dtype)
            k_gain = k_gain.to(dtype=k.dtype)
            q_scale = q_scale.to(dtype=q.dtype)
            k_scale = k_scale.to(dtype=k.dtype)

            # (gpu_batch_size, n_heads, seq_len, head_size) * (1, n_heads, 1, 1)
            # gain is the data-dependent inverse norm (L2 or RMS)
            q = q * q_gain * q_scale
            # (gpu_batch_size, n_heads, seq_len, head_size) * (1, n_heads, 1, 1)
            k = k * k_gain * k_scale

        # FlexAttention
        if self.use_flex_attention:
            y = flex_attention(q, k, v, score_mod=self._score_mod, block_mask=flex_attn_block_mask)
        # SDPA
        else:
            is_causal = self.is_causal and (sdpa_attn_mask is None)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_attn_mask, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(x.size(0), seq_len, d_model)
        y = self.c_proj(y)
        return y

################################################
#                     MLP                      #
################################################
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_name = config.activation.lower()

        if self.activation_name == "swiglu":
            # without gating:
            # the up projection is (d_model, up_proj_factor * d_model) with -if used- up_proj_factor * d_model biases
            # the down projection is (up_proj_factor * d_model, d_model) with -if used- d_model biases
            # with gating:
            # the first matrix needs to start as d_model -i.e., (d_model, X)- 
            # and the last matrix end as d_model -i.e., (X, d_model)
            # we also need to match X (dimension) in both matrices for @
            # and split the output of the first matrix in X/2 -i.e., (d_model, X/2)- for gating
            # so the 2 up_proj_factor * d_model dimensions without gating need to be split 
            # for gating into 3 of size 2 / 3 * up_proj_factor * d_model, resulting in:
            # 2 up projection matrices of size (d_model, 2 / 3 * up_proj_factor * d_model) each, with -if used- 2 * 2 / 3 * up_proj_factor * d_model total biases
            # 1 down projection of size (2 / 3 * up_proj_factor * d_model, d_model) with -if used- d_model biases
            # the total number of weight parameters matches, but, if used, biases -slightly- differ as
            # up_proj_factor * d_model + d_model != 2 * 2 / 3 * up_proj_factor * d_model + d_model
            self.hidden_dim = int(round((2.0/3.0) * config.up_proj_factor * config.d_model)) if config.use_fair_swiglu else config.up_proj_factor * config.d_model
            self.c_fc = nn.Linear(config.d_model, self.hidden_dim * 2, bias=config.use_bias)
            self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.use_bias)
        else:
            self.hidden_dim = config.up_proj_factor * config.d_model
            self.c_fc = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias)
            self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.use_bias)

    def forward(self, x):
        # Gaussian Error Linear Unit
        if self.activation_name == "gelu":
            x = F.gelu(self.c_fc(x))
        # Rectified Linear Unit
        elif self.activation_name == "relu":
            x = F.relu(self.c_fc(x))
        # Rectified Linear Unit^2
        elif self.activation_name == "relu2":
            x = F.relu(self.c_fc(x)) ** 2
        # Sigmoid Linear Unit or Swish
        elif self.activation_name == "silu":
            x = F.silu(self.c_fc(x))
        # Swish Gated Linear Unit
        elif self.activation_name == "swiglu":
            # split the tensor in two along the last dimension
            x1, x2 = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
        else:
            raise ValueError(f"unsupported activation function: {self.activation_name}")
        return self.c_proj(x)

################################################
#                    Block                     #
################################################
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.residual_scale = (2 * config.n_layers) ** -0.5

        if config.norm_type.lower() == "rms":
            self.ln_1 = nn.RMSNorm(config.d_model)
        else:
            self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttention(config)
        if config.norm_type.lower() == "rms":
            self.ln_2 = nn.RMSNorm(config.d_model)
        else:
            self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, flex_attn_block_mask=None, sdpa_attn_mask=None):
        x = x + self.residual_scale * self.attn(self.ln_1(x), flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask)
        x = x + self.residual_scale * self.mlp(self.ln_2(x))
        return x

################################################
#                     GPT                      #
################################################
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.pad_token_id = config.pad_token_id
        self.pos_encoding_type = config.pos_encoding_type.lower()
        self.optimizer_type = config.optimizer_type.lower()
        self.muon_lr_scale = config.muon_lr_scale
        self.muon_backend = config.muon_backend
        self.muon_backend_steps = config.muon_backend_steps
        self.muon_momentum = config.muon_momentum
        self.use_nesterov = config.use_nesterov
        # <-- Uncomment for gradient norm clipping -->
        # self.use_grad_norm_clipping = config.use_grad_norm_clipping
        # self.gradient_clipping_norm = config.gradient_clipping_norm
        # <-- Uncomment for gradient norm clipping -->
        self.use_qk_norm = config.use_qk_norm
        self.use_qk_debug_log = config.use_qk_debug_log
        # <-- Uncomment for logit soft-capping -->
        # self.use_lm_head_logit_softcapping = config.use_lm_head_logit_softcapping
        # self.lm_head_logit_softcap = config.lm_head_logit_softcap
        # self._logits_absmax_stats = None
        # <-- Uncomment for logit soft-capping -->
        self.is_causal = config.is_causal
        self.use_flex_attention = config.use_flex_attention
        self.use_doc_masking = config.use_doc_masking
        self.use_sliding_window_attention = config.use_sliding_window_attention
        self.sliding_window_size = config.sliding_window_size
        self.use_attn_logit_softcapping = config.use_attn_logit_softcapping
        self.attn_logit_softcap = config.attn_logit_softcap
        self.flex_block_size = config.flex_block_size

        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.d_model),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = nn.RMSNorm(config.d_model) if config.norm_type.lower() == "rms" else nn.LayerNorm(config.d_model),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.RMSNorm(config.d_model) if config.norm_type.lower() == "rms" else nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.use_tied_embeddings = config.use_tied_embeddings
        if self.use_tied_embeddings:
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

        # pick Muon params by name (2D weights, excluding embeddings and lm_head)
        muon_param_names = set()
        if self.optimizer_type == "muon":
            for pn, p in param_dict.items():
                if p.dim() == 2 and ("wte" not in pn) and ("lm_head" not in pn):
                    muon_param_names.add(pn)

        # AdamW gets everything else, split by decay rule
        decay_params = [p for n, p in param_dict.items() if (n not in muon_param_names) and (p.dim() >= 2)]
        nodecay_params = [p for n, p in param_dict.items() if (n not in muon_param_names) and (p.dim() < 2)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            messages = [
                f"num decayed parameter tensors (AdamW): {len(decay_params)}, with {num_decay_params:,} parameters",
                f"num non-decayed parameter tensors (AdamW): {len(nodecay_params)}, with {num_nodecay_params:,} parameters",
            ]
            if self.optimizer_type == "muon":
                muon_params_count = sum(param_dict[n].numel() for n in muon_param_names)
                messages.append(f"num Muon parameter tensors: {len(muon_param_names)}, with {muon_params_count:,} parameters")
            for message in messages:
                print(message)
                log_buffer.append(message)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            message = f"using fused AdamW: {use_fused}"
            print(message)
            log_buffer.append(message)

        adamw = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)

        # if not using Muon, still return a dict for consistency with the training loop
        if self.optimizer_type != "muon":
            return {"adamw": adamw}

        # build Muon over its selected params
        muon_params = [param_dict[n] for n in muon_param_names]
        muon = Muon(
            muon_params,
            lr=lr * self.muon_lr_scale,
            momentum=self.muon_momentum,
            nesterov=self.use_nesterov,
            backend=self.muon_backend,
            backend_steps=self.muon_backend_steps,
        )
        return {"adamw": adamw, "muon": muon}
    
    def _build_flex_attn_block_mask(self, attn_mask, document_ids, gpu_batch_size, seq_len, device, ignore_doc_mask=False):
        # Attention builds, per sequence, a matrix of shape (seq_len, seq_len) when it does Q @ K^T
        # Splitting into 128x128 blocks (if block_size=128), each a 128×128 submatrix of 128 rows
        # (queries) × 128 columns (keys), we get:
        #                                       ┌───────────┬───────────┬───────────┬───────────┐
        #   queries 0-127                       │   (0,0)   │   (0,1)   │   (0,2)   │   (0,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        #   queries 128-255                     │   (1,0)   │   (1,1)   │   (1,2)   │   (1,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        #   ...                                 │   (2,0)   │   (2,1)   │   (2,2)   │   (2,3)   │
        #                                       ├───────────┼───────────┼───────────┼───────────┤
        # queries (seq_len-128)-(seq_len-1)     │   (3,0)   │   (3,1)   │   (3,2)   │   (3,3)   │
        #                                       └───────────┴───────────┴───────────┴───────────┘
        #                                        keys 0-127  keys 128-255    ...     keys (seq_len-128)-(seq_len-1)
        
        # In other words, (seq_len, seq_len) → (num_blocks_per_axis, num_blocks_per_axis)
        # The goal is to skip applying a mask to the full matrix, and prematurely
        # determine which blocks of 128x128 can compute full attention, 
        # which need partial masking and which can be dropped entirely
        
        # - full: every (q,k) pair in the tile is allowed → we can skip pointwise masking on this block
        # - partial: some (q,k) pairs are allowed → we must run pointwise masking on this block
        # - none: no (q,k) pairs are allowed → we can drop the block entirely

        # Now, we have various causes of masking:
        # is_causal: q shouldn't look into the future (we allow q >= k, prevent k > q)
        # use_doc_masking: q and k must share the same doc id (to avoid looking at keys from other docs)
        # attn_mask: both tokens must be valid (which, in our case, means both not being padding)
 
        # The speedup comes when we do not call create_block_mask at all and instead hand FlexAttention 
        # a BlockMask that already says, per query block, which KV blocks are:
        # - full-attn (no pointwise mask needed → mask_mod is skipped)
        # - partial (pointwise mask_mod still runs, but only on those tiles)

        block_size = self.flex_block_size
        assert seq_len % block_size == 0, f"seq_len {seq_len} must be a multiple of flex_block_size {block_size}"

        sliding_enabled = self.use_sliding_window_attention and (self.sliding_window_size > 0)
        window_size = self.sliding_window_size

        # per token static (i.e., reused across training steps) steps mask
        def flex_static_mask_mod(batch_index, head_index, query_index, key_index):
            # causal mask
            if self.is_causal:
                keep_flag = query_index >= key_index
            else:
                keep_flag = torch.ones((), dtype=torch.bool, device=device)
            # SWA mask
            if sliding_enabled:
                if self.is_causal:
                    keep_flag = keep_flag & ((query_index - key_index) <= window_size)
                else:
                    keep_flag = keep_flag & (torch.abs(query_index - key_index) <= window_size)
            return keep_flag
        mask_mod = flex_static_mask_mod

        # per token dynamic (i.e., can change across training steps) masks
        if attn_mask is not None:
            # custom (padding) mask
            def valid_mask_mod(batch_index, head_index, query_index, key_index):
                return attn_mask[batch_index, query_index] & attn_mask[batch_index, key_index]
            mask_mod = and_masks(mask_mod, valid_mask_mod)
        if self.use_doc_masking and not ignore_doc_mask and (document_ids is not None):
            # doc mask
            def same_doc_mod(batch_index, head_index, query_index, key_index):
                return document_ids[batch_index, query_index] == document_ids[batch_index, key_index]
            mask_mod = and_masks(mask_mod, same_doc_mod)

        # tiny bit faster path: supports gpu_batch_size >= 1, but currently needs is_causal, plus used 
        # only for training to avoid hitting re-compilation
        # Builds full/partial blocks up front, uses from_kv_blocks to skip applying mask_mod to certain blocks
        # -- Start of (modified) source: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/071225_BosAlign/c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt --
        if self.is_causal and self.training:
            # there are (seq_len // block_size)^2 total blocks, and seq_len // block_size per 
            # axis (i.e., per row or column), e.g., 1024 // 128 = 8 blocks per axis
            num_blocks_per_axis = seq_len // block_size
            # e.g., 0, 1,..., 7
            block_indices = torch.arange(num_blocks_per_axis, dtype=torch.int32, device=device)

            # Each block covers token indices:
            # for queries: q ∈ [i*block_size, (i+1)*block_size-1] with i=0..num_blocks_per_axis-1
            # for keys:    k ∈ [j*block_size, (j+1)*block_size-1] with j=0..num_blocks_per_axis-1
            # and we will compute 2 variables for each of causal, doc, and attn_mask masks:
            # - which blocks are fully dense or valid, i.e., all (q, k) pairs are valid
            # - which blocks are sparse, i.e., some (q, k) pairs are valid, yet not all

            # Starting with causality:
            # block_indices is e.g. (8,), and inserting a length-1 dimension at the end,
            # via block_indices[:, None], we make it (8,1), which makes the operation:
            # block_indices[:, None] >= block_indices broadcast (8,) into (1,8)
            # to result in (8,1) op (1,8) → an (8,8) or (num_blocks_per_axis, num_blocks_per_axis) tensor;
            # to simplify, assuming seq_len 8, and block_size 2 (num_blocks_per_axis=4), we have:
            # [[True,  False, False, False],
            #  [True,  True,  False, False],
            #  [True,  True,  True,  False],
            #  [True,  True,  True,  True]]
            # with the tensor having True values for blocks at least partially (causally) valid
            causal_partially_valid_blocks_2d = block_indices[:, None] >= block_indices
            # on the other hand, if we are more strict and all True values are blocks fully (causally) valid
            # (which excludes the diagonal since some elements have q < k), we have:
            # [[False, False, False, False],
            #  [True,  False, False, False],
            #  [True,  True,  False, False],
            #  [True,  True,  True,  False]]
            causal_fully_valid_blocks_2d = block_indices[:, None] > block_indices
            # broadcast the 2D pattern across batch: (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis)
            causal_partially_valid_blocks = causal_partially_valid_blocks_2d.expand(gpu_batch_size, -1, -1)
            causal_fully_valid_blocks     = causal_fully_valid_blocks_2d.expand(gpu_batch_size, -1, -1)

            # Moving on to doc ids:
            # if at least one (q, k) pair in the block share the same doc, it is partially (doc-wise) valid
            # if all (q, k) pairs in the block share the same doc, it is fully (doc-wise) valid
            if self.use_doc_masking and not ignore_doc_mask and (document_ids is not None):
                # take each sequence in the batch independently
                # e.g., for one sequence in the batch:
                # document_ids[b] = [0,0, 0,1, 1,1, 2,2]
                # blocks [0,0] | [0,1] | [1,1] | [2,2]
                # document_ids has shape (gpu_batch_size, seq_len); here, (B, seq_len)
                # so for one sequence we would take seq_len, e.g.,
                # [0, 0, 0, 1, 1, 1, 2, 2]
                docs = document_ids.to(device=device)
                # and then the lowest and highest doc id in each block_size, e.g., if block size was 2:
                # [0, 0, 0, 1, 1, 1, 2, 2] → view → [[0, 0], [0, 1], [1, 1], [2, 2]]
                # the lowest doc id per block would be [0, 0, 1, 2], with size (num_blocks_per_axis,)
                docs_view = docs.view(gpu_batch_size, num_blocks_per_axis, block_size)
                docs_low  = docs_view[:, :,  0].contiguous()
                # and the highest doc id per block would be [0, 1, 1, 2], with size (num_blocks_per_axis,)
                docs_high = docs_view[:, :, -1].contiguous()
                # (num_blocks_per_axis, 1) op (num_blocks_per_axis,) → 
                # (num_blocks_per_axis, 1) op (num_blocks_per_axis,) left padded to (1, num_blocks_per_axis) → 
                # (num_blocks_per_axis, num_blocks_per_axis), i.e., for one sequence:
                # -------------------------------------------------------------------------
                # (docs_low[:, None] <= docs_high):
                # [[True,  True,  True,  True],  because 0 <= 0,  0 <= 1,  0 <= 1,   0 <= 2
                # [ True,  True,  True,  True],  because 0 <= 0,  0 <= 1,  0 <= 1,   0 <= 2
                # [ False, True,  True,  True],  because 1 !<= 0, 1 <= 1,  1 <= 1,   1 <= 2
                # [ False, False, False, True]], because 2 !<= 0, 2 !<= 1, 2 !<= 1,  2 <= 2
                # -------------------------------------------------------------------------
                # (docs_high[:, None] >= docs_low):
                # [[True, True, False, False],   because 0 >= 0,  0 >= 0,  0 !>= 1,  0 !>= 2
                #  [True, True, True,  False],   because 1 >= 0,  1 >= 0,  1 >= 1,   1 !>= 2
                #  [True, True, True,  False],   because 1 >= 0,  1 >= 0,  1 >= 1,   1 !>= 2
                #  [True, True, True,  True]]    because 2 >= 0,  2 >= 0,  2 >= 1,   2 >= 2
                # -------------------------------------------------------------------------
                # (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low):
                # [[True,  True,  False, False], because block 0's doc ids at least partially overlap with 0 and 1
                #  [True,  True,  True,  False], because block 1's doc ids at least partially overlap with 0, 1, 2
                #  [False, True,  True,  False], because block 2's doc ids at least partially overlap with 1 and 2
                #  [False, False, False, True]]  because block 3's doc ids at least partially overlap with 3
                doc_partially_valid_blocks = (
                    (docs_low[:, :, None] <= docs_high[:, None, :]) &
                    (docs_high[:, :, None] >= docs_low[:, None, :])
                )
                # while for full (q,k) doc validity, we get:
                # -------------------------------------------------------------------------
                # (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
                # [[True,  False, False, False],  because all block 0's (q,k) share the same doc id in block 0
                #  [False, False, False, False],  because all block 1's (q,k) share the same doc id in *no block*
                #  [False, False, True,  False],  because all block 2's (q,k) share the same doc id in block 2
                #  [False, False, False, True]]   because all block 3's (q,k) share the same doc id in block 3
                # recalling that:
                # block 0 [0, 0]
                # block 1 [0, 1]
                # block 2 [1, 1]
                # block 3 [2, 2]
                doc_fully_valid_blocks = (
                    (docs_low[:, :, None] == docs_high[:, None, :]) & 
                    (docs_high[:, :, None] == docs_low[:, None, :])
                )
            else:
                doc_partially_valid_blocks = torch.ones(
                    (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis), dtype=torch.bool, device=device
                )
                doc_fully_valid_blocks = doc_partially_valid_blocks

            # Finally, for padding -or other custom attn_mask-:
            if attn_mask is not None:
                # e.g., for one sequence in the batch: [True, True, True, True, True, False, False, False]
                valid = attn_mask.to(device=device, dtype=torch.bool)
                # viewing the valid tokens as [[True, True], [True, True], [True, False], [False, False]] per sequence
                valid_blocks = valid.view(gpu_batch_size, num_blocks_per_axis, block_size)
                # q any indicates whether there is any query valid in the block:
                # [True, True, True, False] per sequence
                q_any = valid_blocks.any(dim=2)
                # k any indicates whether there is any key valid in the block:
                # [True, True, True, False] per sequence
                k_any = valid_blocks.any(dim=2)
                # in this case, q_any == k_any
                # q all indicates whether all queries are valid in the block:
                # [True, True, False, False] per sequence
                q_all = valid_blocks.all(dim=2)
                # k all indicates whether all keys are valid in the block:
                # [True, True, False, False] per sequence
                k_all = valid_blocks.all(dim=2)
                # in this case, q_all == k_all
                # -------------------------------------------------------------------------
                # q_any[:, :, None] & k_any[:, None, :]:
                # [[True,  True,   True,  False],
                #  [True,  True,   True,  False],
                #  [True,  True,   True,  False],
                #  [False, False,  False, False]] per sequence
                attn_mask_blockmask_any = q_any[:, :, None] & k_any[:, None, :]
                # -------------------------------------------------------------------------
                # q_all[:, :, None] & k_all[:, None, :]:
                # [[True,  True,  False,  False],
                #  [True,  True,  False,  False],
                #  [False, False, False,  False],
                #  [False, False, False,  False]] per sequence
                attn_mask_blockmask_all = q_all[:, :, None] & k_all[:, None, :]
                # -------------------------------------------------------------------------
            else:
                attn_mask_blockmask_any = torch.ones(
                    (gpu_batch_size, num_blocks_per_axis, num_blocks_per_axis), dtype=torch.bool, device=device
                )
                attn_mask_blockmask_all = attn_mask_blockmask_any

            # combining the constraints:
            # blockmask_any → tiles that may have some valid (q,k) pairs and must run mask_mod pointwise, e.g.,
            # [[True,  False, False, False],
            #  [True,  True,  False, False],
            #  [False, True,  True,  False],
            #  [False, False, False, False]] per sequence
            blockmask_any = causal_partially_valid_blocks & doc_partially_valid_blocks & attn_mask_blockmask_any
            # blockmask_all → tiles that are guaranteed fully dense and can skip mask_mod, e.g.,
            # [[False, False, False, False],
            #  [False, False, False, False],
            #  [False, False, False, False],
            #  [False, False, False, False]] per sequence
            blockmask_all = causal_fully_valid_blocks & doc_fully_valid_blocks & attn_mask_blockmask_all

            def map_to_ordered(blockmask_full_or_partial):
                # map (num_blocks_per_axis, num_blocks_per_axis) into its ordered representation
                # and return blockmask and sort-indices, both as (B?, H?, num_query_blocks)
                num_valid_kv_blocks_per_q_block = blockmask_full_or_partial.sum(dim=-1, dtype=torch.int32)
                # descending=False argsort → False before True
                # stable=True argsort      → keep order of appearance, e.g., [True, True] → [0, 1] (and not [1, 0])
                # row 0 of blockmask_any & ~blockmask_all is [True,  False, False, False] →
                # argsort(False-first) gives indices [1, 2, 3, 0] → flipped is [0, 3, 2, 1] (True-first)
                # row 1 of blockmask_any & ~blockmask_all is [True,  True,  False, False]
                # argsort(False-first) gives indices [2, 3, 0, 1] → flipped is [1, 0, 3, 2] (True-first)
                # row 2 of blockmask_any & ~blockmask_all is [False, True,  True,  False]
                # argsort(False-first) gives indices [0, 3, 1, 2] → flipped is [2, 1, 3, 0] (True-first)
                # row 3 of blockmask_any & ~blockmask_all is [False, False, False, False]
                # argsort(False-first) gives indices [0, 1, 2, 3] → flipped is [3, 2, 1, 0] (True-first)
                fully_valid_kv_indices = blockmask_full_or_partial.argsort(dim=-1, descending=False, 
                                                                        stable=True).flip(-1).to(torch.int32)
                # BlockMask expects (B?, H?, num_query_blocks), so we insert two leading dims; 
                # and flip() may produce a non-contiguous view, so we call contiguous(), resulting in, e.g.,
                # -------------------------------------------------------------------------
                # (given) blockmask_any & ~blockmask_all:
                # -------------------------------------------------------------------------
                # [[True,  False, False, False],
                #  [True,  True,  False, False],
                #  [False, True,  True,  False],
                #  [False, False, False, False]] per sequence
                # -------------------------------------------------------------------------
                # num_only_partially_valid_kv_blocks:
                # -------------------------------------------------------------------------
                # [[[1, 2, 2, 0]]] because row 0: 1 valid block; rows 1-2: 2; row 3: 0 (per sequence)
                # -------------------------------------------------------------------------
                # only_partial_kv_indices and while other orders are possible,
                # this keeps elements closest to the diagonal,
                # e.g., if SWA and window_size_blocks=1,
                # index 0 is kept for the first row (because it is the first column and we keep only 1 block)
                # index 1 is kept for the second row (because it is the first column and we keep only 1 block)
                # index 2 is kept for the third row (because it is the first column and we keep only 1 block)
                # index 3 is kept for the fourth row (because it is the first column and we keep only 1 block)
                # NOTE:
                # Despite not being used in this fast path (the path enforces is_causal), it must be noted:
                # This order fails if not is_causal because it keeps last valid block(s), which if is_causal 
                # is correct because future blocks are False, keeping the last *starting from the diagonal*, 
                # but would take the last column(s) always if not is_causal)
                # -------------------------------------------------------------------------
                # [[[0,    3,     2,     1],
                #   [1,    0,     3,     2],
                #   [2,    1,     3,     0],
                #   [3,    2,     1,     0]]] per sequence
                return (num_valid_kv_blocks_per_q_block[:, None, :].contiguous(), 
                        fully_valid_kv_indices[:, None, :].contiguous())

            num_only_partially_valid_kv_blocks, only_partial_kv_indices = map_to_ordered(blockmask_any & ~blockmask_all)
            num_fully_valid_kv_blocks, fully_valid_kv_indices = map_to_ordered(blockmask_all)

            # when using block_size, window_size is not exact (e.g., 2 full blocks + 1 partial block attended to
            # -in general, for block_size = 128 and window_size = 384 (window_size_blocks = 3)
            # excluding doc and other attn masks, that means 128 + 128 + 64 attended to tokens (64 tokens short))-
            if sliding_enabled:
                window_size_blocks = int(math.ceil(window_size / block_size))
            else:
                window_size_blocks = num_blocks_per_axis

            # is_causal → the diagonal block is partial
            # therefore, always exclude one full block per row to make room for the diagonal block (in other words,
            # start counting window_size from the diagonal to the left)
            clamped_num_fully_valid_kv_blocks = torch.clamp_max(num_fully_valid_kv_blocks, window_size_blocks - 1)
            clamped_num_only_partially_valid_kv_blocks = torch.clamp_max(num_only_partially_valid_kv_blocks, 
                                                                        window_size_blocks - clamped_num_fully_valid_kv_blocks)
            
            # https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
            return BlockMask.from_kv_blocks(
                # -------------------------------------------------------------------------
                # which KV tiles to visit for each query tile
                # -------------------------------------------------------------------------
                # kv_num_blocks (Tensor) – Number of kv_blocks in each Q_BLOCK_SIZE row tile
                clamped_num_only_partially_valid_kv_blocks,
                # kv_indices (Tensor) – Indices of key-value blocks in each Q_BLOCK_SIZE row tile
                only_partial_kv_indices,
                # -------------------------------------------------------------------------
                # subset of the above that are known to be fully dense 
                # (every (q,k) pair in the tile is valid)
                # -------------------------------------------------------------------------
                # full_kv_num_blocks (Optional[Tensor]) – Number of full kv_blocks 
                # in each Q_BLOCK_SIZE row tile
                clamped_num_fully_valid_kv_blocks,
                # full_kv_indices (Optional[Tensor]) – Indices of full key-value blocks 
                # in each Q_BLOCK_SIZE row tile
                fully_valid_kv_indices,
                # -------------------------------------------------------------------------
                # per-element boolean function (b,h,q_idx,k_idx) -> bool that's applied 
                # inside partial tiles to zero out disallowed (q,k) pairs
                # (required if is_causal due to partial diagonal tiles)
                # -------------------------------------------------------------------------
                # BLOCK_SIZE (Union[int, tuple[int, int]]) – Size of KV_BLOCK_SIZE x Q_BLOCK_SIZE tiles
                BLOCK_SIZE=block_size,
                # mask_mod (Optional[Callable]) – Function to modify the mask
                mask_mod=mask_mod,
            )
        # -- End of (modified) source: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/071225_BosAlign/c1fd8a38-bb9f-45c4-8af0-d37f70c993f3.txt --

        # old path (for is_causal=False, validation, sampling, and hellaswag)
        return create_block_mask(
            mask_mod,
            B=gpu_batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            BLOCK_SIZE=block_size,
            _compile=self.training
        )

    def _normalize_attn_mask(self, attn_mask, gpu_batch_size, seq_len, device, for_flex):
        if attn_mask is None:
            return None
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        if attn_mask.dim() != 2:
            raise ValueError(f"attn_mask must be 2D [gpu_batch_size, seq_len], got shape {attn_mask.shape}")
        if attn_mask.size(0) != gpu_batch_size or attn_mask.size(1) != seq_len:
            raise ValueError(f"attn_mask shape mismatch: expected [{gpu_batch_size}, {seq_len}], got {list(attn_mask.shape)}")

        attn_mask = attn_mask.to(device=device, non_blocking=True)

        if for_flex:
            return attn_mask

        # [gpu_batch_size, 1, seq_len, 1]
        q_valid = attn_mask[:, None, :, None]
        # [gpu_batch_size, 1, 1, seq_len]
        k_valid = attn_mask[:, None, None, :]
        # [gpu_batch_size, 1, seq_len, seq_len]
        allow_mask_4d = q_valid & k_valid
        if self.is_causal:
            allow_causal = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device).tril()
            allow_mask_4d = allow_mask_4d & allow_causal
        return allow_mask_4d

    def forward(self, indices, targets=None, attn_mask=None, ignore_doc_mask=False, document_ids=None):
        # ignore_doc_mask to avoid using it in:
        # val, sampling, hellaswag
        # even if used in train
        gpu_batch_size, seq_len = indices.size()
        te = self.transformer.wte(indices)

        # if RoPE or NoPE, rotate later in attention or do nothing, respectively
        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            x = te
        else:
            pe = self.transformer.wpe(torch.arange(0, seq_len, dtype=torch.long, device=indices.device))
            x = te + pe

        attn_mask = self._normalize_attn_mask(
            attn_mask=attn_mask,
            gpu_batch_size=gpu_batch_size,
            seq_len=seq_len,
            device=indices.device,
            for_flex=self.use_flex_attention,
        )

        if self.use_flex_attention:
            flex_attn_block_mask = self._build_flex_attn_block_mask(
                attn_mask=attn_mask,
                document_ids=document_ids,
                gpu_batch_size=gpu_batch_size,
                seq_len=seq_len,
                device=indices.device,
                ignore_doc_mask=ignore_doc_mask
            )
            sdpa_attn_mask = None
        else:
            flex_attn_block_mask = None
            sdpa_attn_mask = attn_mask

        for block in self.transformer.h:
            x = block(x, flex_attn_block_mask=flex_attn_block_mask, sdpa_attn_mask=sdpa_attn_mask)
        x = self.transformer.ln_f(x)

        # [gpu_batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        if targets is not None:
            loss_mask = (targets != self.pad_token_id)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            n_non_masked_tokens = loss_mask.sum()
            sum_non_masked_loss_tokens = (loss * loss_mask.view(-1)).sum()
            loss = sum_non_masked_loss_tokens / n_non_masked_tokens
        else:
            loss = None

        return logits, loss

################################################
#                Learning Rate                 #
################################################
def get_lr(step):
    if step < warmup_steps:
        return (step + 1) * max_lr / warmup_steps
    elif step >= warmup_and_cosine_steps:
        lr = min_lr_after_warmup
    else:
        coeff = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (warmup_and_cosine_steps - warmup_steps)))
        lr = min_lr_after_warmup + coeff * (max_lr - min_lr_after_warmup)
    if hard_min_lr > 0:
        lr = max(lr, hard_min_lr)
    return lr

################################################
#                   Sampling                   #
################################################
def get_sample_token_count(step, base=5, step_interval=1000, max_tokens=50):
    return min(base + (step // step_interval) * base, max_tokens)

def sample(sample_sequences, max_new_tokens=5, temperature=1.0, top_k=None, top_p=None):
    gpt_model.eval()
    with torch.inference_mode():
        # convert all prompts to ids,
        # e.g., 'AGI is' to [4760, 40, 318]
        # 'AGI is not' to [4760, 40, 318, 407]
        # and truncate them leaving max_new_tokens for generation
        max_allowed_input_len = raw_gpt_model.max_seq_len - max_new_tokens
        initial_input_ids_list = [
            tokenizer.encode(sequence)[:max_allowed_input_len] 
            for sequence in sample_sequences
        ]

        # find the max input length (in ids) in this batch
        max_input_len = max(len(ids) for ids in initial_input_ids_list)

        # get a length cap to pre-allocate the tensor while not needing to go to max_seq_len
        alloc_len = max_input_len + max_new_tokens

        # rounding up to a nice multiple for better tensor cores / GPU efficiency
        round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
        alloc_len = math.ceil(alloc_len / round_multiple) * round_multiple
        alloc_len = min(alloc_len, raw_gpt_model.max_seq_len)

        # pre-allocate tensor of size len(sample_sequences), alloc_len and fill with padding 
        # to significanly boost performance (versus a new tensor size every generation)!
        generated_sequences = torch.full(
            (len(sample_sequences), alloc_len),
            raw_gpt_model.pad_token_id,
            dtype=torch.long,
            device=device
        )

        # then replace the first padding tokens of each pre-allocated sequence with their original tokens
        for i, seq_ids in enumerate(initial_input_ids_list):
            generated_sequences[i, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)

        # track actual (non-padding) sequence lengths
        actual_sequence_lengths = torch.tensor([len(seq) for seq in initial_input_ids_list], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # then, for each new token to generate, update the mask by effectively comparing if 0, ..., alloc_len < the non-padded length for each sequence
            # using unsqueeze(0) to add a new dimension of size 1 at the beginning to give size (1, alloc_len)
            # and unsqueeze(1) to add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
            # attn_mask = torch.arange(alloc_len, device=device).unsqueeze(0) < actual_sequence_lengths.unsqueeze(1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # and predict, resulting in (len(sample_sequences), seq_len, vocab_size)
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                logits, _ = raw_gpt_model(generated_sequences, attn_mask=None, ignore_doc_mask=True, document_ids=None)
            
            # of which we take the vocab_size values for each sequence's continuation to the last non-pad token,
            # resulting in len(sample_sequences), vocab_size
            last_logits = logits[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths - 1, :]

            if temperature == 0.0:
                # then if temperature is 0, we cannot divide by 0, so take the max logit from vocab_size
                next_token_ids = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # else, divide by the temperature
                temp_adjusted_logits = last_logits / temperature

                # apply top_k filtering
                if top_k is not None and top_k > 0:
                    # topk returns the k largest elements of the given input tensor along a given dimension
                    # resulting in len(sample_sequences), vocab_size
                    sorted_values, _ = torch.topk(temp_adjusted_logits, min(top_k, temp_adjusted_logits.size(-1)), dim=-1, largest=True, sorted=True)
                    # get the k-th largest logit of each sequence (i.e., last in sorted_values)
                    # and add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
                    sequences_k_th_logit = sorted_values[:, -1].unsqueeze(1)
                    # mask out everything less than the top_k logit
                    temp_adjusted_logits[temp_adjusted_logits < sequences_k_th_logit] = -float('Inf')

                # apply top_p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    # apply softmax to get probabilities for the continuation to the last non-pad token
                    # resulting in (still) len(sample_sequences), vocab_size
                    probs = F.softmax(temp_adjusted_logits, dim=-1)
                    # sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                    # obtain the cumulative sums of the probabilities sorted in descending order
                    # e.g., cumulative probs of [0.3, 0.6, 0.8, 1.0] for [0.3, 0.3, 0.2, 0.2]
                    # with size len(sample_sequences), vocab_size
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    if master_process:
                        message = "first 2 sequences cumulative probs:"
                        print(message)
                        log_buffer.append(message)
                        for i in range(min(2, cumulative_probs.size(0))):
                            values = cumulative_probs[i, :10]
                            message = f"\tseq {i}: {values.tolist()}"
                            print(message)
                            log_buffer.append(message)
                    # obtain the indices of logits to remove,
                    # initially excluding the logit that makes the cumulative sum match top_p,
                    # as we then shift to the right one position
                    # cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.75 would thus give (for a single sequence):
                    # [False, False, True, True] which shifted to the right is [False, False, False, True]
                    # while cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.8:
                    # [False, False, True, True] which shifted to the right is [False, False, False, True],
                    # being in both instances the smallest possible set that has at least top_p cumulative probability
                    sorted_indices_to_remove = cumulative_probs >= top_p
                    # shift the indices to the right one position
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    # replacing the last index value, now in the first position, to False
                    sorted_indices_to_remove[..., 0] = False
                    mask = sorted_indices_to_remove.to(torch.bool)

                    scatter_mask = torch.zeros_like(temp_adjusted_logits, dtype=torch.bool)
                    scatter_mask = scatter_mask.scatter(dim=-1, index=sorted_indices, src=mask)
                    if master_process:
                        for i in range(min(2, scatter_mask.size(0))):
                            masked_count = scatter_mask[i].sum().item()
                            total_count = scatter_mask.size(1)
                            #print(f"sequence {i} mask: {masked_count}/{total_count} logits masked")
                            if masked_count == total_count:
                                message = f"all logits masked for sequence {i}!"
                                print(message)
                                log_buffer.append(message)
                    temp_adjusted_logits = temp_adjusted_logits.masked_fill(scatter_mask, -float('inf'))
                
                probs = F.softmax(temp_adjusted_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            # replace the 'actual_sequence_length' position of each sequence with the selected token
            generated_sequences[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths] = next_token_ids[:, 0]
            # increase the actual, non-padded sequence endings
            actual_sequence_lengths += 1

        # and decode, ignoring above 50256
        local_messages = []
        for i in range(len(sample_sequences)):
            decoded = tokenizer.decode([
                token for token in generated_sequences[i, :actual_sequence_lengths[i]].tolist()
                if token < raw_gpt_model.pad_token_id])
            message = f"[rank {ddp_rank}] seq {i} >>> {decoded}"
            print(message)
            local_messages.append(message)

        all_messages = [None for _ in range(ddp_world_size)]
        dist.all_gather_object(all_messages, local_messages)

        if master_process:
            for rank_messages in all_messages:
                log_buffer.extend(rank_messages)
        dist.barrier()
    gpt_model.train()

################################################
#                   HellaSwag                  #
################################################
# ---------------------------------------------------------
# Normalizing token probabilities to account for sequence length and comparing continuations A-D has the issue of an 
# easy continuation achieving high scores because once the answer deviates, the continuation is easy to guess, e.g.,

# (extreme example)
# She is cooking dinner,
# A) but she would rather be dancing in LA with the bride's friends tonight -> hard to guess dancing, LA, bride's
# B) one two three four five six -> easy to guess continuation after one two
# Here, it would not matter B makes less sense than A, because once 'one two' are told to be the continuation
# to the model, it has no way to be surprised by 'three four five six'
# ---------------------------------------------------------

# ---------------------------------------------------------
# From Language Models are Few-Shot Learners
# GPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot setting, outperforming the
# 75.4% accuracy of a fine-tuned 1.5B parameter language model [ZHR+19] but still a fair amount lower than the overall
# SOTA of 85.6% achieved by the fine-tuned multi-task model ALUM.

# Zero-shot
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.7  | 43.6 |  51.0 | 54.7 |  62.8 |  67.4 | 70.9 | 78.9

# One-shot (we will evaluate one-shot, but with Answer: A, Answer: B, Answer: C, Answer: D token probabilities)
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.0  | 42.9 | 50.5  | 53.5 | 61.9  | 66.5  | 70.0 | 78.1

# ...
# ---------------------------------------------------------

# ---------------------------------------------------------
# The A, B, C, D answers are uniformly distributed:

# HellaSwag train split answer label distribution:
#   A: 9986 (25.02%)
#   B: 10031 (25.14%)
#   C: 9867 (24.73%)
#   D: 10021 (25.11%)

# HellaSwag validation split answer label distribution:
#   A: 2515 (25.04%)
#   B: 2485 (24.75%)
#   C: 2584 (25.73%)
#   D: 2458 (24.48%)

# which make even a model biased towards some token, e.g., A,
# as good as a random-guesser model
# ---------------------------------------------------------

# ---------------------------------------------------------
# The fact they are equi-probable allows us to evaluate performance by doing:

# batch sequence 1 passed to the model

# Sentence to pick the continuation of: A little girl in a room standing in front of some chairs is hitting a dora pinata. She hits it a few times and then its someone else's turn. an older girl
# Continuation options:
# A) with a pink apron including a brown jacket tries to stop her but she hits it again.
# B) comes into the room and gives her and the girl laughs while they continue to hit the pinata.
# C) is also there in the room playing and before long some other little girl is swinging her away and she hasn't seen anyone.
# D) hits it so hard that it doesn't break but the whole pinata falls down and the adults have to put it back up.
# Correct continuation: D)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: A)

# batch sequence 2 passed to the model

# ... (one-shot example)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: B)

# to evaluate the likelihood the model assigns to each continuation
# ---------------------------------------------------------

# ---------------------------------------------------------
# For efficiency, we could score the log-probability of:
# " a" (for A)
# " b" (for B)
# " c" (for C)
# " d" (for D)

# but we'd rely on these tokens existing on the tokenizer (could break if we change our tokenizer)
# plus why not add " a", " A", " first",  etc. -if existent-,, eventually deciding to pass 4 batch sequences
# with the continuations included, one per batch sequence, even if more
# wasteful compared to passing a single sequence per HellaSwag sentence
# ---------------------------------------------------------

# ---------------------------------------------------------
# For length-normalization (to avoid penalizing long sequences):
# In prob space, the product of probabilities needs to be raised to 1/len(tokens) -i.e., the geometric mean-,
# while in log-prob space the sum of log-probs needs to be divided by len(tokens),
# e.g., given seq_A with 1 token, seq_B with 3:

# P(seq_A)           = 0.5 (favored)                  | P(seq_B)           = 0.5 * 0.5 * 0.5 = 0.125
# norm(P(seq_A))     = (0.5) ^ 1/1 = 0.5 (fair)       | norm(P(seq_B))     = (0.5 * 0.5 * 0.5) ^ (1/3) = 0.5

# log-P(seq_A)       = −0.69314 (favored)             | log-P(seq_B)       = -2.07944
# norm(log-P(seq_A)) = −0.69314 / 1 = −0.69314 (fair) | norm(log-P(seq_B)) = (−0.69314 + −0.69314 + −0.69314) / 3 = −0.69314
# ---------------------------------------------------------

# ---------------------------------------------------------
# If using hard-targets (only one token is correct, all others are incorrect):

# average loss = (1/sequence length) * sum(-log(probability of correct token at each position))
# likelihood = product of probabilities
# normalized likelihood = product of probabilities ^ (1/sequence length)
# average log-likelihood = log ((product of probabilities) ^ (1/sequence length)) = (1/sequence length) * sum(log-probabilities)
# average negative log-likelihood (nll) = - (average log-likelihood)
# perplexity (ppl) = e ^ (average negative log-likelihood)
# log-ppl = average loss = average nll
# ppl = e ^ (log-ppl) = e ^ (average loss) = e ^ (nll)

# F.cross_entropy() calculates - F.log_softmax(), and then either it reduces to mean or we do so manually
# ---------------------------------------------------------

def evaluate_hellaswag_one_shot(hellaswag_examples_per_batch=16):
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 

    with torch.inference_mode():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))
            # if i == 0:
            #     for rank in range(ddp_world_size):
            #         dist.barrier()
            #         if ddp_rank == rank:
            #             print(f"[rank {ddp_rank}] first HellaSwag batch:")
            #             for j, ex in enumerate(current_examples):
            #                 print(f"[rank {ddp_rank}] example {j}:")
            #                 print(f"ctx: {ex['ctx']}")
            #                 print(f"endings:")
            #                 for k, ending in enumerate(ex['endings']):
            #                     print(f"\t{chr(ord('A') + k)}) {ending}")
            #                 print(f"label: {ex['label']}")
            #                 print("-" * 60)
            #         dist.barrier()

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_prefix_text_base = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                    "Continuation options:\n"
                    f"A) {continuations[0]}\n"
                    f"B) {continuations[1]}\n"
                    f"C) {continuations[2]}\n"
                    f"D) {continuations[3]}\n"
                    "Correct continuation: "
                )
                prefix_ids = tokenizer.encode(common_prefix_text_base)
                
                for j in range(4):
                    final_token_text = f"{chr(ord('A') + j)})"
                    final_token_ids = tokenizer.encode(final_token_text)
                    combined_seq_ids = prefix_ids + final_token_ids
                    prefix_lengths.append(len(prefix_ids))
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)

            # allocate to the actual max length in this pass (rounded for kernels, capped to model max)
            max_len_in_batch = max(len(s) for s in all_seq_ids) if all_seq_ids else 1
            round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
            alloc_len = min(max_model_seq_len, math.ceil(max_len_in_batch / round_multiple) * round_multiple)

            pre_allocated_input_ids = torch.full(
                (max_gpu_sequences_per_batch, alloc_len),
                raw_gpt_model.pad_token_id,
                dtype=torch.long,
                device=device
            )
            pre_allocated_attn_mask = torch.zeros(
                (max_gpu_sequences_per_batch, alloc_len),
                dtype=torch.bool,
                device=device
            )

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                logits, _ = raw_gpt_model(pre_allocated_input_ids, attn_mask=None, ignore_doc_mask=True)
            
            # log softmax: softmax followed by log (to sum log-probs vs. multiply low-value probs)
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                prefix_len = prefix_lengths[seq_idx]
                
                seq_log_prob = 0.0
                num_tokens = 0

                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_tokens += 1

                normalized_log_prob = seq_log_prob / num_tokens if num_tokens > 0 else 0.0

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    # convert local totals to tensors for sum reduction across ranks
    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy

def evaluate_hellaswag_standard(hellaswag_examples_per_batch=16):
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 
    
    with torch.inference_mode():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_context_text = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                )
                
                context_ids = tokenizer.encode(common_context_text)
                
                for j in range(4):
                    continuation_text = f" {continuations[j]}"
                    continuation_ids = tokenizer.encode(continuation_text)
                    
                    combined_seq_ids = context_ids + continuation_ids
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)
                    prefix_lengths.append(len(context_ids))

            # allocate to the actual max length in this pass (rounded for kernels, capped to model max)
            max_len_in_batch = max(len(s) for s in all_seq_ids) if all_seq_ids else 1
            round_multiple = (raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8)
            alloc_len = min(max_model_seq_len, math.ceil(max_len_in_batch / round_multiple) * round_multiple)

            pre_allocated_input_ids = torch.full(
                (max_gpu_sequences_per_batch, alloc_len),
                raw_gpt_model.pad_token_id,
                dtype=torch.long,
                device=device
            )
            pre_allocated_attn_mask = torch.zeros(
                (max_gpu_sequences_per_batch, alloc_len),
                dtype=torch.bool,
                device=device
            )

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # NOTE: for decoding and right padding, tokens cannot attend to padding anyway
                # so we can skip passing an attn_mask
                logits, _ = raw_gpt_model(pre_allocated_input_ids, attn_mask=None, ignore_doc_mask=True)
            
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                prefix_len = prefix_lengths[seq_idx]
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                
                seq_log_prob = 0.0
                num_continuation_tokens = 0
                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_continuation_tokens += 1
                
                normalized_log_prob = seq_log_prob / num_continuation_tokens

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy

################################################
#                Checkpointing                 #
################################################
def keep_latest_checkpoints(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)

    # group files by type
    model_files = sorted(
        [f for f in all_files if f.endswith(".safetensors")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    state_files = sorted(
        [f for f in all_files if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    # for every file that is past the last max_checkpoints_to_keep, remove it,
    # resulting in empty list if max_checkpoints_to_keep >= len(files),
    # and thus preserving all files in that case

    # prune excess model files
    for old_file in model_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed model: {path}"
        print(message)
        log_buffer.append(message)

    # prune excess state files
    for old_file in state_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed training state: {path}"
        print(message)
        log_buffer.append(message)

def save_checkpoint(step, torch_rng_state_cpu, torch_rng_state_cuda, val_loss, train_loss, train_tokens_processed,
                    total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
                    current_shard_idx, grad_accum_mini_steps_per_shard_counter, optimizers):

    # create the model path by adding its timestamp, and train and val losses, if available
    parts = [f"model", f"step_{step:07d}"]
    parts.append(f"val_{val_loss:.4f}")
    parts.append(f"train_{train_loss:.4f}")
    checkpoint_name = "_".join(parts) + ".safetensors"
    safetensors_path = os.path.join(checkpoint_dir, checkpoint_name)

    # save the model locally
    save_model(gpt_model.module, safetensors_path)
    message = f"saved model weights locally in: {safetensors_path}"
    print(message)
    log_buffer.append(message)

    # gather the training state
    training_state = {
        'step': step,
        'torch_rng_state_cpu': torch_rng_state_cpu,
        'torch_rng_state_cuda': torch_rng_state_cuda,
        'train_tokens_processed': train_tokens_processed,
        'total_train_t': total_train_t,
        'total_val_t': total_val_t,
        'total_sample_t': total_sample_t,
        'total_hellaswag_t': total_hellaswag_t,
        'total_t': total_t,
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'current_shard_idx': current_shard_idx,
        'grad_accum_mini_steps_per_shard_counter': grad_accum_mini_steps_per_shard_counter,
        'optimizer_state_dicts': {k: optimizer.state_dict() for k, optimizer in optimizers.items()}
    }

    # create the training state path
    state_path = os.path.join(checkpoint_dir, f"training_state_step_{step:07d}.pt")

    # save the training state locally
    torch.save(training_state, state_path)
    message = f"saved training state locally in: {state_path}"
    print(message)
    log_buffer.append(message)

    # retain last max_checkpoints_to_keep
    keep_latest_checkpoints(checkpoint_dir)

def load_checkpoint():
    model_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_checkpoint_path,
        token=hf_token,
        repo_type="model"
    )
    training_state_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_state_dict_path,
        token=hf_token,
        repo_type="model"
    )

    # load weights
    raw_state_dict = load_file(model_path, device='cpu')
    # print(list(raw_state_dict.keys())[:5])

    # strip `_orig_mod.` prefix from keys
    model_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()
    }
    # print(list(model_state_dict.keys())[:5])

    model_cfg = GPTConfig(max_seq_len=max(seq_len_train, seq_len_val))
    gpt_model = GPT(model_cfg)
    gpt_model.load_state_dict(model_state_dict, strict=False)

    # tie weights, creating wte.weight
    if gpt_model.use_tied_embeddings:
        gpt_model.transformer.wte.weight = gpt_model.lm_head.weight

    # load training state
    training_state = torch.load(training_state_path, map_location="cpu")

    return (
        # continue on the next step after val
        training_state['step'] + 1,
        training_state['torch_rng_state_cpu'],
        training_state['torch_rng_state_cuda'],
        training_state['train_tokens_processed'],
        training_state['total_train_t'],
        training_state['total_val_t'],
        training_state['total_sample_t'],
        training_state['total_hellaswag_t'],
        training_state['total_t'],
        training_state['best_val_loss'],
        training_state['epoch'],
        training_state['current_shard_idx'],
        training_state['grad_accum_mini_steps_per_shard_counter'],
        training_state['optimizer_state_dicts'],
        gpt_model,
    )

################################################
#      Initialization & Non-Model Config       #
################################################
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

# buffer to 'write to disk' only at checkpointing steps, to avoid e.g. stopping training at step 233,
# restoring a model from step 200 -last val step that improves val loss-  and having training logs up
# to 233 then continuing from 201 (e.g., 232, 233, 201, 202, ...).
# This ensures training losses are stored only after a checkpointing step happens.
log_buffer = []

total_tokens_per_step_train = 2**18 # 2**19 == 524,288 or ~0.5M tokens from Language Models are Few-Shot Learners
factor = 2
gpu_batch_size_train = 64 // factor
gpu_batch_size_val = 64 // factor
seq_len_train =  1024 * factor
seq_len_val = 1024 * factor

total_tokens_per_mini_step_train = ddp_world_size * gpu_batch_size_train * seq_len_train
grad_accum_mini_steps = total_tokens_per_step_train // total_tokens_per_mini_step_train
assert total_tokens_per_step_train % total_tokens_per_mini_step_train == 0
if master_process:
    message = f"per-gpu gradient accumulation mini-steps: {grad_accum_mini_steps}"
    print(message)
    log_buffer.append(message)

betas = (0.9,0.95)
eps = 1e-8
max_lr = 5e-3 # 4e-3 | 5e-3
min_lr_after_warmup_ratio = 0.15
warmup_tokens = 0
warmup_and_cosine_tokens = 800*10**6
max_tokens = 5*10**9
weight_decay = 0.1
hard_min_lr = 7e-4

max_steps = max_tokens // total_tokens_per_step_train
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // total_tokens_per_step_train
warmup_and_cosine_steps = warmup_and_cosine_tokens // total_tokens_per_step_train
if master_process:
    messages = [
        f"warmup steps: {warmup_steps:,}",
        f"warmup and cosine steps: {warmup_and_cosine_steps:,}",
        f"max steps: {max_steps:,}"
    ]
    for message in messages:
        print(message)
        log_buffer.append(message)

device_type = "cuda"

torch.set_float32_matmul_precision('high')

tokenizer = tiktoken.get_encoding("gpt2")

val_target = 3.28
val_tokens = 2 ** 21 * 5
val_steps = math.ceil(val_tokens / (ddp_world_size * gpu_batch_size_val * seq_len_val))
val_interval = 50
total_tokens_per_step_val = ddp_world_size * gpu_batch_size_val * seq_len_val
if master_process:
    print(f"{val_tokens:,} val tokens to be consumed in {val_steps:,} steps ({total_tokens_per_step_val:,} tokens per val step)")
# to save compute, start running validation when training loss + train_val_margin <= val_target
train_val_margin = 0.05
allow_val = False

sample_interval = 4000
sample_sequences = [
    "The universe has always been",
    "Who am I? I am a language model",
    "Artificial General Intelligence is",
    "Artificial General Intelligence is not",
    "2+2 is",
    "The quick brown fox jumps",
    "Earth is",
    "Could you tell me what time it is?"
]

hellaswag_interval = float('inf')

# checkpoint_interval = 1000
max_checkpoints_to_keep = -1
resume_from_checkpoint = False
resume_checkpoint_path = "model_step_0000125_val_6.8745_train_6.9019.safetensors"
resume_state_dict_path = "training_state_step_0000125.pt"

if not resume_from_checkpoint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    timestamp = "20250801_085237"

config_and_log_dir = f"./configs_and_logs/{timestamp}"
log_filename = os.path.join(config_and_log_dir, f"log.txt")
config_filename = os.path.join(config_and_log_dir, f"config.txt")

checkpoint_dir = f"./checkpoints/{timestamp}"

if master_process:
    os.makedirs(config_and_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

hf_user = os.environ.get("hf_user")
hf_token = os.environ.get("hf_token")
hub_repo_id = f"{hf_user}/gpt-3-small_{timestamp}"

start_step = 0
train_tokens_processed = 0
total_train_t = 0.0
total_val_t = 0.0
total_sample_t = 0.0
total_hellaswag_t = 0.0
total_t = 0.0
best_val_loss = float('inf')

base_seed = 1337

epoch = 0
current_shard_idx = 0
grad_accum_mini_steps_per_shard_counter = 0

seed = base_seed + ddp_rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

################################################
#           Model Building / Loading           #
################################################
model_cfg = GPTConfig(max_seq_len=max(seq_len_train, seq_len_val))
if resume_from_checkpoint:
    # get start_step, train dataloader config (epoch, grad_accum_mini_steps_per_shard_counter, etc.),
    # optimizer state and model
    (start_step, torch_rng_state_cpu, torch_rng_state_cuda, train_tokens_processed,
        total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
        current_shard_idx, grad_accum_mini_steps_per_shard_counter,
        optimizer_state_dicts, gpt_model) = load_checkpoint()
    if master_process:
        message = f"loaded checkpoint from step {start_step - 1}"
        print(message)
        log_buffer.append(message)
    dist.barrier()

    # move the downloaded gpt_model to device, compile it, set up DDP, and set raw_gpt_model to create the optimizer
    # based on the downloaded model
    gpt_model.to(device)
    #gpt_model = torch.compile(gpt_model, mode="max-autotune", fullgraph=True, dynamic=False)
    gpt_model = torch.compile(gpt_model)
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizers = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)
    # use the downloaded optimizer_state_dicts for the optimizers
    for k, optimizer in optimizers.items():
        if k in optimizer_state_dicts:
            optimizer.load_state_dict(optimizer_state_dicts[k])

    # restore the random number generator state
    torch.set_rng_state(torch_rng_state_cpu)
    torch.cuda.set_rng_state(torch_rng_state_cuda)
    # wait for all ranks to sync
    dist.barrier()
else:
    gpt_model = GPT(model_cfg)
    gpt_model.to(device)
    #gpt_model = torch.compile(gpt_model, mode="max-autotune", fullgraph=True, dynamic=False)
    gpt_model = torch.compile(gpt_model)
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizers = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)

################################################
#             DataLoader Building              #
################################################
data_path = "./data/edu_fineweb10B"
# shuffle or first 10_485_760 tokens of the FineWeb validation shard for the NanoGPT Speedrun
shuffle_val_tokens = True
# if resume_from_checkpoint: epoch, current_shard_idx, and grad_accum_mini_steps_per_shard_counter are overriden
# above and thus set to non-zero values in the DataLoader()
train_data_loader = DataLoader(
    gpu_batch_size_train, seq_len_train, ddp_world_size, ddp_rank, data_path, "train",
    epoch=epoch, current_shard_idx=current_shard_idx,
    grad_accum_mini_steps_per_shard_counter=grad_accum_mini_steps_per_shard_counter,
    pad_token_id=model_cfg.pad_token_id, eos_token_id=model_cfg.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
)
val_data_loader = DataLoader(
    gpu_batch_size_val, seq_len_val, ddp_world_size, ddp_rank, data_path, "val",
    epoch=0, current_shard_idx=0, grad_accum_mini_steps_per_shard_counter=0,
    pad_token_id=model_cfg.pad_token_id, eos_token_id=model_cfg.eos_token_id,
    return_document_ids=raw_gpt_model.use_doc_masking,
    shuffle_val_tokens=shuffle_val_tokens,
)

################################################
# HellaSwag Loading and Splitting across gpus  #
################################################
# load HellaSwag and split it across all GPUs similar to how the train DataLoader assigns different chunks to each GPU
hellaswag_train_dataset = load_dataset("hellaswag", split="train")
hellaswag_val_dataset = load_dataset("hellaswag", split="validation")
total_hellaswag_val_examples = len(hellaswag_val_dataset)
hellaswag_val_examples_per_rank = (total_hellaswag_val_examples + ddp_world_size - 1) // ddp_world_size
hellaswag_val_start_idx = ddp_rank * hellaswag_val_examples_per_rank
hellaswag_val_end_idx = min(hellaswag_val_start_idx + hellaswag_val_examples_per_rank, total_hellaswag_val_examples)
local_hellaswag_val_dataset = hellaswag_val_dataset.select(range(hellaswag_val_start_idx, hellaswag_val_end_idx))
local_message = (
    f"[rank {ddp_rank}] gets HellaSwag sentences "
    f"{hellaswag_val_start_idx:,} to {hellaswag_val_end_idx - 1:,}"
)
print(local_message)
all_messages = [None for _ in range(ddp_world_size)]
dist.all_gather_object(all_messages, local_message)
if master_process:
    log_buffer.extend(all_messages)
dist.barrier()

################################################
#             Config Saving to file            #
################################################
def save_config_info():
    with open(config_filename, "w") as f:
        f.write(f"timestamp: {timestamp}\n")

        f.write(f"ddp world size: {ddp_world_size}\n")

        f.write(f"total tokens per step (train): {total_tokens_per_step_train}\n")
        f.write(f"gpu batch size (train): {gpu_batch_size_train}\n")
        f.write(f"gpu batch size (val): {gpu_batch_size_val}\n")
        f.write(f"seq len (train): {seq_len_train}\n")
        f.write(f"seq len (val): {seq_len_val}\n")
        f.write(f"total tokens per mini-step (train): {total_tokens_per_mini_step_train}\n")
        f.write(f"grad accum mini-steps: {grad_accum_mini_steps}\n")
        f.write(f"total tokens per step (val): {total_tokens_per_step_val}\n")
        
        f.write(f"betas: {betas}\n")
        f.write(f"eps: {eps}\n")
        f.write(f"max lr: {max_lr}\n")
        f.write(f"min lr after warmup ratio: {min_lr_after_warmup_ratio}\n")
        f.write(f"warmup tokens: {warmup_tokens}\n")
        f.write(f"warmup and cosine tokens: {warmup_and_cosine_tokens}\n")
        f.write(f"max tokens: {max_tokens}\n")
        f.write(f"weight decay: {weight_decay}\n")
        f.write(f"hard min lr: {hard_min_lr}\n")

        # derived
        f.write(f"max steps: {max_steps}\n")
        f.write(f"min lr after warmup: {min_lr_after_warmup}\n")
        f.write(f"warmup steps: {warmup_steps}\n")
        f.write(f"warmup and cosine steps: {warmup_and_cosine_steps}\n")

        f.write(f"base seed: {base_seed}\n")
        f.write(f"device type: {device_type}\n")
        f.write(f"tokenizer: gpt2 (tiktoken)\n")

        f.write(f"shuffle val tokens: {shuffle_val_tokens}\n")
        f.write(f"val target: {val_target}\n")
        f.write(f"val steps: {val_steps}\n")
        f.write(f"val interval: {val_interval}\n")
        f.write(f"sample interval: {sample_interval}\n")
        f.write(f"train val margin: {train_val_margin}\n")

        for k, v in model_cfg.__dict__.items():
            f.write(f"model config - {k}: {v}\n")

if master_process:
    save_config_info()

################################################
#              Parameter Logging               #
################################################
# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
if master_process:
    message = f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters"
    print(message)
    log_buffer.append(message)

################################################
#                Kernel Warmup                 #
################################################
def _kernel_warmup(num_train_steps=2):
    # snapshot everything so we don't "cheat"
    model_state = copy.deepcopy(raw_gpt_model.state_dict())
    optimizer_states = {k: copy.deepcopy(opt.state_dict()) for k, opt in optimizers.items()}
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state()

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # train-shape warmup (compile both DDP graphs)
    gpt_model.train()
    with torch.enable_grad():
        for _ in range(num_train_steps):
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)

            for mini_step in range(grad_accum_mini_steps):
                # mimic the real training loop’s DDP behavior
                gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)

                # make shapes match training
                x_train = torch.randint(
                    0, raw_gpt_model.pad_token_id,
                    (gpu_batch_size_train, seq_len_train), device=device
                )
                y_train = torch.randint(
                    0, raw_gpt_model.pad_token_id,
                    (gpu_batch_size_train, seq_len_train), device=device
                )

                # synthesize doc_ids so the doc-masking + SWA FlexAttention path compiles
                if raw_gpt_model.use_doc_masking:
                    # set random-ish EOS boundaries
                    step = max(16, seq_len_train // 8)
                    idxs = torch.arange(seq_len_train, device=device)[None, :]
                    rand_offsets = torch.randint(0, step, (gpu_batch_size_train, 1), device=device)
                    is_eos = ((idxs + rand_offsets) % step == 0)
                    doc_ids_train = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
                else:
                    doc_ids_train = None

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, warm_loss = gpt_model(x_train, y_train, document_ids=doc_ids_train)

                (warm_loss / grad_accum_mini_steps).backward()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # val-shape warmup (compile eval forward)
    gpt_model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        x_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        y_val = torch.randint(
            0, raw_gpt_model.pad_token_id,
            (gpu_batch_size_val, seq_len_val), device=device
        )
        if raw_gpt_model.use_doc_masking:
            step = max(16, seq_len_val // 8)
            idxs = torch.arange(seq_len_val, device=device)[None, :]
            rand_offsets = torch.randint(0, step, (gpu_batch_size_val, 1), device=device)
            is_eos = ((idxs + rand_offsets) % step == 0)
            doc_ids_val = torch.cumsum(is_eos.to(torch.int32), dim=1) - is_eos.to(torch.int32)
        else:
            doc_ids_val = None

        _ = gpt_model(x_val, y_val, document_ids=doc_ids_val)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # sampling-shape warmup
    raw_gpt_model.eval()
    max_new_tokens = get_sample_token_count(start_step)
    max_allowed_input_len = raw_gpt_model.max_seq_len - max_new_tokens
    initial_input_ids_list = [
        tokenizer.encode(sequence)[:max_allowed_input_len] 
        for sequence in sample_sequences
    ]
    max_input_len = max(1, max(len(ids) for ids in initial_input_ids_list))
    alloc_len = max_input_len + max_new_tokens
    round_multiple = raw_gpt_model.flex_block_size if raw_gpt_model.use_flex_attention else 8
    alloc_len = min(raw_gpt_model.max_seq_len, math.ceil(alloc_len / round_multiple) * round_multiple)

    sampling_input_ids = torch.full(
        (len(sample_sequences), alloc_len),
        raw_gpt_model.pad_token_id,
        dtype=torch.long,
        device=device
    )
    for i, ids in enumerate(initial_input_ids_list):
        if ids:
            sampling_input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        _ = raw_gpt_model(
            sampling_input_ids,
            attn_mask=None,
            ignore_doc_mask=True,
            document_ids=None,
        )

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # hellaswag-shape warmup
    raw_gpt_model.eval()
    hellaswag_examples_per_batch = 16
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    alloc_len = min(1024, raw_gpt_model.max_seq_len)

    pre_allocated_input_ids = torch.full(
        (max_gpu_sequences_per_batch, alloc_len),
        raw_gpt_model.pad_token_id,
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        _ = raw_gpt_model(
            pre_allocated_input_ids,
            attn_mask=None,
            ignore_doc_mask=True,
            document_ids=None,
        )

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # restore state
    raw_gpt_model.load_state_dict(model_state)
    for k, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[k])
    torch.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

_kernel_warmup(num_train_steps=2)

##################################################################
#  Training, Validation, Sampling, HellaSwag, Checkpointing loop #
##################################################################
try:
    for step in range(start_step, max_steps):
        
        torch.cuda.synchronize()
        start_train_t = time.time()
        gpt_model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        train_loss = 0.0
        # step_checkpointed = False

        # per-gpu grad accumulation mini-steps
        for mini_step in range(grad_accum_mini_steps):
            x_train, y_train, doc_ids_train = train_data_loader.next_batch()
            x_train = x_train.pin_memory().to(device, non_blocking=True)
            y_train = y_train.pin_memory().to(device, non_blocking=True)
            if doc_ids_train is not None:
                doc_ids_train = doc_ids_train.pin_memory().to(device, non_blocking=True)
            gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, step_train_loss = gpt_model(x_train, y_train, document_ids=doc_ids_train)
            step_train_loss /= grad_accum_mini_steps
            train_loss += step_train_loss.detach()
            step_train_loss.backward()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        
        # # if clipping is enabled and threshold > 0, use it; otherwise use +inf (no-op clip).
        # grad_clip_enabled = raw_gpt_model.use_grad_norm_clipping
        # gradient_clipping_norm = raw_gpt_model.gradient_clipping_norm if (grad_clip_enabled and raw_gpt_model.gradient_clipping_norm > 0.0) else float('inf')
        # # instead of clamping each weight’s gradient (individually), we compute the global L2
        # # norm of all gradients across the parameter set and if ||G|| > max_norm: we scale 
        # # all gradients down proportionally
        # # -------------
        # # the steps are:
        # # forward pass -> loss
        # # loss.backward() -> autograd computes true grads in p.grad
        # # (if using AMP GradScaler) scaler.unscale_(optimizer)
        # # clip grads (global norm) -> in-place scale of p.grad
        # # optimizer.step()
        # # optimizer.zero_grad() (next step)
        # # -------------
        # # Hence, after deriving everything (dJ/dw_i for each w_i), we have
        # # G which can be seen as a (total_params, 1) grad column vector 
        # # that (if gradient clipping is applied) is rescaled to 
        # # have its L2 norm <= grad_clip_norm_threshold
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), gradient_clipping_norm)
        # # recompute global norm after clipping (cheap)
        # if grad_clip_enabled:
        #     # grad_sq = [p.grad.detach().float().pow(2).sum() for p in gpt_model.parameters() if p.grad is not None]
        #     # grad_norm_post_clipping = float(torch.stack(grad_sq).sum().sqrt())
        #     grad_norm_post_clipping = grad_norm_pre_clipping if grad_norm_pre_clipping < raw_gpt_model.gradient_clipping_norm else raw_gpt_model.gradient_clipping_norm
        # else:
        #     grad_norm_post_clipping = grad_norm_pre_clipping
        # grad_norm_text = f"{grad_norm_pre_clipping:.4f}" + (f" pre- / {grad_norm_post_clipping:.4f} post-clipping" if grad_clip_enabled else "")
        # <-- Uncomment for gradient norm clipping -->
        # grad_norm_pre_clipping = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),raw_gpt_model.gradient_clipping_norm)
        # grad_norm_text = (
        #         f"{float(grad_norm_pre_clipping):.4f} pre-"
        # )
        # <-- Uncomment for gradient norm clipping -->
        
        lr = get_lr(step)
        # AdamW gets the base lr
        for param_group in optimizers["adamw"].param_groups:
            param_group['lr'] = lr
        # Muon if present gets scaled lr
        if "muon" in optimizers:
            for param_group in optimizers["muon"].param_groups:
                param_group['lr'] = lr * raw_gpt_model.muon_lr_scale
        # Step present optimizers
        for optimizer in optimizers.values():
            optimizer.step()

        # the logits can vary across ranks
        # <-- Uncomment for logit soft-capping -->
        # logits_absmax_pre_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_pre_capping"]), device=device)
        # logits_absmax_post_capping = torch.tensor(float(raw_gpt_model._logits_absmax_stats["logits_absmax_post_capping"]), device=device)
        # # MAX or MEAN
        # dist.all_reduce(logits_absmax_pre_capping, op=dist.ReduceOp.MAX)
        # dist.all_reduce(logits_absmax_post_capping, op=dist.ReduceOp.MAX)
        # <-- Uncomment for logit soft-capping -->

        tl = float(train_loss)

        torch.cuda.synchronize()
        end_train_t = time.time()
        train_step_t = end_train_t - start_train_t
        total_train_t += train_step_t
        total_t += train_step_t

        if master_process:
            # <-- Uncomment for logit soft-capping -->
            # if raw_gpt_model.use_lm_head_logit_softcapping:
            #     logits_suffix = f" | logits absmax pre/post {logits_absmax_pre_capping.item():.4f}/{logits_absmax_post_capping.item():.4f}"
            # else:
            #     logits_suffix = f" | logits absmax {logits_absmax_pre_capping.item():.4f}"
            # <-- Uncomment for logit soft-capping -->
            # the scale is constant across ranks because it is head-dependant, and all
            # gpus share the heads (even though with different data)
            qk_suffix = ""
            if raw_gpt_model.use_qk_norm and raw_gpt_model.use_qk_debug_log:
                qk_suffix = " | " + qk_scale_debug_string(raw_gpt_model)
            train_tokens_processed += (grad_accum_mini_steps * total_tokens_per_mini_step_train)
            train_log_content = (
                f"step: {step:,} | train loss: {tl:.8f} | "
                f"train ppl: {math.exp(tl):,.2f} | "
                f"train step time: {1000*(train_step_t):,.2f} ms | "
                # f"grad norm: {grad_norm_text} | lr: {lr:.8f} | "
                f"tok/s: {total_tokens_per_step_train / train_step_t:,.2f} | "
                f"total toks: {train_tokens_processed:,} | total time: {total_t/60:,.2f} min"
                f"{qk_suffix}"
                # f"{logits_suffix}"
            )
            print(train_log_content)
            # with open(log_filename, "a") as f:
            #     f.write(train_log_content + "\n")
            log_buffer.append(train_log_content)

        if (step % sample_interval == 0 and step > 0) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_sample_t = time.time()
            max_new_tokens = get_sample_token_count(step)
            sample(sample_sequences, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            end_sample_t = time.time()
            sample_step_t = end_sample_t - start_sample_t
            total_sample_t += sample_step_t
            total_t += sample_step_t
            if master_process:
                message = f"step: {step:,} | sampling time: {(sample_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)

        if (step % hellaswag_interval == 0 and step > 0) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_hellaswag_t = time.time()
            accuracy = evaluate_hellaswag_standard()
            torch.cuda.synchronize()
            end_hellaswag_t = time.time()
            hellaswag_step_t = end_hellaswag_t - start_hellaswag_t
            total_hellaswag_t += hellaswag_step_t
            total_t += hellaswag_step_t
            if master_process:
                message = f"step: {step:,} | HellaSwag acc: {accuracy:.4f} | HellaSwag time: {(hellaswag_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)
        
        # val gating
        if (tl + train_val_margin <= val_target) and not allow_val:
            allow_val = True
            if master_process:
                message = (f"val enabled at step {step} - {tl} train loss")
                print(message)
                log_buffer.append(message)
        
        if (allow_val and (step % val_interval == 0 and step > 0)) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_val_t = time.time()
            gpt_model.eval()
            if master_process:
                message = f"resetting val loader at step {step}"
                print(message)
                log_buffer.append(message)
            val_data_loader.reset()
            with torch.inference_mode():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val, doc_ids_val = val_data_loader.next_batch()
                    x_val = x_val.pin_memory().to(device, non_blocking=True)
                    y_val = y_val.pin_memory().to(device, non_blocking=True)
                    if doc_ids_val is not None:
                        doc_ids_val = doc_ids_val.pin_memory().to(device, non_blocking=True)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, step_val_loss = gpt_model(x_val, y_val, document_ids=doc_ids_val)
                    val_loss += step_val_loss / val_steps
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if master_process:
                    # pass (exact values as) arguments to avoid modifying the counters before it saves
                    # as save_checkpoint must defer writing to disk and the training loop continues
                    # before variables such as step or train_tokens_processed are stored
                    if max_checkpoints_to_keep > 0:
                        save_checkpoint(step=step,
                            torch_rng_state_cpu=torch.get_rng_state(),
                            torch_rng_state_cuda=torch.cuda.get_rng_state(),
                            train_loss=train_loss.item(),
                            val_loss=val_loss.item(),
                            train_tokens_processed=train_tokens_processed,
                            total_train_t=total_train_t,
                            total_val_t=total_val_t,
                            total_sample_t=total_sample_t,
                            total_hellaswag_t=total_hellaswag_t,
                            total_t=total_t,
                            best_val_loss=best_val_loss,
                            epoch=train_data_loader.epoch,
                            current_shard_idx=train_data_loader.current_shard_idx,
                            grad_accum_mini_steps_per_shard_counter=train_data_loader.grad_accum_mini_steps_per_shard_counter,
                            optimizers=optimizers)
                    message = f"new best val loss: {best_val_loss:.8f}"
                    print(message)
                    log_buffer.append(message)
                    # only after we improve val loss and a checkpoint is saved, we push the logs
                    with open(log_filename, "a") as f:
                        for line in log_buffer:
                            f.write(line + "\n")
                        log_buffer.clear()
                dist.barrier()
                # step_checkpointed = True

            torch.cuda.synchronize()
            end_val_t = time.time()
            val_step_t = end_val_t - start_val_t
            total_val_t += val_step_t
            total_t += val_step_t

            if master_process:
                val_log_content = f"step: {step:,} | val loss: {val_loss.item():.8f} | val ppl: {math.exp(val_loss.item()):,.2f} | val time: {1000*(val_step_t):,.2f} ms"
                print(val_log_content)
                # with open(log_filename, "a") as f:
                #     f.write(val_log_content + "\n")
                log_buffer.append(val_log_content)

            if val_loss <= val_target:
                if master_process:
                    print(f"val loss {val_loss.item():.8f} reached target {val_target}")
                break

        # if not step_checkpointed and step % checkpoint_interval == 0 and step > 0 and master_process:
        #     start_checkpointing_t = time.time()
        #     save_checkpoint(train_loss=train_loss)
        #     torch.cuda.synchronize()
        #     end_checkpointing_t = time.time()
        #     checkpointing_step_t = end_checkpointing_t - start_checkpointing_t
        #     total_t += checkpointing_step_t

except Exception as e:
    if master_process:
        print(f"[rank {ddp_rank}] unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    dist.barrier()
    raise
finally:
    dist.barrier()
    destroy_process_group()