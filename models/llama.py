"""
Llama 2 7B GPTQ Quantization using Memboost
=============================================

Layer-streaming GPTQ quantization for Llama 2 with:
  - Hessian-based mixed 2/4-bit precision selection
  - GPTQ optimal rounding with per-group n_levels
  - Layer-by-layer calibration (fits in 8GB VRAM)
  - Calibration via WikiText-2

This script quantizes the model and saves the compressed checkpoint.
It does NOT perform inference — use the quantized output with your
inference engine of choice.

Usage:
    python models/llama.py --model meta-llama/Llama-2-7b-hf --ratio-4bit 0.1
    python models/llama.py --model meta-llama/Llama-2-7b-hf --save quantized_llama2.pt

Dependencies: torch, memboost, transformers (tokenizer + weight download), datasets (calibration)
"""

import argparse
import gc
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import memboost


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_heads: int = 32
    n_inner: int = 11008
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 2048
    ratio_4bit: float = 0.1


# ---------------------------------------------------------------------------
# Minimal Model Components (needed for GPTQ calibration forward passes)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0, device="cpu"):
    """Precompute cos/sin for rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Apply rotary embeddings using HF rotate_half convention."""
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


class QuantizedLinear(nn.Module):
    """Linear layer backed by memboost quantized weights.

    Forward: dequantize + F.linear (used during calibration re-forward).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_weight: Optional[memboost.QuantizedTensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = memboost.dequantize(self.q_weight)  # [out, in] fp16
        return F.linear(x.to(torch.float16), w)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = self.n_head // self.n_kv_heads

        self.q_proj = None  # Set during quantization
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin, 0)
        k = apply_rope(k, rope_cos, rope_sin, 0)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1), scale=1.0 / math.sqrt(self.head_dim),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.o_proj(y)
        return y


class LlamaMLP(nn.Module):
    """SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))"""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x, rope_cos, rope_sin):
        h = self.self_attn(self.input_layernorm(x), rope_cos, rope_sin)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# GPTQ Algorithm
# ---------------------------------------------------------------------------


def gptq_inverse(H: torch.Tensor):
    """Compute regularized Cholesky inverse of Hessian.

    Args:
        H: [K, K] float32 symmetric PSD Hessian.

    Returns:
        (h_diag, Hinv_cho): diagonal of H^{-1} [K] float32,
            and full Cholesky-based inverse [K, K] float32.
    """
    K = H.shape[0]
    diag = H.diagonal()
    damp = 0.01 * diag.abs().max()
    damp = max(damp.item(), 1e-6)
    H.diagonal().add_(damp)
    dead = H.diagonal() < 1e-8
    if dead.any():
        H.diagonal()[dead] = 1.0

    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)

    h_diag = Hinv.diag().clone()
    Hinv.diagonal().add_(1e-8)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)

    return h_diag, Hinv_cho


GROUP_SIZE = memboost.GROUP_SIZE_1ST  # 16


def compute_group_precision(W: torch.Tensor, h_diag: torch.Tensor, ratio_4bit: float):
    """Determine which column-groups get 4-bit precision.

    Matches the CUDA sensitivity kernel exactly:
      sensitivity[g] = sum_{row, col in group} w^2 / h_inv_diag[col]
    """
    M, K = W.shape
    num_groups = K // GROUP_SIZE
    W_sq = W.pow(2)

    h_inv = h_diag.clamp(min=1e-8)
    inv_h = 1.0 / h_inv

    W_sq_scaled = W_sq * inv_h.unsqueeze(0)
    sensitivity = W_sq_scaled.view(M, num_groups, GROUP_SIZE).sum(dim=(0, 2))

    num_4bit = int(ratio_4bit * num_groups)
    group_prec = torch.zeros(num_groups, dtype=torch.uint8, device=W.device)
    if num_4bit > 0:
        _, top_idx = sensitivity.topk(num_4bit)
        group_prec[top_idx] = 1

    return group_prec


def gptq_quantize(
    W: torch.Tensor,
    Hinv_cho: torch.Tensor,
    group_precision: torch.Tensor,
    blocksize: int = 128,
):
    """GPTQ optimal rounding with mixed precision per group.

    Groups marked as 4-bit (group_precision=1) are quantized to 16 levels;
    2-bit groups (group_precision=0) use 4 levels.
    """
    M, K = W.shape
    Q = W.clone()
    num_groups = K // GROUP_SIZE

    gp_cpu = group_precision.cpu().numpy()
    group_n_levels = [16 if gp_cpu[g] == 1 else 4 for g in range(num_groups)]

    group_min = torch.zeros(num_groups, device=W.device)
    group_max = torch.zeros(num_groups, device=W.device)
    group_scale = torch.zeros(num_groups, device=W.device)
    for g in range(num_groups):
        col_s = g * GROUP_SIZE
        grp = W[:, col_s : col_s + GROUP_SIZE]
        gmin = grp.min().item()
        gmax = grp.max().item()
        group_min[g] = gmin
        group_max[g] = gmax
        n_lev = group_n_levels[g]
        group_scale[g] = (gmax - gmin) / (n_lev - 1) if gmax > gmin else 0.0

    for block_start in range(0, K, blocksize):
        block_end = min(block_start + blocksize, K)
        W_block = Q[:, block_start:block_end].clone()
        Err = torch.zeros_like(W_block)

        for j in range(block_end - block_start):
            col = block_start + j
            g = col // GROUP_SIZE
            gmin = group_min[g].item()
            sc = group_scale[g].item()
            n_lev = group_n_levels[g]

            w_col = W_block[:, j]

            if sc > 0:
                q_int = ((w_col - gmin) / sc).round().clamp(0, n_lev - 1)
                zero = round(-gmin / sc)
                q_col = sc * (q_int - zero)
            else:
                q_col = torch.full_like(w_col, gmin)

            Err[:, j] = (w_col - q_col) / Hinv_cho[col, col]
            Q[:, col] = q_col

            if j + 1 < block_end - block_start:
                W_block[:, j + 1 :] -= Err[:, j : j + 1] * Hinv_cho[
                    col, block_start + j + 1 : block_end
                ].unsqueeze(0)

    return Q


# ---------------------------------------------------------------------------
# Calibration Data
# ---------------------------------------------------------------------------


def get_calibration_data(tokenizer, n_samples: int = 128, seq_len: int = 128):
    """Load calibration tokens from WikiText-2, fallback to random."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"])
        tokens = tokenizer.encode(text)
        samples = []
        for i in range(n_samples):
            start = i * seq_len
            if start + seq_len > len(tokens):
                break
            samples.append(
                torch.tensor(tokens[start : start + seq_len], dtype=torch.long)
            )
        if len(samples) >= n_samples:
            return torch.stack(samples)
    except Exception as e:
        print(f"  WikiText-2 unavailable ({e}), using random tokens")

    return torch.randint(0, 32000, (n_samples, seq_len), dtype=torch.long)


# ---------------------------------------------------------------------------
# Layer-Streaming Quantization
# ---------------------------------------------------------------------------


def _make_quantized_linear(W_fp32, h_diag, Hinv_cho, ratio_4bit, blocksize=128):
    """GPTQ + memboost quantize a single weight matrix.

    Two-pass approach:
      1. Pre-compute group_precision from original weights + Hessian
      2. Run GPTQ with per-group n_levels (4 for 2-bit, 16 for 4-bit)
      3. Feed GPTQ output to memboost.quantize() with a crafted hessian_diag
         that forces memboost to select the SAME groups for 4-bit
    """
    M, K = W_fp32.shape

    if ratio_4bit <= 0:
        num_groups = K // GROUP_SIZE
        group_prec = torch.zeros(num_groups, dtype=torch.uint8, device=W_fp32.device)
        Q = gptq_quantize(W_fp32, Hinv_cho, group_prec, blocksize=blocksize)
        W_fp16 = Q.to(torch.float16).contiguous()
        qt = memboost.quantize(W_fp16, ratio_4bit=0.0)
        ql = QuantizedLinear(K, M)
        ql.q_weight = qt
        return ql

    group_prec = compute_group_precision(W_fp32, h_diag, ratio_4bit)
    Q = gptq_quantize(W_fp32, Hinv_cho, group_prec, blocksize=blocksize)

    # Craft hessian_diag to force memboost's sensitivity kernel to
    # select the SAME groups for 4-bit.
    forced_h_diag = torch.ones(K, device=W_fp32.device, dtype=torch.float32)
    for g in range(K // GROUP_SIZE):
        col_s = g * GROUP_SIZE
        col_e = col_s + GROUP_SIZE
        if group_prec[g] == 1:  # 4-bit: want HIGH sensitivity -> small h_inv
            forced_h_diag[col_s:col_e] = 1e-10
        else:  # 2-bit: want LOW sensitivity -> large h_inv
            forced_h_diag[col_s:col_e] = 1e10

    W_fp16 = Q.to(torch.float16).contiguous()
    qt = memboost.quantize(W_fp16, ratio_4bit=ratio_4bit, hessian_diag=forced_h_diag)

    ql = QuantizedLinear(K, M)
    ql.q_weight = qt
    return ql


def _install_quantized_layer(block, attr_path, qt, in_features, out_features):
    """Install a QuantizedLinear from a QuantizedTensor onto a block."""
    ql = QuantizedLinear(in_features, out_features)
    ql.q_weight = qt
    parts = attr_path.split(".")
    obj = block
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], ql)


def quantize_llama(model_name: str, cfg: LlamaConfig, device: torch.device):
    """Layer-streaming GPTQ quantization of Llama 2.

    Loads one layer at a time from HF safetensors via mmap -> calibrate -> GPTQ
    -> quantize -> free fp16 -> next layer.  Never holds full model in RAM.

    Returns:
        quantized_layers: dict mapping layer name to QuantizedTensor
        stats: dict with compression statistics
    """
    import json
    from pathlib import Path
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Resolve safetensors shard paths
    snap_dir = Path(model_name)
    if not snap_dir.exists():
        cache_dir = (
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--{model_name.replace('/', '--')}"
        )
        snapshots = cache_dir / "snapshots"
        snap_dir = next(snapshots.iterdir())

    index_path = snap_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = {}
    for shard_name in set(weight_map.values()):
        shard_files[shard_name] = safe_open(
            str(snap_dir / shard_name), framework="pt", device="cpu"
        )

    def load_tensor(key):
        shard = shard_files[weight_map[key]]
        return shard.get_tensor(key)

    print("Using safetensors mmap loading (low RAM)")

    # Build a minimal embedding layer for calibration
    embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    embed_tokens.weight = nn.Parameter(
        load_tensor("model.embed_tokens.weight").to(torch.float16).to(device),
        requires_grad=False,
    )

    head_dim = cfg.n_embd // cfg.n_head
    rope_cos, rope_sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta, device=device)

    # Calibration data
    print("Preparing calibration data...")
    calib_ids = get_calibration_data(tokenizer)
    n_samples = calib_ids.shape[0]

    hidden_states_cpu = []
    with torch.no_grad():
        for i in range(n_samples):
            ids = calib_ids[i : i + 1].to(device)
            h = embed_tokens(ids).to(torch.float16)
            hidden_states_cpu.append(h.cpu())
    print(f"  Calibration: {n_samples} samples x {calib_ids.shape[1]} tokens")

    del embed_tokens

    quantized_layers = {}
    total_quantized = 0
    total_fp16_bytes = 0
    total_quant_bytes = 0

    for layer_idx in range(cfg.n_layer):
        t_layer = time.time()
        prefix = f"model.layers.{layer_idx}"

        block = LlamaBlock(cfg)
        block.eval()

        # Load layer norms
        block.input_layernorm.weight = nn.Parameter(
            load_tensor(f"{prefix}.input_layernorm.weight")
            .to(torch.float16).to(device),
            requires_grad=False,
        )
        block.post_attention_layernorm.weight = nn.Parameter(
            load_tensor(f"{prefix}.post_attention_layernorm.weight")
            .to(torch.float16).to(device),
            requires_grad=False,
        )

        # Load fp32 weights for GPTQ
        w_q = load_tensor(f"{prefix}.self_attn.q_proj.weight").to(torch.float32).to(device)
        w_k = load_tensor(f"{prefix}.self_attn.k_proj.weight").to(torch.float32).to(device)
        w_v = load_tensor(f"{prefix}.self_attn.v_proj.weight").to(torch.float32).to(device)
        w_o = load_tensor(f"{prefix}.self_attn.o_proj.weight").to(torch.float32).to(device)
        w_gate = load_tensor(f"{prefix}.mlp.gate_proj.weight").to(torch.float32).to(device)
        w_up = load_tensor(f"{prefix}.mlp.up_proj.weight").to(torch.float32).to(device)
        w_down = load_tensor(f"{prefix}.mlp.down_proj.weight").to(torch.float32).to(device)

        # Collect Hessians via forward hooks
        K_attn = w_q.shape[1]
        K_o = w_o.shape[1]
        K_mlp = w_gate.shape[1]
        K_down = w_down.shape[1]

        H_attn = torch.zeros(K_attn, K_attn, device=device, dtype=torch.float32)
        H_o = torch.zeros(K_o, K_o, device=device, dtype=torch.float32)
        H_mlp = torch.zeros(K_mlp, K_mlp, device=device, dtype=torch.float32)
        H_down = torch.zeros(K_down, K_down, device=device, dtype=torch.float32)

        # Temporary fp16 linear layers for calibration
        tmp_q = nn.Linear(K_attn, w_q.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_k = nn.Linear(K_attn, w_k.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_v = nn.Linear(K_attn, w_v.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_o = nn.Linear(K_o, w_o.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_gate = nn.Linear(K_mlp, w_gate.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_up = nn.Linear(K_mlp, w_up.shape[0], bias=False, device=device, dtype=torch.float16)
        tmp_down = nn.Linear(K_down, w_down.shape[0], bias=False, device=device, dtype=torch.float16)

        tmp_q.weight = nn.Parameter(w_q.half(), requires_grad=False)
        tmp_k.weight = nn.Parameter(w_k.half(), requires_grad=False)
        tmp_v.weight = nn.Parameter(w_v.half(), requires_grad=False)
        tmp_o.weight = nn.Parameter(w_o.half(), requires_grad=False)
        tmp_gate.weight = nn.Parameter(w_gate.half(), requires_grad=False)
        tmp_up.weight = nn.Parameter(w_up.half(), requires_grad=False)
        tmp_down.weight = nn.Parameter(w_down.half(), requires_grad=False)

        block.self_attn.q_proj = tmp_q
        block.self_attn.k_proj = tmp_k
        block.self_attn.v_proj = tmp_v
        block.self_attn.o_proj = tmp_o
        block.mlp.gate_proj = tmp_gate
        block.mlp.up_proj = tmp_up
        block.mlp.down_proj = tmp_down

        def make_hook(H_acc):
            def hook_fn(module, inp, out):
                x = inp[0].detach().float()
                x = x.view(-1, x.shape[-1])
                H_acc.add_(x.T @ x)
            return hook_fn

        handles = [
            block.self_attn.q_proj.register_forward_hook(make_hook(H_attn)),
            block.self_attn.o_proj.register_forward_hook(make_hook(H_o)),
            block.mlp.gate_proj.register_forward_hook(make_hook(H_mlp)),
            block.mlp.down_proj.register_forward_hook(make_hook(H_down)),
        ]

        # Forward calibration samples through this layer (fp16 weights)
        with torch.no_grad():
            for i in range(n_samples):
                h = hidden_states_cpu[i].to(device)
                block(h, rope_cos, rope_sin)

        for handle in handles:
            handle.remove()

        # Normalize Hessians
        total_tokens = n_samples * calib_ids.shape[1]
        H_attn.div_(total_tokens)
        H_o.div_(total_tokens)
        H_mlp.div_(total_tokens)
        H_down.div_(total_tokens)

        # GPTQ + memboost quantize each weight matrix

        # -- q/k/v share H_attn --
        h_diag_attn, Hinv_cho_attn = gptq_inverse(H_attn)
        del H_attn

        ql_q = _make_quantized_linear(w_q, h_diag_attn, Hinv_cho_attn, cfg.ratio_4bit)
        total_fp16_bytes += w_q.numel() * 2
        total_quant_bytes += ql_q.q_weight.total_bytes
        del w_q

        ql_k = _make_quantized_linear(w_k, h_diag_attn, Hinv_cho_attn, cfg.ratio_4bit)
        total_fp16_bytes += w_k.numel() * 2
        total_quant_bytes += ql_k.q_weight.total_bytes
        del w_k

        ql_v = _make_quantized_linear(w_v, h_diag_attn, Hinv_cho_attn, cfg.ratio_4bit)
        total_fp16_bytes += w_v.numel() * 2
        total_quant_bytes += ql_v.q_weight.total_bytes
        del w_v, h_diag_attn, Hinv_cho_attn

        # -- o_proj --
        h_diag_o, Hinv_cho_o = gptq_inverse(H_o)
        del H_o
        ql_o = _make_quantized_linear(w_o, h_diag_o, Hinv_cho_o, cfg.ratio_4bit)
        total_fp16_bytes += w_o.numel() * 2
        total_quant_bytes += ql_o.q_weight.total_bytes
        del w_o, h_diag_o, Hinv_cho_o

        # -- gate/up share H_mlp --
        h_diag_mlp, Hinv_cho_mlp = gptq_inverse(H_mlp)
        del H_mlp
        ql_gate = _make_quantized_linear(w_gate, h_diag_mlp, Hinv_cho_mlp, cfg.ratio_4bit)
        total_fp16_bytes += w_gate.numel() * 2
        total_quant_bytes += ql_gate.q_weight.total_bytes
        del w_gate

        ql_up = _make_quantized_linear(w_up, h_diag_mlp, Hinv_cho_mlp, cfg.ratio_4bit)
        total_fp16_bytes += w_up.numel() * 2
        total_quant_bytes += ql_up.q_weight.total_bytes
        del w_up, h_diag_mlp, Hinv_cho_mlp

        # -- down_proj --
        h_diag_down, Hinv_cho_down = gptq_inverse(H_down)
        del H_down
        ql_down = _make_quantized_linear(w_down, h_diag_down, Hinv_cho_down, cfg.ratio_4bit)
        total_fp16_bytes += w_down.numel() * 2
        total_quant_bytes += ql_down.q_weight.total_bytes
        del w_down, h_diag_down, Hinv_cho_down

        total_quantized += 7

        # Store quantized tensors
        layer_prefix = f"layers.{layer_idx}"
        quantized_layers[f"{layer_prefix}.self_attn.q_proj"] = ql_q.q_weight
        quantized_layers[f"{layer_prefix}.self_attn.k_proj"] = ql_k.q_weight
        quantized_layers[f"{layer_prefix}.self_attn.v_proj"] = ql_v.q_weight
        quantized_layers[f"{layer_prefix}.self_attn.o_proj"] = ql_o.q_weight
        quantized_layers[f"{layer_prefix}.mlp.gate_proj"] = ql_gate.q_weight
        quantized_layers[f"{layer_prefix}.mlp.up_proj"] = ql_up.q_weight
        quantized_layers[f"{layer_prefix}.mlp.down_proj"] = ql_down.q_weight

        # Install quantized layers on block for calibration re-forward
        block.self_attn.q_proj = ql_q
        block.self_attn.k_proj = ql_k
        block.self_attn.v_proj = ql_v
        block.self_attn.o_proj = ql_o
        block.mlp.gate_proj = ql_gate
        block.mlp.up_proj = ql_up
        block.mlp.down_proj = ql_down

        # Re-forward calibration through quantized layer to get next-layer inputs
        new_hidden_cpu = []
        with torch.no_grad():
            for i in range(n_samples):
                h = hidden_states_cpu[i].to(device)
                h_out = block(h, rope_cos, rope_sin)
                new_hidden_cpu.append(h_out.cpu())

        hidden_states_cpu = new_hidden_cpu
        del block

        torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - t_layer
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"  Layer {layer_idx:2d}: {elapsed:.1f}s, VRAM {vram:.0f} MiB")

    shard_files.clear()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nQuantized {total_quantized} linear layers across {cfg.n_layer} layers")
    print(f"  FP16 weight memory:      {total_fp16_bytes / 1024**2:.1f} MiB")
    print(f"  Quantized weight memory: {total_quant_bytes / 1024**2:.1f} MiB")
    print(f"  Compression ratio:       {total_fp16_bytes / total_quant_bytes:.2f}x")
    print(f"  GPU memory allocated:    {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")

    stats = {
        "fp16_bytes": total_fp16_bytes,
        "quantized_bytes": total_quant_bytes,
        "compression_ratio": total_fp16_bytes / total_quant_bytes,
        "num_layers": total_quantized,
    }

    return quantized_layers, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ quantize Llama 2 weights with memboost"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--ratio-4bit", type=float, default=0.1,
                        help="Fraction of groups at 4-bit precision")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save quantized checkpoint (.pt)")
    args = parser.parse_args()

    cfg = LlamaConfig(ratio_4bit=args.ratio_4bit)
    device = torch.device("cuda")

    t0 = time.time()
    quantized_layers, stats = quantize_llama(args.model, cfg, device)
    elapsed = time.time() - t0
    print(f"Quantization completed in {elapsed:.1f}s")

    if args.save:
        print(f"\nSaving quantized checkpoint to {args.save}...")
        save_dict = {}
        for name, qt in quantized_layers.items():
            save_dict[name] = qt.state_dict()
        save_dict["_stats"] = stats
        save_dict["_model_name"] = args.model
        save_dict["_ratio_4bit"] = args.ratio_4bit
        torch.save(save_dict, args.save)
        print(f"  Saved ({stats['quantized_bytes'] / 1024**2:.1f} MiB quantized weights)")

    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
