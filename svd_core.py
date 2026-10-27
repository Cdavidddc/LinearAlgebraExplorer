"""SVD compression of GPT-2 weight matrices.

Plain:     W' = U_k Σ_k V_kᵀ                               minimizes ||W - W'||_F   (Eckart-Young on W)
Whitened:  XXᵀ = LLᵀ,  WL = UΣVᵀ,  W' = U_k Σ_k V_kᵀ L⁻¹    minimizes ||WX - W'X||_F (Eckart-Young on WL)

Each layer's full SVD is computed once; compressing at any rank is then just slicing.
"""
import copy
import math
import random

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

METHODS = ("plain", "whitened")


# ---------------------------------------------------------------- layers
def target_layers(model):
    """Every Conv1D weight matrix inside the transformer blocks (4 per block, 48 in GPT-2 small)."""
    return {name: mod for name, mod in model.named_modules()
            if isinstance(mod, Conv1D) and name.startswith("transformer.h.")}


def weight_matrix(conv):
    """GPT-2's Conv1D stores W_conv as (in, out) and computes x @ W_conv; we use W = W_convᵀ, (m x n)."""
    return conv.weight.data.T.double()


def short_name(name):
    parts = name.split(".")                      # transformer.h.6.mlp.c_fc
    kind = {"c_attn": "attn qkv", "c_fc": "mlp in"}.get(parts[-1])
    if kind is None:
        kind = "attn out" if parts[-2] == "attn" else "mlp out"
    return f"block {parts[2]} · {kind}"


def rank_for_ratio(m, n, ratio):
    """Largest k with k(m+n) <= (1-ratio)·mn: the break-even rule."""
    return max(1, int((1 - ratio) * m * n / (m + n)))


# ---------------------------------------------------------------- calibration
@torch.no_grad()
def collect_grams(model, ids, n_samples, seq_len, device, seed=0, progress=None):
    """Accumulate XXᵀ (n x n) for the input of every target layer over random training windows."""
    layers = target_layers(model)
    grams = {name: torch.zeros(mod.weight.shape[0], mod.weight.shape[0],
                               dtype=torch.float64, device=device)
             for name, mod in layers.items()}
    hooks = []
    for name, mod in layers.items():
        def hook(_mod, inp, _out, name=name):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            grams[name] += x.T @ x
        hooks.append(mod.register_forward_hook(hook))
    rng = random.Random(seed)
    try:
        for i in range(n_samples):
            s = rng.randrange(0, len(ids) - seq_len)
            model(ids[s:s + seq_len].unsqueeze(0).to(device))
            if progress:
                progress((i + 1) / n_samples, f"Calibrating on sample {i + 1}/{n_samples}")
    finally:
        for h in hooks:
            h.remove()
    return grams


def whitening_factor(gram, eps=1e-4):
    """Lower-triangular L with XXᵀ + damping = LLᵀ (damping grows until Cholesky succeeds)."""
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    damp = eps * gram.diagonal().mean().clamp_min(1e-12)
    for _ in range(8):
        L, info = torch.linalg.cholesky_ex(gram + damp * eye)
        if info.item() == 0:
            return L
        damp *= 10
    raise RuntimeError("Cholesky failed even with heavy damping")


# ---------------------------------------------------------------- per-layer factorizations
class LayerSVD:
    """Full SVDs of W and of WL for one layer, so any rank k is a cheap slice."""

    def __init__(self, W, L):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U2, S2, Vh2 = torch.linalg.svd(W @ L, full_matrices=False)
        B2 = torch.linalg.solve_triangular(L, Vh2, upper=False, left=False)   # rows of Vᵀ L⁻¹
        self.W = W.float()
        self.L = L.float()
        self.shape = tuple(W.shape)
        self.sv = {"plain": S.float(), "whitened": S2.float()}
        self._f = {"plain": (U.float(), S.float(), Vh.float()),
                   "whitened": (U2.float(), S2.float(), B2.float())}
        self._err_cache = {}

    def factors(self, method, k):
        U, S, B = self._f[method]
        return U[:, :k] * S[:k], B[:k]          # A (m x k), B (k x n)

    def errors(self, method, k):
        """(relative weight error, relative output error) for rank k."""
        key = (method, k)
        if key not in self._err_cache:
            A, B = self.factors(method, k)
            D = self.W - A @ B
            w_err = (D.norm() / self.W.norm()).item()
            o_err = ((D @ self.L).norm() / (self.W @ self.L).norm()).item()
            self._err_cache[key] = (w_err, o_err)
        return self._err_cache[key]


def build_layer_svds(model, grams, device, eps=1e-4, progress=None):
    """Turn calibration statistics into a LayerSVD per target layer (consumes `grams` to save memory)."""
    svds = {}
    layers = target_layers(model)
    for i, (name, mod) in enumerate(layers.items()):
        gram = grams.pop(name).to(device=device, dtype=torch.float64)
        W = weight_matrix(mod).to(device)
        svds[name] = LayerSVD(W, whitening_factor(gram, eps))
        del gram
        if progress:
            progress((i + 1) / len(layers), f"Factoring {short_name(name)}")
    return svds


# ---------------------------------------------------------------- compressed model
class LowRankLinear(nn.Module):
    """Drop-in replacement for a Conv1D: y = A(Bx) + b, costing k(m+n) instead of mn."""

    def __init__(self, A, B, bias):
        super().__init__()
        self.A = nn.Parameter(A.contiguous())   # (m, k)
        self.B = nn.Parameter(B.contiguous())   # (k, n)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.B.T) @ self.A.T + self.bias


def compress(model, svds, ratio, method):
    model = copy.deepcopy(model)
    for name, mod in target_layers(model).items():
        m, n = svds[name].shape
        A, B = svds[name].factors(method, rank_for_ratio(m, n, ratio))
        new = LowRankLinear(A.clone(), B.clone(), mod.bias.data.clone())
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, new)
    return model.eval()


def matrix_params(model):
    """Parameters in the block weight matrices (excludes embeddings, LayerNorm and biases)."""
    return sum(p.numel() for n, p in model.named_parameters()
               if n.startswith("transformer.h.") and n.endswith((".weight", ".A", ".B"))
               and ".ln_" not in n)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def perplexity(model, ids, seq_len, device):
    """exp(mean next-token NLL) over non-overlapping windows."""
    nll, count = 0.0, 0
    for start in range(0, len(ids), seq_len):
        chunk = ids[start:start + seq_len].unsqueeze(0).to(device)
        if chunk.size(1) < 2:
            break
        n = chunk.size(1) - 1
        nll += model(chunk, labels=chunk).loss.item() * n
        count += n
    return math.exp(nll / count)
