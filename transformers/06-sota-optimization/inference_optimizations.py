"""
Inference Optimizations for Transformers
- KV cache helpers
- Speculative decoding scaffold
- PyTorch 2.x compile/bettertransformer hooks
"""

import torch
import torch.nn as nn


def init_kv_cache(max_batches: int, max_seq: int, num_heads: int, head_dim: int, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    k_cache = torch.zeros(max_batches, num_heads, max_seq, head_dim, device=device)  # (B, H, N, D_k)
    v_cache = torch.zeros_like(k_cache)  # (B, H, N, D_v)
    return {'k': k_cache, 'v': v_cache, 'pos': torch.zeros(max_batches, dtype=torch.long, device=device)}


def append_kv(cache, k_new: torch.Tensor, v_new: torch.Tensor, batch_idx: int):
    pos = cache['pos'][batch_idx].item()
    L = k_new.size(-2)
    cache['k'][batch_idx, :, pos:pos+L] = k_new
    cache['v'][batch_idx, :, pos:pos+L] = v_new
    cache['pos'][batch_idx] += L


def torch_compile_model(model: nn.Module):
    """Compile model with torch.compile if available."""
    if hasattr(torch, 'compile'):
        try:
            return torch.compile(model)
        except Exception:
            return model
    return model


def enable_bettertransformer_encoder(model: nn.Module) -> nn.Module:
    """Enable BetterTransformer fastpath for encoder-only models (if supported)."""
    try:
        from optimum.bettertransformer import BetterTransformer
        return BetterTransformer.transform(model)
    except Exception:
        return model


class SpeculativeDecoder:
    """Draft-and-verify speculative decoding scaffold.

    - draft_model proposes multiple tokens per step
    - target_model verifies/corrects
    """
    def __init__(self, draft_model: nn.Module, target_model: nn.Module, num_spec_tokens: int = 4):
        self.draft = draft_model
        self.target = target_model
        self.K = num_spec_tokens

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64):
        cur = input_ids
        for _ in range(max_new_tokens // self.K + 1):
            # draft K tokens
            draft_out = self._generate_step(self.draft, cur, self.K)
            # verify with target (placeholder logic)
            target_out = self._generate_step(self.target, cur, 1)
            cur = torch.cat([cur, draft_out[:, -self.K:]], dim=1)
        return cur

    def _generate_step(self, model, ids, steps: int):
        # Placeholder: last-token argmax loop
        out = ids
        for _ in range(steps):
            logits = model(out)  # expected (B, T, V)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out
