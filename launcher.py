#!/usr/bin/env python3
"""
Wraps mlx_lm.server with TurboQuant KV cache compression.

The model is a hybrid Attention+SSM architecture (qwen3_5):
  - SSM/linear layers  → ArraysCache(size=2)  — left untouched
  - Attention layers   → KVCache by default   — replaced with TurboQuantKVCache
                                                 except first/last FP16_LAYERS
"""
import mlx_lm.server as _server
from mlx_lm.models.cache import make_prompt_cache as _orig_make_cache, KVCache
from turboquant_mlx import TurboQuantKVCache, apply_patch
from turboquant_mlx.cache import TurboQuantKVCache as _TQC

FP16_LAYERS = 4
TQ_BITS     = 3


# ── deepcopy fix ────────────────────────────────────────────────────────────
# The LRUPromptCache calls copy.deepcopy() when reusing a cached prompt.
# TurboQuantKVCache stores mx.Dtype and _Quantizer (mx.array internals) that
# Python's generic deepcopy can't handle.  We implement __deepcopy__ using the
# class's own from_state / meta_state serialisation which avoids all mlx types.
def _tq_deepcopy(self, memo):
    if self.empty():
        new_obj = _TQC(
            bits=self.quant_bits,
            seed=self.seed,
            fused=self.fused,
            sparse_v_threshold=self.sparse_v_threshold,
            v_only=self.v_only,
        )
    else:
        new_obj = _TQC.from_state(self.state, self.meta_state)
        new_obj.fused              = self.fused
        new_obj.sparse_v_threshold = self.sparse_v_threshold
        new_obj.v_only             = self.v_only
    memo[id(self)] = new_obj
    return new_obj

_TQC.__deepcopy__ = _tq_deepcopy
# ────────────────────────────────────────────────────────────────────────────


def _turboquant_prompt_cache(model, max_kv_size=None):
    # Get the model's own correct cache types per layer
    # (ArraysCache for SSM/linear, KVCache for attention)
    original = _orig_make_cache(model, max_kv_size)

    attn_positions = [i for i, c in enumerate(original) if type(c) is KVCache]
    n_attn = len(attn_positions)

    result = list(original)
    for rank, pos in enumerate(attn_positions):
        if FP16_LAYERS <= rank < n_attn - FP16_LAYERS:
            result[pos] = TurboQuantKVCache(bits=TQ_BITS, fused=True)

    compressed = sum(1 for c in result if isinstance(c, TurboQuantKVCache))
    print(
        f"[TurboQuant] {len(attn_positions)} attn layers | "
        f"{compressed} compressed @ {TQ_BITS}-bit fused | "
        f"{len(attn_positions) - compressed} FP16 | "
        f"{len(original) - len(attn_positions)} SSM untouched"
    )
    return result


# Fused Metal attention patch (attention SDPA only, does not touch SSM layers)
apply_patch()

# server.py does `from .models.cache import make_prompt_cache` which binds the
# name into server.py's own namespace. We must overwrite that namespace entry
# directly — patching the source module has no effect after import.
_server.make_prompt_cache = _turboquant_prompt_cache

from mlx_lm.server import main
main()
