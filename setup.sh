#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────────────
# MLX TurboQuant Server — one-shot setup & launch
# Requires: Apple Silicon Mac (M1/M2/M3/M4), macOS 14+
#
# Usage:
#   chmod +x mlx-qwen-setup.sh
#   ./mlx-qwen-setup.sh                                      # default model
#   ./mlx-qwen-setup.sh mlx-community/Qwen3-8B-Instruct-4bit # custom model
#
# Environment overrides (optional):
#   MLX_MODEL_DIR   — override model storage path
#   MLX_VENV_DIR    — python venv path  (default: ~/.mlx-env)
#   MLX_PORT        — server port       (default: 8080)
#   HF_TOKEN        — HuggingFace token if model is gated
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL_REPO="mlx-community/Qwen3.6-35B-A3B-4bit"
MODEL_REPO="${1:-$DEFAULT_MODEL_REPO}"
MODEL_NAME="${MODEL_REPO##*/}"                                  # e.g. Qwen3.6-35B-A3B-4bit
MODEL_DIR="${MLX_MODEL_DIR:-${HOME}/.cache/mlx-models/${MODEL_REPO}}"
VENV_DIR="${MLX_VENV_DIR:-${HOME}/.mlx-env}"
PORT="${MLX_PORT:-8080}"
MAX_TOKENS=32768

MLX_LM_VERSION="0.31.3"
TURBOQUANT_REPO="https://github.com/arozanov/turboquant-mlx.git"
TURBOQUANT_COMMIT="7c6e3dc9459936852e19e571201d35d07d65e120"

LAUNCHER="${HOME}/Downloads/mlx-qwen-turboquant-launcher.py"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { print -P "%F{cyan}[setup]%f $*"; }
ok()   { print -P "%F{green}[ok]%f $*"; }
warn() { print -P "%F{yellow}[warn]%f $*"; }
die()  { print -P "%F{red}[error]%f $*" >&2; exit 1; }

# ── 1. Platform check ─────────────────────────────────────────────────────────
log "Checking platform..."
[[ "$(uname -s)" == "Darwin" ]] || die "macOS required."
arch=$(uname -m)
[[ "$arch" == "arm64" ]] || die "Apple Silicon (arm64) required. Got: $arch"
ok "Apple Silicon macOS detected."

# ── 2. Python 3.12 ───────────────────────────────────────────────────────────
log "Locating Python 3.12+..."
PYTHON=""
for candidate in \
    "${HOME}/.local/share/uv/python/cpython-3.12"*/bin/python3 \
    /opt/homebrew/bin/python3.12 \
    /usr/local/bin/python3.12 \
    $(command -v python3.12 2>/dev/null || true) \
    $(command -v python3    2>/dev/null || true); do
    [[ -z "$candidate" || ! -x "$candidate" ]] && continue
    ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null || true)
    if [[ "$ver" == "(3, 1"* ]]; then   # 3.10+ is fine; 3.12 preferred
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || die "Python 3.10+ not found. Install via: brew install python@3.12  or  curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "Using Python: $PYTHON ($(${PYTHON} --version 2>&1))"

# ── 3. Virtual environment ────────────────────────────────────────────────────
if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
    log "Creating venv at ${VENV_DIR}..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Venv created."
else
    ok "Venv already exists at ${VENV_DIR}."
fi
PY="${VENV_DIR}/bin/python3"
PIP="${VENV_DIR}/bin/pip3"

# ── 4. Core Python packages ───────────────────────────────────────────────────
log "Installing core packages (mlx-lm==${MLX_LM_VERSION}, transformers, huggingface_hub, regex)..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet \
    "mlx-lm==${MLX_LM_VERSION}" \
    "transformers>=4.45.0" \
    "huggingface_hub>=0.24.0" \
    "regex"
ok "Core packages installed."

# ── 5. turboquant-mlx (from GitHub pinned commit) ────────────────────────────
TURBOQUANT_INSTALLED=$("$PY" -c "import turboquant_mlx; print('yes')" 2>/dev/null || echo "no")
if [[ "$TURBOQUANT_INSTALLED" != "yes" ]]; then
    log "Installing turboquant-mlx from GitHub..."
    "$PIP" install --quiet "git+${TURBOQUANT_REPO}@${TURBOQUANT_COMMIT}"
    ok "turboquant-mlx installed."
else
    ok "turboquant-mlx already installed."
fi

# ── 6. Patch mlx_lm server (tool_calls streaming fix) ────────────────────────
SERVER_PY="${VENV_DIR}/lib/python3.12/site-packages/mlx_lm/server.py"
# Check for Python 3.11 fallback path
[[ -f "$SERVER_PY" ]] || SERVER_PY="${VENV_DIR}/lib/python3.11/site-packages/mlx_lm/server.py"
[[ -f "$SERVER_PY" ]] || die "Cannot find mlx_lm/server.py in venv."

PATCH_SENTINEL="tool_calls intentionally not cleared here"
if ! grep -q "$PATCH_SENTINEL" "$SERVER_PY"; then
    log "Applying tool_calls streaming patch to mlx_lm/server.py..."
    "$PY" - <<'PATCHSCRIPT'
import re, sys

server_py = None
import glob, os
venv = os.environ.get("VIRTUAL_ENV", "")
for p in glob.glob(f"{venv}/lib/python3.*/site-packages/mlx_lm/server.py"):
    server_py = p
    break

if not server_py:
    print("ERROR: could not locate server.py", file=sys.stderr)
    sys.exit(1)

content = open(server_py).read()

old = (
    "                    resp = self.generate_response(\n"
    "                        text,\n"
    "                        None,\n"
    "                        tool_calls=tool_formatter(tool_calls),\n"
    "                        reasoning_text=reasoning_text,\n"
    "                    )\n"
    "                    self.wfile.write(f\"data: {json.dumps(resp)}\\n\\n\".encode())\n"
    "                    self.wfile.flush()\n"
    "                    reasoning_text = \"\"\n"
    "                    text = \"\"\n"
    "                    tool_calls = []"
)
new = (
    "                    resp = self.generate_response(\n"
    "                        text,\n"
    "                        None,\n"
    "                        tool_calls=[],\n"
    "                        reasoning_text=reasoning_text,\n"
    "                    )\n"
    "                    self.wfile.write(f\"data: {json.dumps(resp)}\\n\\n\".encode())\n"
    "                    self.wfile.flush()\n"
    "                    reasoning_text = \"\"\n"
    "                    text = \"\"\n"
    "                    # tool_calls intentionally not cleared here — sent in final chunk"
)

if old not in content:
    print("WARNING: patch target not found — mlx_lm version may differ. Skipping patch.")
    sys.exit(0)

patched = content.replace(old, new, 1)
open(server_py, "w").write(patched)
print(f"Patched: {server_py}")
PATCHSCRIPT
    ok "Patch applied."
else
    ok "Patch already applied."
fi

# ── 7. Launcher script ────────────────────────────────────────────────────────
if [[ ! -f "$LAUNCHER" ]]; then
    log "Writing launcher script to ${LAUNCHER}..."
    cat > "$LAUNCHER" <<'PYEOF'
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


# Fused Metal attention patch
apply_patch()

# Replace the server's cache factory
_server.make_prompt_cache = _turboquant_prompt_cache

from mlx_lm.server import main
main()
PYEOF
    ok "Launcher written."
else
    ok "Launcher already exists at ${LAUNCHER}."
fi

# ── 8. Download model ─────────────────────────────────────────────────────────
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    log "Downloading model ${MODEL_REPO} to ${MODEL_DIR} (~18 GB, this will take a while)..."
    mkdir -p "$(dirname "$MODEL_DIR")"
    HF_ARGS=""
    [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS="--token ${HF_TOKEN}"
    "$PY" -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or None
snapshot_download(
    repo_id=os.environ['MLX_MODEL_REPO'],
    local_dir=os.environ['MLX_MODEL_DIR'],
    token=token,
    ignore_patterns=['*.md','*.txt','*.bin'],
)
print('Model downloaded.')
" MLX_MODEL_REPO="$MODEL_REPO" MLX_MODEL_DIR="$MODEL_DIR"
    ok "Model downloaded to ${MODEL_DIR}."
else
    ok "Model already present at ${MODEL_DIR}."
fi

# ── 9. Launch server ──────────────────────────────────────────────────────────
# NOTE: prompt cache memory is controlled solely by --prompt-cache-size.
#       --prefill-step-size 2048 is safe on M4 Max 36 GB; 4096 causes OOM.

export HF_HUB_OFFLINE=1

# Auto-detect thinking mode — only Qwen3 models support enable_thinking
CHAT_TEMPLATE_ARGS="{}"
if [[ "$MODEL_REPO" == *"Qwen3"* ]]; then
    CHAT_TEMPLATE_ARGS='{"enable_thinking":false}'
fi

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<your-ip>")

print ""
ok "Model:  ${MODEL_REPO}"
ok "Server: http://${LOCAL_IP}:${PORT}"
print ""

while true; do
    print "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server — model: ${MODEL_NAME}..."
    "$PY" "$LAUNCHER" \
        --model    "$MODEL_DIR" \
        --host     0.0.0.0 \
        --port     "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --prompt-cache-size 2 \
        --temp     0.1 \
        --prefill-step-size 2048 \
        --decode-concurrency 2 --prompt-concurrency 1 \
        --chat-template-args "$CHAT_TEMPLATE_ARGS" \
        --log-level INFO

    EXIT_CODE=$?
    print "[$(date '+%Y-%m-%d %H:%M:%S')] Server exited (code ${EXIT_CODE}). Restarting in 3s..."
    sleep 3
done
