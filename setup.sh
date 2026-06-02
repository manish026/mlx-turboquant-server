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

# Accept drag-and-drop local path OR a HuggingFace repo ID
# Drag-and-drop from Finder pastes the full path — strip any trailing slash/space
ARG="${1:-}"
ARG="${ARG%/}"           # strip trailing slash
ARG="${ARG## }"          # strip leading space (drag-and-drop sometimes adds one)
ARG="${ARG%% }"          # strip trailing space

if [[ -z "$ARG" ]]; then
    # No argument — use default HF repo
    MODEL_REPO="$DEFAULT_MODEL_REPO"
    MODEL_DIR="${MLX_MODEL_DIR:-${HOME}/.cache/mlx-models/${MODEL_REPO}}"
    LOCAL_MODEL=false
elif [[ "$ARG" == /* || "$ARG" == ~* || "$ARG" == ./* ]]; then
    # Looks like a local path (drag-and-drop or absolute path)
    MODEL_DIR="${MLX_MODEL_DIR:-${ARG/#\~/$HOME}}"   # expand ~ if present
    MODEL_REPO="${MODEL_DIR##*/}"                     # use folder name as model name
    LOCAL_MODEL=true
else
    # HuggingFace repo ID e.g. mlx-community/Qwen3-8B-4bit
    MODEL_REPO="$ARG"
    MODEL_DIR="${MLX_MODEL_DIR:-${HOME}/.cache/mlx-models/${MODEL_REPO}}"
    LOCAL_MODEL=false
fi

MODEL_NAME="${MODEL_REPO##*/}"                        # e.g. Qwen3.6-35B-A3B-4bit
VENV_DIR="${MLX_VENV_DIR:-${HOME}/.mlx-env}"
PORT="${MLX_PORT:-8080}"
MAX_TOKENS=32768

MLX_LM_VERSION="0.31.3"
TURBOQUANT_REPO="https://github.com/arozanov/turboquant-mlx.git"
TURBOQUANT_COMMIT="7c6e3dc9459936852e19e571201d35d07d65e120"

SCRIPT_DIR="${0:A:h}"                                          # directory of this script
LAUNCHER="${SCRIPT_DIR}/launcher.py"

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
    mkdir -p "$VENV_DIR"
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
[[ -f "$LAUNCHER" ]] || die "launcher.py not found at ${LAUNCHER} — did you git clone the full repo?"
ok "Launcher found at ${LAUNCHER}."

# ── 8. Download model ─────────────────────────────────────────────────────────
if [[ "$LOCAL_MODEL" == true ]]; then
    [[ -f "${MODEL_DIR}/config.json" ]] || die "No config.json found at ${MODEL_DIR} — is this a valid MLX model folder?"
    ok "Using local model at ${MODEL_DIR}."
elif [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    log "Downloading model ${MODEL_REPO} to ${MODEL_DIR} (this may take a while)..."
    mkdir -p "$(dirname "$MODEL_DIR")"
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

# Free the port if something is already using it
if lsof -ti :"$PORT" &>/dev/null; then
    warn "Port ${PORT} in use — killing existing process..."
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null
    sleep 1
fi

# Auto-detect thinking mode — only Qwen3 models support enable_thinking
CHAT_TEMPLATE_ARGS="{}"
if [[ "$MODEL_REPO" == *"Qwen3"* ]]; then
    CHAT_TEMPLATE_ARGS='{"enable_thinking":false}'
fi

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<your-ip>")
LOG_FILE="${SCRIPT_DIR}/server.log"

print ""
ok "Model:  ${MODEL_REPO}"
ok "Server: http://${LOCAL_IP}:${PORT}"
ok "Log:    ${LOG_FILE}  (tail -f ${LOG_FILE})"
print ""

while true; do
    print "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server — model: ${MODEL_NAME}..."
    "$PY" "$LAUNCHER" \
        --model    "$MODEL_DIR" \
        --host     0.0.0.0 \
        --port     "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --prompt-cache-size 5 \
        --temp     0.1 \
        --prefill-step-size 2048 \
        --decode-concurrency 2 --prompt-concurrency 1 \
        --chat-template-args "$CHAT_TEMPLATE_ARGS" \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${pipestatus[1]}
    print "[$(date '+%Y-%m-%d %H:%M:%S')] Server exited (code ${EXIT_CODE}). Restarting in 3s..."
    sleep 3
done
