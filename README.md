# mlx-turboquant-server

One-command setup and launch for running any MLX model as an OpenAI-compatible server on Apple Silicon, with [TurboQuant](https://github.com/arozanov/turboquant-mlx) KV cache compression.

## What it does

- Installs Python venv, all dependencies, and downloads the model automatically
- Runs `mlx_lm.server` with TurboQuant 3-bit KV cache (4.6x compression, ~98% FP16 speed)
- Exposes an OpenAI-compatible API on your local network
- Auto-restarts on crash
- Idempotent — safe to re-run, skips steps already done

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14+
- ~20 GB free disk (for the default Qwen3.6-35B model)
- Python 3.10+ (`brew install python@3.12` or [uv](https://github.com/astral-sh/uv))

## Usage

```bash
# Clone
git clone https://github.com/manish026/mlx-turboquant-server.git
cd mlx-turboquant-server
chmod +x setup.sh

# Run with default model (Qwen3.6-35B-A3B-4bit, auto-downloaded)
./setup.sh

# Run with any MLX model from HuggingFace (auto-downloaded)
./setup.sh mlx-community/Qwen3-8B-Instruct-4bit
./setup.sh mlx-community/Mistral-7B-Instruct-v0.3-4bit
./setup.sh mlx-community/Llama-3.2-3B-Instruct-4bit

# Already have a model on disk? Drag and drop the folder into Terminal:
./setup.sh /path/to/your/model
```

On first run it will install everything and download the model. Subsequent runs start the server immediately.

## Connect from other machines

After the server starts, it prints the URL:

```
http://<your-mac-ip>:8080
```

Use it as an OpenAI-compatible endpoint in any client (Qwen Code, Open WebUI, Continue.dev, etc.):

```bash
curl http://<your-mac-ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3.6-35B-A3B-4bit", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLX_MODEL_DIR` | `~/.cache/mlx-models/<repo>` | Override model storage path |
| `MLX_VENV_DIR` | `~/.mlx-env` | Python venv path |
| `MLX_PORT` | `8080` | Server port |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## What's included

- **TurboQuant 3-bit KV cache** — compresses attention KV cache, dramatically reducing memory usage for long contexts
- **Tool call streaming fix** — patches `mlx_lm/server.py` to correctly deliver tool calls in the final stream chunk (fixes compatibility with OpenAI-style agentic clients)
- **Auto-detected thinking mode** — `enable_thinking` is set automatically for Qwen3 models
- **Auto-restart loop** — server respawns automatically on crash

## Memory profile (M4 Max 36 GB, Qwen3.6-35B-A3B-4bit)

| Component | Size |
|---|---|
| Model weights (4-bit) | ~18 GB |
| TurboQuant KV cache (3-bit, 32K tokens) | ~250 MB |
| Prompt cache (2 sequences) | ~1.75 GB |
| OS + Metal runtime | ~5 GB |
| **Total** | **~26 GB** |

## Credits

- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) — TurboQuant KV cache compression
- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX inference server
