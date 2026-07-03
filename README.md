# vLLM model serving — rtx-5090-linux

Local vLLM on this box (`rtx-5090-linux`, RTX 5090 32 GB), serving an
OpenAI-compatible API on `127.0.0.1:8000`.

**One model is loaded at a time** — a single 32 GB GPU can't hold two of these at
once. Every `make up-*` tears down the running container first. All three targets
publish to the same `127.0.0.1:8000` endpoint; only the currently-loaded
served-model-name answers, so a client addresses whichever model is up.

## Models

| Served model name | HF repo | Quant | Start ctx | Weights (loaded) |
|---|---|---|---|---|
| `qwen3.6-27b`     | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ INT4              | 131072 | ~14 GB |
| `qwen3.6-35b-a3b` | `nvidia/Qwen3.6-35B-A3B-NVFP4`  | NVFP4 (mixed FP4/FP8) | 131072 | ~20 GB |
| `gemma4-26b`      | `nvidia/Gemma-4-26B-A4B-NVFP4`  | NVFP4 (mixed FP4/FP8) | 131072 | ~17 GB |

These are **conservative starting** context sizes for a 32 GB card. vLLM fails
**at startup** with a CUDA OOM if the KV cache won't fit — the fix is to lower
`--max-model-len` in the Makefile. After a model boots, `make logs` prints the
actual `GPU KV cache size` in tokens; raise `--max-model-len` toward that if you
want more. `--kv-cache-dtype fp8` is on everywhere to ~double KV headroom.

**Notes on these NVFP4 checkpoints (learned the hard way):**
- They're **multimodal** — the vision tower adds several GB. The targets pass
  `--language-model-only` to drop it (text-only; fine for coding). Remove that
  flag if you need image input, but then you must shrink `--max-model-len`.
- They're **mixed precision** (`ModelOptFp8LinearMethod` on some layers), so they
  load at ~20 GB, not the ~14 GB a pure-4-bit 27B would. That's what forces the
  context down from the native 262144.
- On the RTX 5090, vLLM currently runs these **weight-only via the Marlin kernel**
  (`GPU does not have native support for FP4`) rather than native FP4 tensor cores
  — correct results, slightly below peak throughput.

**Quant rationale:** NVFP4 is a 4-bit *float* format with two-level block scaling
(~2% eval degradation, near-lossless vs. INT4). It needs a Blackwell cu130 vLLM
image (see below); the default cu129 image fails NVFP4 engine init on sm_120.

## Prerequisites (one time, needs sudo)

Docker + the NVIDIA Container Toolkit. Run these once:

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # then log out/in (or `newgrp docker`) for non-root docker

# NVIDIA Container Toolkit (GPU passthrough)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# sanity check
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Usage

```bash
make                       # list targets
make up-qwen3.6-27b        # load a model (downloads weights on first run)
make logs                  # follow startup until "Application startup complete"
make status                # loaded model id + GPU memory
make down                  # stop + free the GPU
```

First start of each model downloads its weights into the `vllm-hf` docker volume
(tens of GB, persists across runs). Test the endpoint directly:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

The port is bound to `127.0.0.1` only. If you need remote access, front it with an
OpenAI-compatible gateway or reverse proxy rather than exposing the port directly.

## Tuning knobs

- **More/less context** → adjust `--max-model-len`. If startup OOMs on KV, lower it.
- **Max quality on the 27B** → swap the repo to `Qwen/Qwen3.6-27B-FP8` (8-bit,
  ~28 GB) for the absolute quality ceiling, at the cost of context (~32–64K).
- **NVFP4 not auto-detected** → if vLLM doesn't pick up the format, add
  `--quantization modelopt_fp4` to the target.
- **Blackwell image** → NVFP4 needs a recent build (the `gemma4-0505-cu130` tag or
  a current `cu128`+ image). Pin via `VLLM_IMAGE` / `GEMMA_IMAGE` in `.env`.
- **Save VRAM (text-only)** → add `--language-model-only` to the Qwen targets to
  skip the vision encoder.

## Files

- `Makefile` — model lifecycle (`up-*`, `down`, `logs`, `status`)
- `.env` — optional `HF_TOKEN` / image overrides (gitignored, chmod 600)
- `.env.example` — template
