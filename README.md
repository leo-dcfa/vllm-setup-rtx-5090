# vLLM model serving — rtx-5090-linux

Local vLLM on this box (`rtx-5090-linux`, RTX 5090 32 GB), serving an
OpenAI-compatible API on `127.0.0.1:8000`.

**One model is loaded at a time** — a single 32 GB GPU can't hold two of these at
once. Every `make up-*` tears down the running container first. All three targets
publish to the same `127.0.0.1:8000` endpoint; only the currently-loaded
served-model-name answers, so a client addresses whichever model is up.

## Models

| Served model name | HF repo | Quant | Start ctx | Weights |
|---|---|---|---|---|
| `qwen3.6-27b`     | `nvidia/Qwen3.6-27B-NVFP4`      | NVFP4 (4-bit FP) | 262144 | ~14 GB |
| `qwen3.6-35b-a3b` | `nvidia/Qwen3.6-35B-A3B-NVFP4`  | NVFP4 (4-bit FP) | 262144 | ~17 GB |
| `gemma4-26b`      | `nvidia/Gemma-4-26B-A4B-NVFP4`  | NVFP4 (4-bit FP) | 131072 | ~16.5 GB |

Native max is 262144 for all three. vLLM fails **at startup** if KV won't fit —
just lower `--max-model-len` in the Makefile and re-run. `--kv-cache-dtype fp8`
(near-lossless on Blackwell) is on everywhere to roughly double KV headroom.

**Quant rationale (32 GB Blackwell):** NVFP4 is a 4-bit *float* format with
two-level block scaling, run on the RTX 5090's native FP4 tensor cores — "BF16
accuracy at INT4 density" (~2% eval degradation, ~1.6× faster than BF16, ~1.3×
faster than INT4/GPTQ). Unlike INT4 (GPTQ/AWQ), it's near-lossless. Halving the
weights vs. FP8 means all three now fit with **full-length context** on a single
32 GB card. Requires a Blackwell-capable vLLM image (see below).

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
