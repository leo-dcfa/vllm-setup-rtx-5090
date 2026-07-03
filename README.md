# vLLM model serving — rtx-5090-linux

Local vLLM on this box (`rtx-5090-linux`, RTX 5090 32 GB), serving an
OpenAI-compatible API on `127.0.0.1:8000`.

**One model is loaded at a time** — a single 32 GB GPU can't hold two of these at
once. Every `make up-*` tears down the running container first. All three targets
publish to the same `127.0.0.1:8000` endpoint; only the currently-loaded
served-model-name answers, so a client addresses whichever model is up.

## Models

| Served model name | HF repo | Quant | Max ctx | Weights (loaded) |
|---|---|---|---|---|
| `qwen3.6-27b`     | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | NVFP4 (pure `modelopt_fp4`, text-only) | 196608 | ~17.6 GB |
| `gemma4-26b`      | `nvidia/Gemma-4-26B-A4B-NVFP4`             | NVFP4 (`modelopt_fp4`)                 | 262144 | ~17 GB |

Both run the **native** FlashInfer-CUTLASS FP4 path on the RTX 5090 and are
verified serving coherent output. `--kv-cache-dtype fp8` is on everywhere to
~double KV headroom. gemma4 uses sliding-window attention (cheap KV) so it holds
full **256K**; the 27B is a hybrid Mamba model measured at **192K** (1.14×
concurrency — a full 196608-token request fits). To go higher/lower, adjust
`--max-model-len`; if a boot fails with `... larger than the maximum number of
tokens that can be stored in KV cache (N)`, set it to `N`.

> The 35B (`qwen3.6-35b-a3b`) target was **dropped** — see the commented block in
> the Makefile. The `nvidia/` checkpoint OOMs at ~30 GB on this card, and it's
> redundant next to the 27B + gemma4. It *can* run via `RedHatAI/...` at ~96K if
> you ever want it back (restore recipe is in the Makefile).

**Notes on NVFP4 on this box (learned the hard way):**
- **Native FP4 needs a current image.** On a current `cu130-nightly` build vLLM
  recognizes sm_120 and runs the native **FlashInfer-CUTLASS FP4** path (real FP4
  tensor cores). The old `gemma4-0505-cu130` tag fell back to the **Marlin** kernel
  (`GPU does not have native support for FP4`) — weight-only dequant, no FP4 FLOPS,
  plus a *negative-scale* bug that could emit empty/gibberish tokens. That fallback
  is the "NVFP4 issue" people warn about; the fix is the image, not a flag.
- **Pick pure `modelopt_fp4`, not `modelopt_mixed`.** The `nvidia/` 27B & 35B
  checkpoints are `modelopt_mixed` (vision tower + lm_head in BF16). On this build
  they load at ~30 GB and OOM on 32 GB *regardless* of context/util — the vision
  BF16 weights don't shed. The text-only pure-FP4 27B loads at 17.6 GB and fits
  with 192K. gemma4 is pure `modelopt_fp4` and Just Works.
- **Hybrid Mamba models need two flags.** The 27B (and 35B) are hybrid attention +
  Mamba/SSM. `--max-num-seqs 2` is load-bearing: the SSM state cache is sized by
  max-num-seqs (×48 layers) and is *not* gated by `--gpu-memory-utilization`, so the
  default (~1024) alone eats ~10 GB and OOMs. `--max-num-batched-tokens 8192` bounds
  the profiling pass (a Mamba assertion / OOM without it when fp8 KV cache is on).

**Quant rationale:** NVFP4 is a 4-bit *float* format with two-level block scaling
(~2% eval degradation, near-lossless vs. INT4). There is no fast 6-bit path in
vLLM (MXFP6 is research-grade, GGUF Q6 is slow/unaccelerated); the real step up
from NVFP4 is FP8, which costs most of your context on 32 GB. It needs a current
Blackwell `cu130-nightly` image with the sm_120 FP4 kernels.

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

- **More/less context** → adjust `--max-model-len`. If startup fails on KV, the
  error names the exact token ceiling that fits; set `--max-model-len` to it.
- **Confirm native FP4** → `make logs` should **not** print `GPU does not have
  native support for FP4` or `NVFP4 Marlin ... negative scales`. If it does,
  you're on an old image — bump `VLLM_IMAGE` to a current `cu130-nightly`.
- **Faster cold starts** → the torch.compile cache is persisted at `~/.cache/vllm`
  (mounted into the container), so the ~40s compile only happens on the first
  launch of each model/flag combo. KV-cache profiling still runs each boot.
- **Max quality on the 27B** → swap the repo to `Qwen/Qwen3.6-27B-FP8` (8-bit,
  ~28 GB) for the absolute quality ceiling, at the cost of context (~32–64K).
- **NVFP4 not auto-detected** → vLLM reads the format from `config.json`; do
  **not** pass `--quantization`. Only if detection fails, add
  `--quantization modelopt_fp4`.
- **Blackwell image** → native sm_120 NVFP4 needs a current build. Use
  `vllm/vllm-openai:cu130-nightly`; pin via `VLLM_IMAGE` in `.env`.
- **Save VRAM (text-only)** → `--language-model-only` (already on) skips the
  vision encoder; dropping it enables image input but costs context.

## Files

- `Makefile` — model lifecycle (`up-*`, `down`, `logs`, `status`)
- `.env` — optional `HF_TOKEN` / image overrides (gitignored, chmod 600)
- `.env.example` — template
