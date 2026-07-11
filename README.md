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
| `qwen3.6-27b`     | `unsloth/Qwen3.6-27B-NVFP4`    | FP8 attn/head + NVFP4 MLP (`compressed-tensors`), MTP spec decode | 131072 | ~23 GB |
| `gemma4-26b`      | `nvidia/Gemma-4-26B-A4B-NVFP4` | NVFP4 (`modelopt_fp4`)                                            | 262144 | ~17 GB |

Both run the **native** FlashInfer-CUTLASS FP4 path on the RTX 5090 and are
verified serving coherent output. `--kv-cache-dtype fp8` is on everywhere to
~double KV headroom (the Unsloth quant is explicitly calibrated for fp8 KV).
gemma4 uses sliding-window attention (cheap KV) so it holds full **256K**; the
27B serves **128K** (1.35× concurrency) with a measured per-request ceiling of
**137,600** tokens at util 0.94 (vs only ~99K at util 0.90 — the extra 4% is
load-bearing). Don't read the "GPU KV cache size: 177K tokens" log line as
context: on hybrid Mamba models the state pages share that pool, so
pool-tokens ≠ max-model-len. To go higher/lower, adjust `--max-model-len`; if
a boot fails, the error names the exact ceiling that fits — set it to that.

The 27B is the **Unsloth quant** (2026-07-10): attention + lm_head in FP8, MLPs
in NVFP4, with the checkpoint's (fixed) MTP head enabled as its own speculative
draft (`--speculative-config mtp`, ~1.4–2.2× decode). It replaced
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (pure FP4, ~17.6 GB, reached 192K) —
better quality and speed for ~32K less context.

> The 35B (`qwen3.6-35b-a3b`) target was **dropped** — see the commented block in
> the Makefile. The `nvidia/` checkpoint OOMs at ~30 GB on this card, and it's
> redundant next to the 27B + gemma4. It *can* run via `RedHatAI/...` at ~96K if
> you ever want it back (restore recipe is in the Makefile).

**Notes on NVFP4 on this box (learned the hard way):**
- **Native FP4 needs a current image — and "current" rots silently.** On a
  current `nightly` build vLLM recognizes sm_120 and runs the native
  **FlashInfer-CUTLASS FP4** path (real FP4 tensor cores). The old
  `gemma4-0505-cu130` tag fell back to the **Marlin** kernel (`GPU does not have
  native support for FP4`) — weight-only dequant, no FP4 FLOPS, plus a
  *negative-scale* bug that could emit empty/gibberish tokens. That fallback is
  the "NVFP4 issue" people warn about; the fix is the image, not a flag.
  **Two traps found 2026-07:** (a) the upstream `cu130-nightly` tag is dead —
  frozen at 2026-04-23; the rolling tag is now plain `nightly`; (b) docker never
  re-pulls a tag it has cached, so run `make pull` before expecting new-model
  support (symptom of a stale image: weight-load failure like
  `no module or parameter named 'lm_head.weight_scale'`).
- **Not all "mixed" quants are equal.** The `nvidia/` 27B & 35B checkpoints are
  `modelopt_mixed` (vision tower + lm_head in **BF16**). On this build they load at
  ~30 GB and OOM on 32 GB *regardless* of context/util — the vision BF16 weights
  don't shed. The Unsloth 27B is also mixed, but the mix is FP8 attention/head +
  NVFP4 MLPs (compressed-tensors), so everything big is quantized and it fits
  (~23 GB). gemma4 is pure `modelopt_fp4` and Just Works.
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
make pull                  # refresh the vLLM image (docker won't re-pull on its own)
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
  The Unsloth quant already puts attention + lm_head in FP8, so the gap is
  smaller than it was with pure FP4.
- **More context on the 27B** → the old text-only pure-FP4 checkpoint
  (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, ~17.6 GB) reached **192K**; swap
  it back if you need long context more than the Unsloth quant's quality/speed.
- **Slow generation on the 27B** → check `make logs` for the speculative-decode
  acceptance rate; the MTP draft head should accept most tokens. To disable
  spec decode, remove `$(MTP_SPEC)` from the target.
- **Quant not auto-detected** → vLLM reads the format from `config.json`; do
  **not** pass `--quantization`. (Unsloth 27B = `compressed-tensors`, the others
  = `modelopt_fp4`.) Also don't force an MoE backend — Unsloth reports a ~2.5x
  regression when overriding vLLM's auto-selected (cute-DSL) backend.
- **Blackwell image** → native sm_120 NVFP4 needs a current build. Use
  `vllm/vllm-openai:nightly` (the `cu130-nightly` tag is dead, see above); pin
  via `VLLM_IMAGE` in `.env`, refresh with `make pull`.
- **Save VRAM (text-only)** → `--language-model-only` (already on) skips the
  vision encoder; dropping it enables image input but costs context.

## Files

- `Makefile` — model lifecycle (`up-*`, `down`, `logs`, `status`)
- `.env` — optional `HF_TOKEN` / image overrides (gitignored, chmod 600)
- `.env.example` — template
