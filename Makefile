# vLLM model lifecycle — rtx-5090-linux (single RTX 5090, 32 GB), Docker (cu130)
#
# Runs vLLM in Docker using the Blackwell cu130 NIGHTLY image, which ships
# PREBUILT sm_120 (RTX 5090) NVFP4 kernels AND — unlike the older frozen
# gemma4-0505-cu130 tag — recognizes sm_120 in the FP4 backend-selection logic,
# so NVFP4 runs on the NATIVE FlashInfer-CUTLASS FP4 path instead of the Marlin
# weight-only fallback (which forfeits FP4 tensor-core FLOPS and has a
# negative-scale gibberish bug on sm_12x). No runtime JIT compilation.
# ONE model at a time (single 32 GB GPU): every `up-*` target stops whatever is
# running, then launches a container serving an OpenAI-compatible API on
# 127.0.0.1:8000.
#
#   make up-qwen3.6-27b        # 27B hybrid Mamba, NVFP4 (text-only), 192K ctx
#   make up-gemma4-26b         # 26B MoE, NVFP4, 256K ctx
#   (35B MoE target is dropped — see the commented block below for why/how to restore)
#   make down                  # stop + free the GPU
#   make logs / status / health
#
# Image is pinned via VLLM_IMAGE in .env (default below). Native sm_120 NVFP4
# needs a current build. BEWARE: the cu130-nightly tag upstream is DEAD (frozen
# 2026-04-23) — the rolling tag is now plain "nightly", and `docker pull` must be
# run explicitly (a locally cached tag is never re-pulled, so "nightly" rots
# silently; symptom: brand-new quants fail weight loading, e.g. missing
# lm_head.weight_scale). Run `make pull` to refresh. HF weights are shared from
# the host cache (~/.cache/huggingface); the torch.compile cache is shared from
# ~/.cache/vllm so cold starts don't recompile every launch.
# Prereq: Docker + NVIDIA Container Toolkit (already set up on this box).

SHELL      := /usr/bin/env bash
PORT       := 8000
NAME       := vllm-local
IMAGE      := vllm/vllm-openai:nightly   # override with VLLM_IMAGE in .env
HF_CACHE   := $(HOME)/.cache/huggingface
VLLM_CACHE := $(HOME)/.cache/vllm
ENV_FILE   := $(CURDIR)/.env

# Common flags every model shares. --host 0.0.0.0 inside the container; the port
# is published to 127.0.0.1 on the host, so it stays local-only.
# --kv-cache-dtype fp8 ~doubles KV headroom (=> more context); prefix caching
# reuses shared prompt prefixes across requests.
SERVE_COMMON = --host 0.0.0.0 --port $(PORT) --kv-cache-dtype fp8 --enable-prefix-caching

# MTP self-speculative decoding (draft head baked into the checkpoint).
# Kept in a variable because the JSON commas would split $(call ...) arguments.
MTP_SPEC := --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'

# $(call serve,<model>,<served-name>,<extra flags>)
define serve
	@$(MAKE) --no-print-directory down
	@set -a; [ -f $(ENV_FILE) ] && . $(ENV_FILE); set +a; \
	IMG=$${VLLM_IMAGE:-$(IMAGE)}; \
	echo "→ $(2) starting in Docker ($$IMG). Weights load from host HF cache — follow: make logs"; \
	docker run -d --name $(NAME) --gpus all --ipc=host --restart unless-stopped \
	  -p 127.0.0.1:$(PORT):$(PORT) \
	  -v $(HF_CACHE):/root/.cache/huggingface \
	  -v $(VLLM_CACHE):/root/.cache/vllm \
	  -e PYTORCH_CUDA_ALLOC_CONF=$${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
	  $${HF_TOKEN:+-e HF_TOKEN=$$HF_TOKEN} \
	  $$IMG \
	  $(1) --served-model-name $(2) $(SERVE_COMMON) $(3) >/dev/null \
	  && echo "→ container $(NAME) started (pid inside docker)" \
	  || echo "→ docker run failed"
endef

.DEFAULT_GOAL := help

## ----------------------------------------------------------------------------
## Model targets (each replaces whatever is currently loaded)
## ----------------------------------------------------------------------------

# --max-model-len is set to the checkpoints' NATIVE max (262144). The full-
# attention Qwen models may not fit 256K of KV on 32 GB — if startup fails with
# "max seq len (262144) is larger than the maximum number of tokens that can be
# stored in KV cache (N)", set --max-model-len to N (measured ceilings below).
# gemma4 uses sliding-window attention (cheap KV) and should reach full 256K.

.PHONY: up-qwen3.6-27b
up-qwen3.6-27b: ## 27B hybrid Mamba — unsloth/Qwen3.6-27B-NVFP4 (~23GB, FP8 attn + NVFP4 MLP, MTP spec decode)
	# Unsloth quant (2026-07-10): mixed compressed-tensors — attention + head in FP8,
	# MLPs in NVFP4, calibrated with fp8 KV cache. Higher quality than the pure-FP4
	# sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP checkpoint it replaced, but ~23GB
	# loaded vs ~17.6GB, so max context drops from 192K. Unlike the nvidia/ mixed
	# checkpoint (BF16 vision tower + lm_head, ~30GB, OOMs), this one quantizes the
	# big pieces and fits. Vision tower is present but skipped
	# (--language-model-only); drop that flag to enable image input at a context cost.
	# Hybrid model (48/64 layers linear-attention/SSM): --max-num-seqs 2 and
	# a --max-num-batched-tokens bound remain load-bearing (see README). 4096
	# (down from 8192, 2026-07-14): 8192-token prefill chunks OOM'd mid-request
	# at util 0.94 — the fp4_gemm fallback workspace for ~8K shapes isn't covered
	# by startup profiling and the headroom is <300MB. Halving the chunk halves
	# the activation spike; util 0.94 itself must not be lowered (context ceiling).
	# $(MTP_SPEC) enables the checkpoint's fixed MTP head as its own draft model
	# (Unsloth-recommended, ~1.4-2.2x decode speedup); remove it if boot OOMs.
	# util 0.94 is load-bearing: at 0.90 the measured per-request ceiling is only
	# ~99K tokens; at 0.94 it is 137,600 (measured, nightly 2026-07-11 — don't
	# trust the larger "GPU KV cache size" log line: on this hybrid model the
	# Mamba state pages share that pool, so pool-tokens ≠ context). 131072 boots
	# with 1.35x concurrency; 163840 does NOT fit. Needs the CURRENT "nightly"
	# image — older builds fail loading with "no module or parameter named
	# 'lm_head.weight_scale'" (quantized lm_head support is recent).
	$(call serve,unsloth/Qwen3.6-27B-NVFP4,qwen3.6-27b,--max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.94 --reasoning-parser qwen3 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml $(MTP_SPEC))

# DROPPED — 35B hybrid attn+SSM MoE. The nvidia/Qwen3.6-35B-A3B-NVFP4 checkpoint
# OOMs at ~30GB during init on this 32GB card, and is redundant next to the
# dense 27B-NVFP4 (better quality + longer context) and gemma4-26b. It CAN run on
# a single 5090 (~96K ctx, ~117 tok/s) — but only via the RedHatAI checkpoint plus
# the experimental Mamba-prefix-caching path. To restore, uncomment and use:
#   nvidia repo -> RedHatAI/Qwen3.6-35B-A3B-NVFP4 (needs ~18GB download)
#   flags: --max-model-len 98304 --max-num-seqs 2 --max-num-batched-tokens 8192 \
#          --gpu-memory-utilization 0.94 --enable-expert-parallel \
#          --reasoning-parser qwen3 --language-model-only \
#          --enable-auto-tool-choice --tool-call-parser qwen3_xml
# .PHONY: up-qwen3.6-35b-a3b
# up-qwen3.6-35b-a3b: ## 35B hybrid attn+SSM MoE — RedHatAI/Qwen3.6-35B-A3B-NVFP4 (~96K ctx)
# 	$(call serve,RedHatAI/Qwen3.6-35B-A3B-NVFP4,qwen3.6-35b-a3b,--max-model-len 98304 --max-num-seqs 2 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.94 --enable-expert-parallel --reasoning-parser qwen3 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml)

.PHONY: up-gemma4-26b
up-gemma4-26b: ## 26B MoE — nvidia/Gemma-4-26B-A4B-NVFP4
	$(call serve,nvidia/Gemma-4-26B-A4B-NVFP4,gemma4-26b,--max-model-len 262144 --gpu-memory-utilization 0.92 --language-model-only --enable-auto-tool-choice --tool-call-parser gemma4)

## ----------------------------------------------------------------------------
## Lifecycle / observability
## ----------------------------------------------------------------------------

.PHONY: pull
pull: ## Refresh the vLLM image (docker never re-pulls a cached tag on its own)
	@set -a; [ -f $(ENV_FILE) ] && . $(ENV_FILE); set +a; \
	IMG=$${VLLM_IMAGE:-$(IMAGE)}; \
	docker pull $$IMG && docker inspect $$IMG --format 'image built: {{.Created}}'

.PHONY: down
down: ## Stop + remove the running vLLM container (frees the GPU)
	@docker rm -f $(NAME) >/dev/null 2>&1 && echo "→ stopped $(NAME)" || echo "→ nothing running"

.PHONY: logs
logs: ## Follow the container log
	@docker logs -f $(NAME)

.PHONY: status
status: ## Show running container + served model + GPU usage
	@docker ps --filter name=$(NAME) --format '  container: {{.Names}}  {{.Status}}' | grep . || echo "  container: not running"
	@echo "served model:"; curl -s -m 5 http://127.0.0.1:$(PORT)/v1/models | (command -v jq >/dev/null && jq -r '.data[].id' || cat) 2>/dev/null || echo "  (endpoint not up)"
	@echo "gpu:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

.PHONY: health
health: ## One-shot health check (HTTP code)
	@curl -s -m 5 -o /dev/null -w "health: %{http_code}\n" http://127.0.0.1:$(PORT)/health

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z0-9._-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
