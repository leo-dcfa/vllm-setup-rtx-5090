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
# needs a current build (>= the PRs that added ENABLE_NVFP4_SM120 /
# ENABLE_CUTLASS_MOE_SM120): use cu130-nightly. Stable release tags are cu129
# only and may lack the prebuilt sm_120 FP4 kernels. HF weights are shared from
# the host cache (~/.cache/huggingface); the torch.compile cache is shared from
# ~/.cache/vllm so cold starts don't recompile every launch.
# Prereq: Docker + NVIDIA Container Toolkit (already set up on this box).

SHELL      := /usr/bin/env bash
PORT       := 8000
NAME       := vllm-local
IMAGE      := vllm/vllm-openai:cu130-nightly   # override with VLLM_IMAGE in .env
HF_CACHE   := $(HOME)/.cache/huggingface
VLLM_CACHE := $(HOME)/.cache/vllm
ENV_FILE   := $(CURDIR)/.env

# Common flags every model shares. --host 0.0.0.0 inside the container; the port
# is published to 127.0.0.1 on the host, so it stays local-only.
# --kv-cache-dtype fp8 ~doubles KV headroom (=> more context); prefix caching
# reuses shared prompt prefixes across requests.
SERVE_COMMON = --host 0.0.0.0 --port $(PORT) --kv-cache-dtype fp8 --enable-prefix-caching

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
up-qwen3.6-27b: ## 27B hybrid Mamba — sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP (~15GB, text-only)
	# NOT the nvidia/Qwen3.6-27B-NVFP4 checkpoint: that one keeps the vision tower +
	# lm_head in BF16, and on this build it loads at ~30GB and OOMs on 32GB no matter
	# the context/flags. This is the TEXT-ONLY NVFP4 sibling (vision stripped) — ~15GB,
	# validated at 200K ctx on a single 5090. It is also hybrid (48/64 layers are
	# Mamba SSM), so --max-num-batched-tokens 8192 is required (like the 35B).
	# --language-model-only is still needed: the architecture is the multimodal
	# Qwen3_5ForConditionalGeneration wrapper, so without it vLLM tries to load an
	# image processor and fails (the checkpoint stripped the vision preprocessor).
	# MTP head is present for optional spec decoding (add --speculative-config, ~1.7x).
	$(call serve,sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP,qwen3.6-27b,--max-model-len 196608 --max-num-seqs 2 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml)

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
