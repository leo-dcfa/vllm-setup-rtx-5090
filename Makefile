# vLLM model lifecycle — rtx-5090-linux (single RTX 5090, 32 GB), Docker (cu130)
#
# Runs vLLM in Docker using the Blackwell cu130 image, which ships PREBUILT
# sm_120 NVFP4/FlashInfer kernels — so there is NO runtime JIT compilation
# (the venv path failed to JIT-build the sm120 GEMM: nvcc/ninja/CCCL mismatch).
# ONE model at a time (single 32 GB GPU): every `up-*` target stops whatever is
# running, then launches a container serving an OpenAI-compatible API on
# 127.0.0.1:8000.
#
#   make up-qwen3.6-27b        # dense 27B, NVFP4
#   make up-qwen3.6-35b-a3b    # 35B MoE, NVFP4
#   make up-gemma4-26b         # 26B MoE, NVFP4
#   make down                  # stop + free the GPU
#   make logs / status / health
#
# Image is pinned via VLLM_IMAGE in .env (default below); the cu129/:latest image
# fails NVFP4 engine init on sm_120, so use a cu130 tag. HF weights are shared
# from the host cache (~/.cache/huggingface) so models download only once.
# Prereq: Docker + NVIDIA Container Toolkit (already set up on this box).

SHELL     := /usr/bin/env bash
PORT      := 8000
NAME      := vllm-local
IMAGE     := vllm/vllm-openai:gemma4-0505-cu130   # override with VLLM_IMAGE in .env
HF_CACHE  := $(HOME)/.cache/huggingface
ENV_FILE  := $(CURDIR)/.env

# Common flags every model shares. --host 0.0.0.0 inside the container; the port
# is published to 127.0.0.1 on the host, so it stays local-only.
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

.PHONY: up-qwen3.6-27b
up-qwen3.6-27b: ## dense 27B — cyankiwi/Qwen3.6-27B-AWQ-INT4 (~14GB; NVFP4 OOMs in this image)
	$(call serve,cyankiwi/Qwen3.6-27B-AWQ-INT4,qwen3.6-27b,--max-model-len 65536 --gpu-memory-utilization 0.85 --reasoning-parser qwen3 --language-model-only --enforce-eager --enable-auto-tool-choice --tool-call-parser qwen3_xml)

.PHONY: up-qwen3.6-35b-a3b
up-qwen3.6-35b-a3b: ## 35B MoE — nvidia/Qwen3.6-35B-A3B-NVFP4
	$(call serve,nvidia/Qwen3.6-35B-A3B-NVFP4,qwen3.6-35b-a3b,--max-model-len 131072 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --language-model-only --enable-auto-tool-choice --tool-call-parser hermes)

.PHONY: up-gemma4-26b
up-gemma4-26b: ## 26B MoE — nvidia/Gemma-4-26B-A4B-NVFP4
	$(call serve,nvidia/Gemma-4-26B-A4B-NVFP4,gemma4-26b,--max-model-len 65536 --gpu-memory-utilization 0.90 --language-model-only --enable-auto-tool-choice --tool-call-parser gemma4)

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
