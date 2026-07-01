# vLLM model lifecycle — rtx-5090-linux (single RTX 5090, 32 GB)
#
# ONE model is loaded at a time (single GPU). Every `up-*` target first tears
# down whatever is running, then starts the requested model as a Docker
# container named `vllm`, serving an OpenAI-compatible API on 127.0.0.1:8000.
# All three targets publish to this one endpoint; only the currently-loaded
# served-model-name answers, so a client addresses whichever model is up.
#
#   make up-qwen3.6-27b        # dense 27B, FP8
#   make up-qwen3.6-35b-a3b    # 35B MoE, GPTQ-Int4
#   make up-gemma4-26b         # 26B MoE, FP8-dynamic
#   make down                  # stop + free the GPU
#   make logs / ps / status    # observe
#
# First start of a model downloads weights into the `vllm-hf` docker volume
# (tens of GB) — watch `make logs` until you see "Application startup complete".

SHELL      := /usr/bin/env bash
CONTAINER  := vllm
PORT       := 8000
HF_VOLUME  := vllm-hf

# Blackwell (sm_120) needs a recent CUDA 12.8+/13.0 build. Override in .env if you
# pin a specific tag. Gemma 4 needs the dedicated build that ships its kernels.
VLLM_IMAGE  ?= vllm/vllm-openai:latest
GEMMA_IMAGE ?= vllm/vllm-openai:gemma4-0505-cu130

# .env holds HF_TOKEN (for gated repos) and any *_IMAGE overrides. Optional.
ENV_FILE := $(CURDIR)/.env
ENV_ARG  := $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE),)

# Common docker run flags. Port bound to loopback only — front with a proxy if remote.
DOCKER_RUN = docker run -d --name $(CONTAINER) --restart unless-stopped \
	--gpus all --ipc=host \
	-p 127.0.0.1:$(PORT):8000 \
	-v $(HF_VOLUME):/root/.cache/huggingface \
	$(ENV_ARG)

# Flags shared by every vLLM invocation.
SERVE_COMMON = --host 0.0.0.0 --enable-prefix-caching

.DEFAULT_GOAL := help

## ----------------------------------------------------------------------------
## Model targets (each replaces whatever is currently loaded)
## ----------------------------------------------------------------------------

.PHONY: up-qwen3.6-27b
up-qwen3.6-27b: down ## dense 27B — nvidia/Qwen3.6-27B-NVFP4
	$(DOCKER_RUN) $(VLLM_IMAGE) \
	  --model nvidia/Qwen3.6-27B-NVFP4 \
	  --served-model-name qwen3.6-27b \
	  --max-model-len 262144 \
	  --gpu-memory-utilization 0.90 \
	  --kv-cache-dtype fp8 \
	  --reasoning-parser qwen3 \
	  $(SERVE_COMMON)
	@echo "→ qwen3.6-27b starting. Follow: make logs"

.PHONY: up-qwen3.6-35b-a3b
up-qwen3.6-35b-a3b: down ## 35B MoE — nvidia/Qwen3.6-35B-A3B-NVFP4
	$(DOCKER_RUN) $(VLLM_IMAGE) \
	  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
	  --served-model-name qwen3.6-35b-a3b \
	  --max-model-len 262144 \
	  --gpu-memory-utilization 0.92 \
	  --kv-cache-dtype fp8 \
	  --reasoning-parser qwen3 \
	  $(SERVE_COMMON)
	@echo "→ qwen3.6-35b-a3b starting. Follow: make logs"

.PHONY: up-gemma4-26b
up-gemma4-26b: down ## 26B MoE — nvidia/Gemma-4-26B-A4B-NVFP4
	$(DOCKER_RUN) $(GEMMA_IMAGE) \
	  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
	  --served-model-name gemma4-26b \
	  --max-model-len 131072 \
	  --gpu-memory-utilization 0.90 \
	  --kv-cache-dtype fp8 \
	  $(SERVE_COMMON)
	@echo "→ gemma4-26b starting. Follow: make logs"

## ----------------------------------------------------------------------------
## Lifecycle / observability
## ----------------------------------------------------------------------------

.PHONY: down
down: ## Stop and remove the vLLM container (frees the GPU)
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 && echo "→ stopped $(CONTAINER)" || echo "→ nothing running"

.PHONY: logs
logs: ## Follow container logs
	docker logs -f $(CONTAINER)

.PHONY: ps
ps: ## Show the vLLM container
	@docker ps --filter name=$(CONTAINER) --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

.PHONY: status
status: ## Show loaded model + GPU usage
	@echo "== container ==";  docker ps --filter name=$(CONTAINER) --format '{{.Names}}\t{{.Status}}' || true
	@echo "== served model =="; curl -s http://127.0.0.1:$(PORT)/v1/models | (command -v jq >/dev/null && jq -r '.data[].id' || cat) || echo "(endpoint not up)"
	@echo "== gpu =="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z0-9._-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
