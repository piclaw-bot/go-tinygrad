TMPDIR ?= /workspace/tmp
GOTMPDIR ?= /workspace/tmp
export TMPDIR GOTMPDIR

.PHONY: all build test clean server chat gen vet

all: build

build: gen server chat

gen:
	go build -o bin/llmgen ./cmd/llmgen

server:
	go build -o bin/llmserver ./cmd/llmserver

chat:
	go build -o bin/llmchat ./cmd/llmchat

test:
	go test -count=1 -timeout=120s ./loader/safetensors/... ./model/... ./simd/... ./tensor/...

vet:
	go vet ./...

clean:
	rm -rf bin/

# GPU-heavy tests (require GEMMA4_TRACE_TEST=1 and GPU)
test-gpu:
	GEMMA4_TRACE_TEST=1 go test -count=1 -run TestGemma4GPUBench ./model -v

# Quick smoke test
smoke:
	@echo "=== build ==="
	go build -o /dev/null ./cmd/llmgen
	go build -o /dev/null ./cmd/llmserver
	go build -o /dev/null ./cmd/llmchat
	@echo "=== vet ==="
	go vet ./...
	@echo "=== unit tests ==="
	go test -count=1 -timeout=60s ./loader/safetensors/... ./simd/... ./tensor/...
	@echo "=== ok ==="
