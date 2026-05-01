package model

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-tinygrad/gpu"
)

var (
	gpuAvailable bool
	gpuInitOnce  sync.Once
	gpuWeights   map[uintptr]*gpu.Buffer
	gpuMu        sync.Mutex
	gpuXBuf      *gpu.Buffer
	gpuOutBuf    *gpu.Buffer
	gpuMaxDim    int
	gpuUploaded  int
)

func initGPU() {
	gpuAvailable = gpu.SgemmReady()
	if gpuAvailable {
		gpuWeights = make(map[uintptr]*gpu.Buffer)
		fmt.Println("[model] GPU SGEMM available for inference")
	}
}

func gpuEnsureBufs(maxN int) {
	if maxN <= gpuMaxDim { return }
	if gpuXBuf != nil { gpuXBuf.Free() }
	if gpuOutBuf != nil { gpuOutBuf.Free() }
	var err error
	gpuXBuf, err = gpu.Malloc(maxN)
	if err != nil { gpuAvailable = false; return }
	gpuOutBuf, err = gpu.Malloc(maxN)
	if err != nil { gpuAvailable = false; return }
	gpuMaxDim = maxN
}

func gpuUploadWeight(data []float32) *gpu.Buffer {
	key := uintptr(unsafe.Pointer(&data[0]))
	gpuMu.Lock()
	if buf, ok := gpuWeights[key]; ok {
		gpuMu.Unlock()
		return buf
	}
	gpuMu.Unlock()

	buf, err := gpu.Malloc(len(data))
	if err != nil { return nil }
	if err := buf.Upload(data); err != nil { buf.Free(); return nil }

	gpuMu.Lock()
	gpuWeights[key] = buf
	gpuUploaded++
	if gpuUploaded == 1 {
		fmt.Println("[model] GPU: uploading weights to VRAM...")
	}
	gpuMu.Unlock()
	return buf
}

func gemvGPU(out, x []float32, w []float32, inDim, outDim int) bool {
	gpuInitOnce.Do(initGPU)
	if !gpuAvailable { return false }
	if true { return false } // GPU gemv disabled: per-call overhead exceeds benefit for Mx1

	maxN := outDim
	if inDim > maxN { maxN = inDim }
	gpuEnsureBufs(maxN)
	if !gpuAvailable { return false }

	wBuf := gpuUploadWeight(w)
	if wBuf == nil { return false }

	gpuXBuf.Upload(x[:inDim])
	gpu.Sgemm(outDim, 1, inDim, 1.0, wBuf, gpuXBuf, gpuOutBuf)
	gpu.Sync()
	gpuOutBuf.Download(out[:outDim])
	return true
}
