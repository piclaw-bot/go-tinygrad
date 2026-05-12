package gpu

// CUDA Streams for overlapped execution.
//
// tinygrad insight: all GPU ops go on the default stream (stream 0),
// and a secondary stream handles weight prefetch. The two overlap on
// the GPU's execution engines (compute + copy).
//
// We add:
//   - CUstream type and creation via purego
//   - Prefetch stream: warms L2 cache for next layer's weights
//   - Event-based sync between streams
//   - CUDA Graph capture and replay

import (
	"fmt"
	"unsafe"
)

// CUstream handle
type CUstream uintptr

// CUevent handle
type CUevent uintptr

// CUgraph / CUgraphExec handles
type CUgraph uintptr
type CUgraphExec uintptr

var (
	// Stream functions
	cuStreamCreate      func(*CUstream, uint32) CUresult
	cuStreamDestroy     func(CUstream) CUresult
	cuStreamSynchronize func(CUstream) CUresult

	// Event functions
	cuEventCreate      func(*CUevent, uint32) CUresult
	cuEventRecord      func(CUevent, CUstream) CUresult
	cuEventSynchronize func(CUevent) CUresult
	cuStreamWaitEvent  func(CUstream, CUevent, uint32) CUresult
	cuEventDestroy     func(CUevent) CUresult

	// Graph functions
	cuStreamBeginCapture func(CUstream, uint32) CUresult
	cuStreamEndCapture   func(CUstream, *CUgraph) CUresult
	cuGraphInstantiate   func(*CUgraphExec, CUgraph, uintptr, uintptr, uint64) CUresult
	cuGraphLaunch        func(CUgraphExec, CUstream) CUresult
	cuGraphDestroy       func(CUgraph) CUresult
	cuGraphExecDestroy   func(CUgraphExec) CUresult

	// Prefetch stream and events
	prefetchStream CUstream
	computeEvent   CUevent
	prefetchEvent  CUevent
	streamsReady   bool
)

// initStreams creates CUDA streams and events for overlapped execution.
func initStreams() error {
	if streamsReady {
		return nil
	}

	if !Init() {
		return fmt.Errorf("CUDA not initialized")
	}

	// Create prefetch stream (non-blocking, can overlap with default stream)
	EnsureContext()
	if r := cuStreamCreate(&prefetchStream, 1); r != CUDA_SUCCESS { // CU_STREAM_NON_BLOCKING = 1
		return fmt.Errorf("create prefetch stream: error %d", r)
	}

	// Create sync events (disable timing for lower overhead)
	if r := cuEventCreate(&computeEvent, 2); r != CUDA_SUCCESS { // CU_EVENT_DISABLE_TIMING = 2
		return fmt.Errorf("create compute event: error %d", r)
	}
	if r := cuEventCreate(&prefetchEvent, 2); r != CUDA_SUCCESS {
		return fmt.Errorf("create prefetch event: error %d", r)
	}

	streamsReady = true
	fmt.Println("[gpu] Streams + events initialized (prefetch overlap)")
	return nil
}

// PrefetchWeights launches a lightweight read kernel on the prefetch stream
// to warm L2 cache for the given GPU weight buffers.
func PrefetchWeights(weights ...*GPUQuantWeight) {
	if !streamsReady || prefetchStream == 0 {
		return
	}
	EnsureContext()

	// Wait for compute to finish current layer before prefetching next.
	if r := cuEventRecord(computeEvent, 0); r != CUDA_SUCCESS { // record on default stream
		return
	}
	if r := cuStreamWaitEvent(prefetchStream, computeEvent, 0); r != CUDA_SUCCESS {
		return
	}

	// Launch prefetch touches on the prefetch stream.
	for _, w := range weights {
		if !validGPUQuantWeight(w) {
			continue
		}
		// Touch weight memory to warm L2: read 1 float per 128B cache line.
		totalBytes := uint64(w.InDim/8) * uint64(w.OutDim) * 4
		n := uint32(totalBytes / 128) // touch every 128 bytes
		if n < 1 {
			n = 1
		}
		_ = LaunchKernelOnStream(fnPrefetch, (n+255)/256, 1, 1, 256, 1, 1, 0, prefetchStream,
			unsafe.Pointer(&w.QWeight.Ptr), unsafe.Pointer(&n))
	}
}

// MarkComputeDone records an event on the default (compute) stream.
func MarkComputeDone() {
	if streamsReady && computeEvent != 0 && cuEventRecord != nil {
		_ = cuEventRecord(computeEvent, 0)
	}
}

// WaitPrefetch makes the default stream wait for prefetch to complete.
func WaitPrefetch() {
	if streamsReady && prefetchStream != 0 && prefetchEvent != 0 {
		if r := cuEventRecord(prefetchEvent, prefetchStream); r != CUDA_SUCCESS {
			return
		}
		_ = cuStreamWaitEvent(0, prefetchEvent, 0) // default stream waits
	}
}

// SyncAll synchronizes both streams.
func SyncAll() {
	EnsureContext()
	if streamsReady && prefetchStream != 0 && cuStreamSynchronize != nil {
		_ = cuStreamSynchronize(prefetchStream)
	}
	cuCtxSynchronize()
}

// --- CUDA Graph support ---

// CapturedGraph holds an instantiated CUDA graph for replay.
type CapturedGraph struct {
	graph CUgraph
	exec  CUgraphExec
}

// BeginCapture starts capturing GPU operations on the default stream.
func BeginCapture() error {
	if !streamsReady {
		if err := initStreams(); err != nil {
			return err
		}
	}
	EnsureContext()
	// CU_STREAM_CAPTURE_MODE_GLOBAL = 0
	if r := cuStreamBeginCapture(0, 0); r != CUDA_SUCCESS {
		return fmt.Errorf("begin capture: error %d", r)
	}
	return nil
}

// EndCapture stops capturing and instantiates the graph.
func EndCapture() (*CapturedGraph, error) {
	EnsureContext()
	var g CUgraph
	if r := cuStreamEndCapture(0, &g); r != CUDA_SUCCESS {
		return nil, fmt.Errorf("end capture: error %d", r)
	}

	var exec CUgraphExec
	if r := cuGraphInstantiate(&exec, g, 0, 0, 0); r != CUDA_SUCCESS {
		cuGraphDestroy(g)
		return nil, fmt.Errorf("graph instantiate: error %d", r)
	}

	return &CapturedGraph{graph: g, exec: exec}, nil
}

// Launch replays the captured graph on the default stream.
func (cg *CapturedGraph) Launch() error {
	if cg == nil || cg.exec == 0 {
		return fmt.Errorf("nil CUDA graph executable")
	}
	EnsureContext()
	if r := cuGraphLaunch(cg.exec, 0); r != CUDA_SUCCESS {
		return fmt.Errorf("graph launch: error %d", r)
	}
	return nil
}

// Destroy frees graph resources.
func (cg *CapturedGraph) Destroy() {
	if cg == nil {
		return
	}
	if cg.exec != 0 {
		cuGraphExecDestroy(cg.exec)
	}
	if cg.graph != 0 {
		cuGraphDestroy(cg.graph)
	}
}

// LaunchKernelOnStream is like LaunchKernel but on a specific stream.
func LaunchKernelOnStream(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem uint32, stream CUstream, args ...unsafe.Pointer) error {
	if fn == 0 {
		return fmt.Errorf("nil CUDA function")
	}
	if gridX == 0 || gridY == 0 || gridZ == 0 || blockX == 0 || blockY == 0 || blockZ == 0 {
		return fmt.Errorf("invalid CUDA launch dimensions grid=(%d,%d,%d) block=(%d,%d,%d)", gridX, gridY, gridZ, blockX, blockY, blockZ)
	}
	for i, arg := range args {
		if arg == nil {
			return fmt.Errorf("nil CUDA kernel argument %d", i)
		}
	}
	EnsureContext()
	var argPtr unsafe.Pointer
	if len(args) > 0 {
		ptrs := make([]unsafe.Pointer, len(args))
		copy(ptrs, args)
		argPtr = unsafe.Pointer(&ptrs[0])
	}
	r := cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, uintptr(stream), argPtr, nil)
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(stream): error %d", r)
	}
	return nil
}

var fnPrefetch CUfunction

func shutdownStreams() {
	if !streamsReady {
		return
	}
	EnsureContext()
	if computeEvent != 0 && cuEventDestroy != nil {
		cuEventDestroy(computeEvent)
		computeEvent = 0
	}
	if prefetchEvent != 0 && cuEventDestroy != nil {
		cuEventDestroy(prefetchEvent)
		prefetchEvent = 0
	}
	if prefetchStream != 0 && cuStreamDestroy != nil {
		cuStreamDestroy(prefetchStream)
		prefetchStream = 0
	}
	streamsReady = false
}
