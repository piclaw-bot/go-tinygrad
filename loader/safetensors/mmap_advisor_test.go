package safetensors

import (
	"syscall"
	"testing"
)

func TestMmapAdvisorBasic(t *testing.T) {
	// Create a small anonymous mmap for testing
	pageSize := syscall.Getpagesize()
	size := pageSize * 16
	data, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(data)

	a := NewMmapAdvisor(data)

	// Prefetch a range
	a.Prefetch(0, int64(pageSize*4))
	nRanges, hotBytes, _ := a.Stats()
	if nRanges != 1 {
		t.Fatalf("expected 1 range, got %d", nRanges)
	}
	if hotBytes != int64(pageSize*4) {
		t.Fatalf("expected %d hot bytes, got %d", pageSize*4, hotBytes)
	}
	if a.TotalPrefetches.Load() != 1 {
		t.Fatalf("expected 1 prefetch, got %d", a.TotalPrefetches.Load())
	}

	// Touch another range
	a.Touch(int64(pageSize*8), int64(pageSize*2))
	nRanges, _, _ = a.Stats()
	if nRanges != 2 {
		t.Fatalf("expected 2 ranges, got %d", nRanges)
	}

	// Evict first range
	a.Evict(0, int64(pageSize*4))
	if a.TotalEvictions.Load() != 1 {
		t.Fatalf("expected 1 eviction, got %d", a.TotalEvictions.Load())
	}

	// Check range stats
	ranges := a.RangeStats()
	for _, r := range ranges {
		t.Logf("range: offset=%d bytes=%d state=%d hits=%d evicts=%d",
			r.Offset, r.Bytes, r.State, r.Hits, r.Evicts)
	}
}

func TestMmapAdvisorMerge(t *testing.T) {
	pageSize := syscall.Getpagesize()
	size := pageSize * 32
	data, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(data)

	a := NewMmapAdvisor(data)

	// Add adjacent ranges
	a.Prefetch(0, int64(pageSize*2))
	a.Prefetch(int64(pageSize*2), int64(pageSize*2))
	a.Prefetch(int64(pageSize*4), int64(pageSize*2))

	// Also add a non-adjacent range
	a.Prefetch(int64(pageSize*10), int64(pageSize*2))

	nBefore, _, _ := a.Stats()
	t.Logf("before merge: %d ranges", nBefore)

	a.MergeRanges()

	nAfter, _, _ := a.Stats()
	t.Logf("after merge: %d ranges", nAfter)

	if nAfter != 2 {
		t.Fatalf("expected 2 merged ranges, got %d", nAfter)
	}

	ranges := a.RangeStats()
	for _, r := range ranges {
		t.Logf("merged range: offset=%d bytes=%d hits=%d", r.Offset, r.Bytes, r.Hits)
	}

	// First merged range should span 6 pages
	if ranges[0].Bytes != int64(pageSize*6) {
		t.Fatalf("expected first range %d bytes, got %d", pageSize*6, ranges[0].Bytes)
	}
}

func TestMmapAdvisorWithFile(t *testing.T) {
	// Use an actual safetensors file if available
	path := "../../models/smollm2-135m/model.safetensors"
	f, err := Open(path)
	if err != nil {
		path = "../models/smollm2-135m/model.safetensors"
		f, err = Open(path)
		if err != nil {
			t.Skipf("model not found: %v", err)
		}
	}
	defer f.Close()

	if f.mmapData == nil {
		t.Skip("file not mmap'd")
	}

	a := NewMmapAdvisor(f.mmapData)

	// Simulate layer-by-layer access pattern
	for name, info := range f.Tensors {
		off := int64(info.DataOffsets[0]) + int64(8+f.headerSize) // absolute offset in mmap
		sz := int64(info.DataOffsets[1] - info.DataOffsets[0])
		a.Prefetch(off, sz)
		_ = name
	}

	nRanges, hotBytes, peakBytes := a.Stats()
	t.Logf("after prefetching all tensors: %d ranges, %.2f MB hot, %.2f MB peak",
		nRanges, float64(hotBytes)/(1024*1024), float64(peakBytes)/(1024*1024))

	a.MergeRanges()
	nMerged, _, _ := a.Stats()
	t.Logf("after merge: %d ranges (was %d)", nMerged, nRanges)
}

func TestEagerLoadWithFile(t *testing.T) {
	path := "../../models/smollm2-135m/model.safetensors"
	f, err := Open(path)
	if err != nil {
		path = "../models/smollm2-135m/model.safetensors"
		f, err = Open(path)
		if err != nil {
			t.Skipf("model not found: %v", err)
		}
	}
	defer f.Close()

	bytes, err := f.EagerLoad()
	if err != nil {
		t.Fatalf("EagerLoad: %v", err)
	}
	if bytes != int64(len(f.mmapData)) {
		t.Fatalf("EagerLoad bytes=%d want %d", bytes, len(f.mmapData))
	}
	nRanges, hotBytes, peakBytes := f.Advisor.Stats()
	t.Logf("eager loaded %.2f MB: ranges=%d hot=%.2f MB peak=%.2f MB",
		float64(bytes)/(1024*1024), nRanges, float64(hotBytes)/(1024*1024), float64(peakBytes)/(1024*1024))
	if nRanges == 0 || hotBytes == 0 || peakBytes == 0 {
		t.Fatal("expected eager load to update advisor stats")
	}
}
