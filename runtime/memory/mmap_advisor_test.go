package memory

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
