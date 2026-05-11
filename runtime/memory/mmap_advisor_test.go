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
	if err := a.Prefetch(0, int64(pageSize*4)); err != nil {
		t.Fatalf("Prefetch: %v", err)
	}
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
	if err := a.Evict(0, int64(pageSize*4)); err != nil {
		t.Fatalf("Evict: %v", err)
	}
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
	if err := a.Prefetch(0, int64(pageSize*2)); err != nil {
		t.Fatalf("Prefetch 0: %v", err)
	}
	if err := a.Prefetch(int64(pageSize*2), int64(pageSize*2)); err != nil {
		t.Fatalf("Prefetch 1: %v", err)
	}
	if err := a.Prefetch(int64(pageSize*4), int64(pageSize*2)); err != nil {
		t.Fatalf("Prefetch 2: %v", err)
	}

	// Also add a non-adjacent range
	if err := a.Prefetch(int64(pageSize*10), int64(pageSize*2)); err != nil {
		t.Fatalf("Prefetch 3: %v", err)
	}

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

func TestMmapAdvisorAccountingIsIdempotent(t *testing.T) {
	pageSize := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, pageSize*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(data)

	a := NewMmapAdvisor(data)
	if err := a.Prefetch(0, int64(pageSize)); err != nil {
		t.Fatalf("Prefetch: %v", err)
	}
	if err := a.Prefetch(0, int64(pageSize)); err != nil {
		t.Fatalf("Prefetch repeat: %v", err)
	}
	_, hot, peak := a.Stats()
	if hot != int64(pageSize) || peak != int64(pageSize) {
		t.Fatalf("repeated prefetch hot/peak=%d/%d, want %d/%d", hot, peak, pageSize, pageSize)
	}

	if err := a.Evict(0, int64(pageSize)); err != nil {
		t.Fatalf("Evict: %v", err)
	}
	if err := a.Evict(0, int64(pageSize)); err != nil {
		t.Fatalf("Evict repeat: %v", err)
	}
	_, hot, peak = a.Stats()
	if hot != 0 || peak != int64(pageSize) {
		t.Fatalf("repeated evict hot/peak=%d/%d, want 0/%d", hot, peak, pageSize)
	}
}

func TestMmapAdvisorInvalidRangesAreIgnored(t *testing.T) {
	pageSize := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, pageSize*2, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(data)

	a := NewMmapAdvisor(data)
	for _, tc := range []struct{ off, bytes int64 }{
		{off: -1, bytes: int64(pageSize)},
		{off: int64(len(data)), bytes: int64(pageSize)},
		{off: 0, bytes: 0},
	} {
		if err := a.Prefetch(tc.off, tc.bytes); err != nil {
			t.Fatalf("Prefetch(%d,%d): %v", tc.off, tc.bytes, err)
		}
		a.Touch(tc.off, tc.bytes)
		if err := a.Evict(tc.off, tc.bytes); err != nil {
			t.Fatalf("Evict(%d,%d): %v", tc.off, tc.bytes, err)
		}
	}
	if n, hot, peak := a.Stats(); n != 0 || hot != 0 || peak != 0 {
		t.Fatalf("invalid ranges changed stats: ranges=%d hot=%d peak=%d", n, hot, peak)
	}
}

func TestMmapAdvisorMergeDoesNotMakeColdRangesHot(t *testing.T) {
	pageSize := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, pageSize*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer syscall.Munmap(data)

	a := NewMmapAdvisor(data)
	if err := a.Prefetch(0, int64(pageSize)); err != nil {
		t.Fatalf("Prefetch hot: %v", err)
	}
	if err := a.Prefetch(int64(pageSize), int64(pageSize)); err != nil {
		t.Fatalf("Prefetch cold candidate: %v", err)
	}
	if err := a.Evict(int64(pageSize), int64(pageSize)); err != nil {
		t.Fatalf("Evict: %v", err)
	}
	a.MergeRanges()

	n, hot, peak := a.Stats()
	if n != 2 {
		t.Fatalf("merged cold+hot into %d ranges, want 2", n)
	}
	if hot != int64(pageSize) || peak != int64(pageSize*2) {
		t.Fatalf("hot/peak=%d/%d, want %d/%d", hot, peak, pageSize, pageSize*2)
	}
}
