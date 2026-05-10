package model

import "testing"

func TestFloatKVCheckpointRestore(t *testing.T) {
	k := [][]float32{{1, 2}, {10}}
	v := [][]float32{{3, 4}, {20}}
	cp := CheckpointFloatKV(k, v)

	k[0] = append(k[0], 5, 6)
	v[0] = append(v[0], 7, 8)
	k[1] = append(k[1], 11)
	v[1] = append(v[1], 21)
	cp.Restore(k, v)

	if got, want := len(k[0]), 2; got != want {
		t.Fatalf("k[0] len=%d want %d", got, want)
	}
	if got, want := len(v[0]), 2; got != want {
		t.Fatalf("v[0] len=%d want %d", got, want)
	}
	if got, want := len(k[1]), 1; got != want {
		t.Fatalf("k[1] len=%d want %d", got, want)
	}
	if got, want := len(v[1]), 1; got != want {
		t.Fatalf("v[1] len=%d want %d", got, want)
	}
	if k[0][0] != 1 || k[0][1] != 2 || v[1][0] != 20 {
		t.Fatalf("restore changed existing values: k=%v v=%v", k, v)
	}
}

func TestCompressedKVCheckpointRestoreAcrossCompression(t *testing.T) {
	cfg := DefaultTurboQuantConfig()
	cfg.ResidualWindow = 2
	tq := NewTurboQuantState(2, 1, cfg)
	cache := NewCompressedKVCache(2, 1, 2, tq, false)

	cache.Append([]float32{1, 2}, []float32{10, 20})
	cache.Append([]float32{3, 4}, []float32{30, 40})
	cp := cache.Checkpoint()
	wantK := append([]float32(nil), cache.GetK()...)
	wantV := append([]float32(nil), cache.GetV()...)

	// These appends force compression and drop rows from FullK/FullV, which a
	// length-only rollback could not restore correctly.
	cache.Append([]float32{5, 6}, []float32{50, 60})
	cache.Append([]float32{7, 8}, []float32{70, 80})
	if cache.SeqLen() != 4 || cache.CompressedCount() == 0 {
		t.Fatalf("expected compressed candidate state, seq=%d compressed=%d", cache.SeqLen(), cache.CompressedCount())
	}

	cache.Restore(cp)
	if cache.SeqLen() != 2 {
		t.Fatalf("restored seq len=%d want 2", cache.SeqLen())
	}
	if cache.CompressedCount() != 0 || cache.FullCount() != 2 {
		t.Fatalf("restored counts compressed=%d full=%d, want 0/2", cache.CompressedCount(), cache.FullCount())
	}
	gotK := cache.GetK()
	gotV := cache.GetV()
	if !sameFloat32s(gotK, wantK) || !sameFloat32s(gotV, wantV) {
		t.Fatalf("restore mismatch: gotK=%v wantK=%v gotV=%v wantV=%v", gotK, wantK, gotV, wantV)
	}
}

func TestCompressedKVCheckpointSliceHelpers(t *testing.T) {
	cache := NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{3, 4})
	caches := []*CompressedKVCache{cache, nil}
	cp := CheckpointCompressedKV(caches)
	cache.Append([]float32{5, 6}, []float32{7, 8})
	RestoreCompressedKV(caches, cp)
	if got, want := cache.SeqLen(), 1; got != want {
		t.Fatalf("seq len=%d want %d", got, want)
	}
	if got, want := cache.GetK(), []float32{1, 2}; !sameFloat32s(got, want) {
		t.Fatalf("restored K=%v want %v", got, want)
	}
}
