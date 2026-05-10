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

func TestFloatKVCheckpointKeepAppended(t *testing.T) {
	k := [][]float32{{1, 2, 10, 11, 12, 13, 14, 15}, {100, 101, 102, 103}}
	v := [][]float32{{3, 4, 20, 21, 22, 23, 24, 25}, {200, 201, 202, 203}}
	cp := FloatKVCheckpoint{KLen: []int{2, 0}, VLen: []int{2, 0}}
	if err := cp.KeepAppended(k, v, []int{2, 1}, 2); err != nil {
		t.Fatalf("KeepAppended: %v", err)
	}
	if got, want := k[0], []float32{1, 2, 10, 11, 12, 13}; !sameFloat32s(got, want) {
		t.Fatalf("k[0]=%v want %v", got, want)
	}
	if got, want := v[0], []float32{3, 4, 20, 21, 22, 23}; !sameFloat32s(got, want) {
		t.Fatalf("v[0]=%v want %v", got, want)
	}
	if got, want := k[1], []float32{100, 101}; !sameFloat32s(got, want) {
		t.Fatalf("k[1]=%v want %v", got, want)
	}
	if got, want := v[1], []float32{200, 201}; !sameFloat32s(got, want) {
		t.Fatalf("v[1]=%v want %v", got, want)
	}
	if err := cp.KeepAppended(k, v, []int{2, 1}, 99); err == nil {
		t.Fatal("KeepAppended accepted too many tokens")
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

func TestCompressedKVCheckpointKeepAppendedAcrossCompression(t *testing.T) {
	cfg := DefaultTurboQuantConfig()
	cfg.ResidualWindow = 2
	tq := NewTurboQuantState(2, 1, cfg)
	cache := NewCompressedKVCache(2, 1, 2, tq, false)
	cache.Append([]float32{1, 2}, []float32{10, 20})
	cache.Append([]float32{3, 4}, []float32{30, 40})
	cp := cache.Checkpoint()

	cache.Append([]float32{5, 6}, []float32{50, 60})
	cache.Append([]float32{7, 8}, []float32{70, 80})
	cache.Append([]float32{9, 10}, []float32{90, 100})
	if cache.SeqLen() != 5 || cache.CompressedCount() == 0 {
		t.Fatalf("expected staged compressed state, seq=%d compressed=%d", cache.SeqLen(), cache.CompressedCount())
	}
	if err := cache.KeepAppended(cp, 2); err != nil {
		t.Fatalf("KeepAppended: %v", err)
	}
	if cache.SeqLen() != 4 {
		t.Fatalf("seq len=%d want 4", cache.SeqLen())
	}
	gotK := cache.GetK()
	gotV := cache.GetV()
	wantK := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	wantV := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	// The first two positions may have been re-compressed after replay, but the
	// kept staged residual rows must remain exact and the rejected row must vanish.
	if got, want := gotK[4:], wantK[4:]; !closeFloat32s(got, want, 0.1) {
		t.Fatalf("kept K suffix=%v want %v (full K=%v)", got, want, gotK)
	}
	if got, want := gotV[4:], wantV[4:]; !closeFloat32s(got, want, 0.1) {
		t.Fatalf("kept V suffix=%v want %v (full V=%v)", got, want, gotV)
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
