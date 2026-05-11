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
	if err := cp.Restore(k, v); err != nil {
		t.Fatalf("Restore: %v", err)
	}

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

func TestFloatKVCheckpointRestoreReportsInvalidCheckpoint(t *testing.T) {
	k := [][]float32{{1}}
	v := [][]float32{{2}}
	cp := FloatKVCheckpoint{KLen: []int{2}, VLen: []int{1}}
	if err := cp.Restore(k, v); err == nil {
		t.Fatal("Restore accepted checkpoint longer than current cache")
	}
}

func TestLayerKVDimsAndModelCommitAcceptedFloatKV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 2, HeadDim: 4},
		Layers: []LlamaLayer{
			{HasKV: true},
			{HasKV: true, HeadDimLocal: 8},
			{HasKV: false, KVSourceLayer: 0},
		},
	}
	dims, err := m.LayerKVDims()
	if err != nil {
		t.Fatalf("LayerKVDims: %v", err)
	}
	wantDims := []int{8, 16, 0}
	for i := range wantDims {
		if dims[i] != wantDims[i] {
			t.Fatalf("dims=%v want %v", dims, wantDims)
		}
	}

	k := [][]float32{make([]float32, 8+3*8), make([]float32, 3*16), nil}
	v := [][]float32{make([]float32, 8+3*8), make([]float32, 3*16), nil}
	for i := range k[0] {
		k[0][i] = float32(i)
		v[0][i] = float32(i + 100)
	}
	for i := range k[1] {
		k[1][i] = float32(i + 200)
		v[1][i] = float32(i + 300)
	}
	cp := FloatKVCheckpoint{KLen: []int{8, 0, 0}, VLen: []int{8, 0, 0}}
	acceptance := MTPAcceptance{AcceptedPrefixLen: 1} // keep two verifier positions
	if err := m.CommitAcceptedFloatKV(k, v, cp, acceptance); err != nil {
		t.Fatalf("CommitAcceptedFloatKV: %v", err)
	}
	if got, want := len(k[0]), 8+2*8; got != want {
		t.Fatalf("len(k[0])=%d want %d", got, want)
	}
	if got, want := len(k[1]), 2*16; got != want {
		t.Fatalf("len(k[1])=%d want %d", got, want)
	}
	if got, want := len(k[2]), 0; got != want {
		t.Fatalf("len(k[2])=%d want %d", got, want)
	}
}

func TestLayerKVDimValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: 0, HeadDim: 4}, Layers: []LlamaLayer{{HasKV: true}}}
	if _, err := m.LayerKVDim(0); err == nil {
		t.Fatal("LayerKVDim accepted num_key_value_heads=0")
	}
	m.Config.NumKVHeads = 1
	if _, err := m.LayerKVDim(1); err == nil {
		t.Fatal("LayerKVDim accepted out-of-range layer")
	}
}

func TestCompressedKVCheckpointRestoreReportsInvalidCheckpoint(t *testing.T) {
	cache := NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{3, 4})
	cp := cache.Checkpoint()
	cp.compressedKLen = 1
	if err := cache.Restore(cp); err == nil {
		t.Fatal("Restore accepted checkpoint with impossible compressed length")
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

	if err := cache.Restore(cp); err != nil {
		t.Fatalf("Restore: %v", err)
	}
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
	// KeepAppended may replay kept staged rows through Append, so tiny quantization
	// differences are acceptable if the residual window recompresses rows again.
	if !closeFloat32s(gotK, wantK, 0.1) {
		t.Fatalf("kept K=%v want %v", gotK, wantK)
	}
	if !closeFloat32s(gotV, wantV, 0.1) {
		t.Fatalf("kept V=%v want %v", gotV, wantV)
	}
}

func TestCompressedKVCheckpointSliceHelpers(t *testing.T) {
	cache := NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{3, 4})
	caches := []*CompressedKVCache{cache, nil}
	cp := CheckpointCompressedKV(caches)
	cache.Append([]float32{5, 6}, []float32{7, 8})
	if err := RestoreCompressedKV(caches, cp); err != nil {
		t.Fatalf("RestoreCompressedKV: %v", err)
	}
	if got, want := cache.SeqLen(), 1; got != want {
		t.Fatalf("seq len=%d want %d", got, want)
	}
	if got, want := cache.GetK(), []float32{1, 2}; !sameFloat32s(got, want) {
		t.Fatalf("restored K=%v want %v", got, want)
	}
}
