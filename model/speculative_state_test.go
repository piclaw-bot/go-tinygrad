package model

import "testing"

func TestCPUDecodeStateVerifierBackendFallback(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{}, Layers: []LlamaLayer{}}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1}, 1, "kv")
	if err != nil {
		t.Fatal(err)
	}
	if got := st.VerifierBackend(); got != "replay" {
		t.Fatalf("VerifierBackend=%q want replay fallback", got)
	}
}

func TestCPUDecodeStateCommitAcceptedFloatKV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 1, HeadDim: 2},
		Layers: []LlamaLayer{{HasKV: true}, {HasKV: false}},
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{10, 11}, 4)
	if err != nil {
		t.Fatalf("NewCPUDecodeStateForSpeculative: %v", err)
	}
	cp := st.Checkpoint()
	st.KVCacheK[0] = append(st.KVCacheK[0], 1, 2, 3, 4, 5, 6)
	st.KVCacheV[0] = append(st.KVCacheV[0], 10, 20, 30, 40, 50, 60)
	st.Output = append(st.Output, 101, 102, 103)
	acc, err := AcceptMTPDraft([]int{101, 999}, []int{101, 102, 777})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if err := st.CommitAccepted(cp, acc); err != nil {
		t.Fatalf("CommitAccepted: %v", err)
	}
	if !sameInts(st.Output, []int{10, 11, 101, 102}) {
		t.Fatalf("output=%v", st.Output)
	}
	if len(st.KVCacheK[0]) != 4 || len(st.KVCacheV[0]) != 4 {
		t.Fatalf("KV lens K/V=%d/%d want 4/4", len(st.KVCacheK[0]), len(st.KVCacheV[0]))
	}
}

func TestCPUDecodeStateGenerateGreedyNilState(t *testing.T) {
	var st *CPUDecodeState
	if err := st.GenerateGreedy(1); err == nil {
		t.Fatal("nil state GenerateGreedy returned nil error")
	}
}

func TestCPUDecodeStateDecodeOneGreedyNilState(t *testing.T) {
	var st *CPUDecodeState
	if _, err := st.DecodeOneGreedy(); err == nil {
		t.Fatal("nil state DecodeOneGreedy returned nil error")
	}
}

func TestCPUDecodeStateVerifyGreedyBlockNilState(t *testing.T) {
	var st *CPUDecodeState
	if _, err := st.VerifyGreedyBlock([]int{1}); err == nil {
		t.Fatal("nil state VerifyGreedyBlock returned nil error")
	}
}

func TestCPUDecodeStateCommitAcceptedOutputOnly(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 1, HeadDim: 2},
		Layers: []LlamaLayer{{HasKV: true}},
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1, 2}, 4)
	if err != nil {
		t.Fatal(err)
	}
	cp := st.Checkpoint()
	acc, err := AcceptMTPDraft([]int{3, 4}, []int{3, 9, 10})
	if err != nil {
		t.Fatal(err)
	}
	if err := st.CommitAcceptedOutputOnly(cp, acc); err != nil {
		t.Fatalf("CommitAcceptedOutputOnly: %v", err)
	}
	if !sameInts(st.Output, []int{1, 2, 3, 9}) {
		t.Fatalf("output=%v", st.Output)
	}
	if len(st.KVCacheK[0]) != 0 || len(st.KVCacheV[0]) != 0 {
		t.Fatalf("output-only commit changed KV")
	}
}

func TestCPUDecodeStateRestoreFloatKV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 1, HeadDim: 1},
		Layers: []LlamaLayer{{HasKV: true}},
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	cp := st.Checkpoint()
	st.Output = append(st.Output, 2)
	st.KVCacheK[0] = append(st.KVCacheK[0], 42)
	st.KVCacheV[0] = append(st.KVCacheV[0], 43)
	if err := st.Restore(cp); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if !sameInts(st.Output, []int{1}) || len(st.KVCacheK[0]) != 0 || len(st.KVCacheV[0]) != 0 {
		t.Fatalf("restored output=%v K=%v V=%v", st.Output, st.KVCacheK[0], st.KVCacheV[0])
	}
}
