package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

// CPUDecodeState is the incremental state shape needed by a KV-reusing
// speculative verifier. The current generator still owns the full token-step
// implementation, but keeping this state contract explicit lets the verifier
// block land without changing the public GenerateSpeculative API again.
type CPUDecodeState struct {
	Model        *LlamaModel
	Output       []int
	KVCacheK     [][]float32
	KVCacheV     [][]float32
	CompressedKV []*kv.CompressedKVCache
	KVDims       []int
}

// CPUDecodeCheckpoint records a restorable point before staging verifier tokens.
type CPUDecodeCheckpoint struct {
	OutputLen    int
	FloatKV      kv.FloatKVCheckpoint
	CompressedKV []kv.CompressedKVCheckpoint
}

func NewCPUDecodeStateForSpeculative(m *LlamaModel, prepared []int, maxTokens int) (*CPUDecodeState, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens=%d must be >= 0", maxTokens)
	}
	dims, err := m.LayerKVDims()
	if err != nil {
		return nil, err
	}
	st := &CPUDecodeState{
		Model:    m,
		Output:   append([]int(nil), prepared...),
		KVCacheK: make([][]float32, len(m.Layers)),
		KVCacheV: make([][]float32, len(m.Layers)),
		KVDims:   dims,
	}
	maxSeq := len(prepared) + maxTokens
	if maxSeq < 1 {
		maxSeq = 1
	}
	for i, dim := range dims {
		if dim <= 0 {
			continue
		}
		if maxSeq > int(^uint(0)>>1)/dim {
			return nil, fmt.Errorf("layer %d KV capacity overflow: seq=%d dim=%d", i, maxSeq, dim)
		}
		st.KVCacheK[i] = make([]float32, 0, maxSeq*dim)
		st.KVCacheV[i] = make([]float32, 0, maxSeq*dim)
	}
	return st, nil
}

func (s *CPUDecodeState) DecodeOneGreedy() (int, error) {
	if s == nil || s.Model == nil {
		return 0, fmt.Errorf("nil decode state/model")
	}
	verified := s.Model.generatePrepared(s.Output, 1)
	if len(verified) <= len(s.Output) {
		return 0, fmt.Errorf("decode produced no token")
	}
	tok := verified[len(s.Output)]
	s.Output = append(s.Output, tok)
	return tok, nil
}

func (s *CPUDecodeState) Checkpoint() CPUDecodeCheckpoint {
	cp := CPUDecodeCheckpoint{OutputLen: len(s.Output)}
	if s.CompressedKV != nil {
		cp.CompressedKV = kv.CheckpointCompressedKV(s.CompressedKV)
	} else {
		cp.FloatKV = kv.CheckpointFloatKV(s.KVCacheK, s.KVCacheV)
	}
	return cp
}

func (s *CPUDecodeState) Restore(cp CPUDecodeCheckpoint) error {
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	s.Output = s.Output[:cp.OutputLen]
	if s.CompressedKV != nil {
		return kv.RestoreCompressedKV(s.CompressedKV, cp.CompressedKV)
	}
	return cp.FloatKV.Restore(s.KVCacheK, s.KVCacheV)
}

func (s *CPUDecodeState) CommitAcceptedOutputOnly(cp CPUDecodeCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	s.Output = s.Output[:cp.OutputLen]
	s.Output = append(s.Output, acceptance.OutputTokens...)
	return nil
}

// VerifyGreedyBlock verifies drafted tokens with the real model and returns the
// greedy acceptance result. This first implementation intentionally uses the
// existing prepared-prompt CPU generator as the verifier backend; replacing the
// body with a KV-reusing DecodeOne loop should not require changes in
// GenerateSpeculative.
func (s *CPUDecodeState) VerifyGreedyBlock(drafted []int) (MTPAcceptance, error) {
	if s == nil || s.Model == nil {
		return MTPAcceptance{}, fmt.Errorf("nil decode state/model")
	}
	verifyN := len(drafted) + 1
	shadow := *s
	shadow.Output = append([]int(nil), s.Output...)
	verifierTokens := make([]int, 0, verifyN)
	for i := 0; i < verifyN; i++ {
		tok, err := shadow.DecodeOneGreedy()
		if err != nil {
			return MTPAcceptance{}, fmt.Errorf("verifier token %d: %w", i, err)
		}
		verifierTokens = append(verifierTokens, tok)
	}
	return AcceptMTPDraft(drafted, verifierTokens)
}

func (s *CPUDecodeState) CommitAccepted(cp CPUDecodeCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	if cp.OutputLen < 0 || cp.OutputLen > len(s.Output) {
		return fmt.Errorf("checkpoint output len=%d outside current len=%d", cp.OutputLen, len(s.Output))
	}
	keep := acceptance.KVKeepTokens()
	if s.CompressedKV != nil {
		if err := kv.KeepCompressedKVAppended(s.CompressedKV, cp.CompressedKV, keep); err != nil {
			return err
		}
	} else {
		if err := CommitAcceptedFloatKV(s.KVCacheK, s.KVCacheV, cp.FloatKV, s.KVDims, acceptance); err != nil {
			return err
		}
	}
	s.Output = s.Output[:cp.OutputLen]
	s.Output = append(s.Output, acceptance.OutputTokens...)
	return nil
}
