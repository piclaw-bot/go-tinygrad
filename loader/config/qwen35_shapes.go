package config

import "fmt"

type Qwen35FullAttentionShapes struct {
	QProj    []int `json:"q_proj"`
	KProj    []int `json:"k_proj"`
	VProj    []int `json:"v_proj"`
	OProj    []int `json:"o_proj"`
	QNorm    []int `json:"q_norm"`
	KNorm    []int `json:"k_norm"`
	GateSize int   `json:"gate_size"`
}

func Qwen35FullAttentionShapesFor(hidden, numHeads, numKVHeads, headDim int) (Qwen35FullAttentionShapes, error) {
	if hidden <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 {
		return Qwen35FullAttentionShapes{}, fmt.Errorf("invalid Qwen3.5 attention dims hidden=%d heads=%d kv_heads=%d head_dim=%d", hidden, numHeads, numKVHeads, headDim)
	}
	qOut, ok := checkedMul(numHeads, headDim)
	if !ok {
		return Qwen35FullAttentionShapes{}, fmt.Errorf("Q projection dimension overflow")
	}
	qGateOut, ok := checkedMul(qOut, 2)
	if !ok {
		return Qwen35FullAttentionShapes{}, fmt.Errorf("Q+gate projection dimension overflow")
	}
	kvOut, ok := checkedMul(numKVHeads, headDim)
	if !ok {
		return Qwen35FullAttentionShapes{}, fmt.Errorf("KV projection dimension overflow")
	}
	return Qwen35FullAttentionShapes{
		QProj:    []int{qGateOut, hidden},
		KProj:    []int{kvOut, hidden},
		VProj:    []int{kvOut, hidden},
		OProj:    []int{hidden, qOut},
		QNorm:    []int{headDim},
		KNorm:    []int{headDim},
		GateSize: qOut,
	}, nil
}

func checkedMul(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a != 0 && b > maxInt/a {
		return 0, false
	}
	return a * b, true
}
