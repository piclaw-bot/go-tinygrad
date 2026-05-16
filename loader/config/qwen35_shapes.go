package config

import "fmt"

type Qwen35LinearAttentionShapes struct {
	QKV      []int `json:"qkv"`
	Gate     []int `json:"gate"`
	Conv1D   []int `json:"conv1d"`
	DTBias   []int `json:"dt_bias"`
	A        []int `json:"a"`
	Beta     []int `json:"beta"`
	Alpha    []int `json:"alpha"`
	Norm     []int `json:"norm"`
	Out      []int `json:"out"`
	KeyDim   int   `json:"key_dim"`
	ValueDim int   `json:"value_dim"`
	ConvDim  int   `json:"conv_dim"`
	HeadVDim int   `json:"head_v_dim"`
}

type Qwen35FullAttentionShapes struct {
	QProj    []int `json:"q_proj"`
	KProj    []int `json:"k_proj"`
	VProj    []int `json:"v_proj"`
	OProj    []int `json:"o_proj"`
	QNorm    []int `json:"q_norm"`
	KNorm    []int `json:"k_norm"`
	GateSize int   `json:"gate_size"`
}

func Qwen35LinearAttentionShapesFor(hidden, ssmInner, ssmState, ssmConvKernel, ssmDtRank, ssmGroupCount int) (Qwen35LinearAttentionShapes, error) {
	if hidden <= 0 || ssmInner <= 0 || ssmState <= 0 || ssmConvKernel <= 0 || ssmDtRank <= 0 || ssmGroupCount <= 0 {
		return Qwen35LinearAttentionShapes{}, fmt.Errorf("invalid Qwen3.5 linear-attention dims hidden=%d inner=%d state=%d conv=%d dt_rank=%d groups=%d", hidden, ssmInner, ssmState, ssmConvKernel, ssmDtRank, ssmGroupCount)
	}
	keyDim, ok := checkedMul(ssmState, ssmGroupCount)
	if !ok {
		return Qwen35LinearAttentionShapes{}, fmt.Errorf("linear-attention key dimension overflow")
	}
	headVDim := ssmInner / ssmDtRank
	if headVDim <= 0 || headVDim*ssmDtRank != ssmInner {
		return Qwen35LinearAttentionShapes{}, fmt.Errorf("linear-attention inner=%d not divisible by dt_rank=%d", ssmInner, ssmDtRank)
	}
	valueDim := ssmInner
	convDim := keyDim*2 + valueDim
	if convDim < keyDim || convDim < valueDim {
		return Qwen35LinearAttentionShapes{}, fmt.Errorf("linear-attention conv dimension overflow")
	}
	return Qwen35LinearAttentionShapes{
		QKV:      []int{hidden, convDim},
		Gate:     []int{hidden, valueDim},
		Conv1D:   []int{ssmConvKernel, convDim},
		DTBias:   []int{ssmDtRank},
		A:        []int{ssmDtRank},
		Beta:     []int{hidden, ssmDtRank},
		Alpha:    []int{hidden, ssmDtRank},
		Norm:     []int{headVDim},
		Out:      []int{valueDim, hidden},
		KeyDim:   keyDim,
		ValueDim: valueDim,
		ConvDim:  convDim,
		HeadVDim: headVDim,
	}, nil
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
