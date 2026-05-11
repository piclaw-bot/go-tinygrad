package model

// Workspace holds pre-allocated buffers for zero-alloc inference.
// Sized once at model load, reused across all layers.
type Workspace struct {
	seqLen  int
	hidden  int
	inter   int
	heads   int
	headDim int

	// Ping-pong hidden state buffers
	buf0 []float32 // [seqLen, hidden]
	buf1 []float32 // [seqLen, hidden]

	// Intermediate buffers
	qkvBuf     []float32 // [seqLen, hidden*3]
	attnOut    []float32 // [seqLen, hidden]
	ffnBuf     []float32 // [seqLen, intermediate]
	scores     []float32 // [heads, seqLen, seqLen]
	tempHidden []float32 // [seqLen, hidden]
	outEmb     []float32 // [hidden] for output embedding
}

func newWorkspace(seqLen int, cfg BertConfig) *Workspace {
	h := cfg.HiddenSize
	inter := cfg.Intermediate
	heads := cfg.NumHeads
	headDim := h / heads
	return &Workspace{
		seqLen:     seqLen,
		hidden:     h,
		inter:      inter,
		heads:      heads,
		headDim:    headDim,
		buf0:       make([]float32, seqLen*h),
		buf1:       make([]float32, seqLen*h),
		qkvBuf:     make([]float32, seqLen*h*3),
		attnOut:    make([]float32, seqLen*h),
		ffnBuf:     make([]float32, seqLen*inter),
		scores:     make([]float32, heads*seqLen*seqLen),
		tempHidden: make([]float32, seqLen*h),
		outEmb:     make([]float32, h),
	}
}
