package model

// BF16 native forward path for models trained in BF16 (Gemma3/4).
//
// Instead of:  []float32 + toBF16() truncation at each step
// This uses:   []uint16 (native BF16) throughout, with F32 accumulation in GEMV
//
// Benefits:
//   - Half the memory bandwidth for hidden states
//   - No wasted conversion cycles
//   - Matches training precision exactly
//   - Better for ARM SBCs with limited memory bandwidth

import (
	"github.com/rcarmo/go-pherence/backends/simd"
)

// BF16Hidden wraps a BF16 hidden state buffer.
type BF16Hidden struct {
	data []uint16
	n    int
}

// NewBF16Hidden allocates a BF16 hidden state of size n.
func NewBF16Hidden(n int) *BF16Hidden {
	return &BF16Hidden{data: make([]uint16, n), n: n}
}

// Data returns the underlying BF16 data.
func (h *BF16Hidden) Data() []uint16 { return h.data }

// Len returns the number of elements.
func (h *BF16Hidden) Len() int { return h.n }

// ToF32 converts to float32 slice (for interop with F32 code paths).
func (h *BF16Hidden) ToF32() []float32 {
	return simd.BF16ToF32Slice(h.data)
}

// FromF32 converts from float32 slice into this BF16 buffer.
func (h *BF16Hidden) FromF32(f32 []float32) {
	for i, v := range f32 {
		h.data[i] = simd.F32ToBF16(v)
	}
}

// Copy copies src into this buffer.
func (h *BF16Hidden) Copy(src *BF16Hidden) {
	copy(h.data, src.data)
}

// RMSNorm applies RMSNorm in-place using SIMD assembly.
func (h *BF16Hidden) RMSNorm(w []uint16, eps float32) {
	simd.BF16RMSNormAsm(h.data, w, eps)
}

// Add computes h = a + b element-wise using SIMD assembly.
func (h *BF16Hidden) Add(a, b *BF16Hidden) {
	simd.BF16VecAddAsm(h.data, a.data, b.data)
}

// Dot computes dot product with another BF16 vector using SIMD assembly.
func (h *BF16Hidden) Dot(other *BF16Hidden) float32 {
	return simd.BF16DotAsm(h.data, other.data)
}

// Scale multiplies all elements by a scalar.
func (h *BF16Hidden) Scale(s float32) {
	for i := range h.data {
		f := simd.BF16ToF32(h.data[i]) * s
		h.data[i] = simd.F32ToBF16(f)
	}
}

// GemvNT computes out = x @ W^T where x is BF16 and W is F32.
// Mixed precision: BF16 activations × F32 weights → BF16 output.
func (h *BF16Hidden) GemvNT(x *BF16Hidden, w []float32, inDim, outDim int) {
	simd.BF16GemvNT(h.data, x.data, w, inDim, outDim)
}

// UseBF16 returns true if the model should use native BF16 forward path.
func UseBF16(cfg LlamaConfig) bool {
	return cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text"
}
