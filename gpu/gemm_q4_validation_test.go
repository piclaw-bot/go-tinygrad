package gpu

import "testing"

func TestGemmQ4RejectsU32OverflowDims(t *testing.T) {
	maxU32 := int(^uint32(0))
	w := &GPUQuantWeight{
		QWeight: &Buffer{Ptr: 1, Size: 4},
		Scales:  &Buffer{Ptr: 1, Size: 4},
		GIdx:    &Buffer{Ptr: 1, Size: 4},
		InDim:   8,
		OutDim:  1,
		Groups:  1,
	}
	in := NewDevBuf(8)
	out := NewDevBuf(1)
	GemmQ4(out, in, w, maxU32+1)
	if out.OnGPU() {
		t.Fatal("GemmQ4 accepted batch dimension exceeding CUDA u32 interface")
	}
}
