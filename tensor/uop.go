package tensor

import (
	"fmt"
	"sync"
)

// UOp is the universal IR node. Every computation is a UOp in a DAG.
// UOps are interned (hash-consed): identical subgraphs share a single node.
type UOp struct {
	Op    Ops
	DType DType
	Src   []*UOp
	Arg   any

	// Cached hash for interning
	hash uint64
	// Realized data buffer (set after Realize)
	buf *Buffer
}

// Buffer holds realized tensor data.
type Buffer struct {
	Data   []byte
	DType  DType
	Length int // number of elements
}

// Float32Data returns the buffer as a float32 slice (unsafe, no copy).
func (b *Buffer) Float32Data() []float32 {
	if b.DType != Float32 {
		panic("Float32Data on non-float32 buffer")
	}
	return byteSliceToFloat32(b.Data)
}

// --- UOp constructors ---

// newUOp creates a UOp. Skips interning for common ops to avoid equal() overhead.
func newUOp(op Ops, dtype DType, src []*UOp, arg any) *UOp {
	u := &UOp{Op: op, DType: dtype, Src: src, Arg: arg}
	// Skip interning for ops that are always unique in practice
	// (different source pointers from different layer iterations)
	if op.IsUnary() || op.IsBinary() || op.IsReduce() || op == OpReshape || op == OpPermute {
		return u
	}
	u.hash = u.computeHash()
	return internUOp(u)
}

// ConstOp creates a constant scalar UOp.
func ConstOp(dtype DType, val float64) *UOp {
	return newUOp(OpConst, dtype, nil, val)
}

// BufferOp creates a buffer allocation UOp. Not interned (each buffer is unique).
func BufferOp(dtype DType, shape []int) *UOp {
	return &UOp{Op: OpBuffer, DType: dtype, Arg: cloneShape(shape)}
}

// --- UOp interning ---

var (
	uopCache sync.Map // map[uint64][]*UOp
)

func internUOp(u *UOp) *UOp {
	// Fast path: check if identical UOp already exists
	if existing, ok := uopCache.Load(u.hash); ok {
		for _, e := range existing.([]*UOp) {
			if u.equal(e) {
				return e
			}
		}
		// Hash collision: append
		uopCache.Store(u.hash, append(existing.([]*UOp), u))
		return u
	}
	uopCache.Store(u.hash, []*UOp{u})
	return u
}

func (u *UOp) equal(other *UOp) bool {
	if u.Op != other.Op || u.DType != other.DType || len(u.Src) != len(other.Src) {
		return false
	}
	for i, s := range u.Src {
		if s != other.Src[i] {
			return false
		}
	}
	// Fast Arg comparison: nil == nil, otherwise compare by type
	if u.Arg == nil && other.Arg == nil {
		return true
	}
	if u.Arg == nil || other.Arg == nil {
		return false
	}
	// Same pointer = same value (common for interned slices)
	return u.Arg == other.Arg
}

func (u *UOp) computeHash() uint64 {
	h := uint64(u.Op) * 0x9e3779b97f4a7c15
	h ^= uint64(u.DType.Bits) * 0x517cc1b727220a95
	for _, s := range u.Src {
		h ^= s.hash * 0x6c62272e07bb0142
		h = (h << 13) | (h >> 51)
	}
	h ^= hashAny(u.Arg)
	return h
}

func hashAny(v any) uint64 {
	if v == nil {
		return 0
	}
	// Fast path for common types
	switch val := v.(type) {
	case float64:
		return uint64(val * 2545.0)
	case []int:
		h := uint64(len(val))
		for _, x := range val {
			h = h*31 + uint64(x)
		}
		return h
	default:
		return uint64(len(fmt.Sprintf("%v", v))) * 0x2545f4914f6cdd1d
	}
}

// --- Shape helpers ---

func cloneShape(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

func shapeSize(shape []int) int {
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d < 0 || (d > 0 && n > maxInt/d) {
			return -1
		}
		n *= d
	}
	return n
}

// BroadcastArg stores shapes for broadcast binary ops.
type BroadcastArg struct {
	OutDims []int
	LhsDims []int
	RhsDims []int
}
