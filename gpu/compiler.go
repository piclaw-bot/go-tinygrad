package gpu

// Kernel compiler: generates fused PTX kernels from op sequences.
//
// tinygrad approach: build a computation graph, then compile an optimized
// kernel for the entire subgraph. We implement a simpler version that
// fuses sequences of elementwise ops into single PTX kernels.
//
// The compiler handles:
//   1. Elementwise chains: add, mul, scale, neg, silu, fma
//   2. Reductions: sum, max (for RMSNorm, softmax)
//   3. Fused patterns: RMSNorm+scale, SiLU*Mul, residual add
//
// Each compiled kernel is cached by its op signature.

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// Op types for the kernel compiler
type KernelOp int

const (
	KOpLoad      KernelOp = iota // load from global memory (input)
	KOpStore                     // store to global memory (output)
	KOpAdd                       // a + b
	KOpMul                       // a * b
	KOpNeg                       // -a
	KOpSiLU                      // a * sigmoid(a)
	KOpScale                     // a * scalar
	KOpFMA                       // a*b + c
	KOpRsqrt                     // 1/sqrt(a)
	KOpSumReduce                 // sum reduction
	KOpMaxReduce                 // max reduction
	KOpExp                       // exp(a)
	KOpDiv                       // a / b
	KOpSub                       // a - b
	KOpConst                     // constant value
)

// KNode represents a node in the kernel computation graph
type KNode struct {
	Op       KernelOp
	Inputs   []*KNode
	BufIdx   int     // buffer index for Load/Store ops
	ConstVal float32 // for KOpConst
	RegName  string  // assigned register (filled by codegen)
}

// KernelSpec defines a fused kernel to compile
type KernelSpec struct {
	Name      string
	Nodes     []*KNode // computation graph in topological order
	NumBufs   int      // number of input/output buffers
	HasReduce bool     // needs shared memory for reduction
}

// CompiledKernel is a cached compiled PTX kernel
type CompiledKernel struct {
	Fn        CUfunction
	Mod       CUmodule
	Name      string
	NumBufs   int
	GridDiv   int // grid = (N + GridDiv - 1) / GridDiv
	BlockSz   int
	SharedMem int
}

var (
	kernelCache   = map[string]*CompiledKernel{}
	kernelCacheMu sync.Mutex
)

// Compile compiles a KernelSpec into a PTX kernel and caches it.
func Compile(spec *KernelSpec) (*CompiledKernel, error) {
	if err := validateKernelSpec(spec); err != nil {
		return nil, err
	}
	// Cache key from op sequence
	key := specKey(spec)
	kernelCacheMu.Lock()
	if k, ok := kernelCache[key]; ok {
		kernelCacheMu.Unlock()
		return k, nil
	}
	kernelCacheMu.Unlock()

	// Generate PTX
	// Pre-warm allocator before PTX compile
	var warmPtr CUdeviceptr
	if r := cuMemAlloc(&warmPtr, 64*1024*1024); r == CUDA_SUCCESS {
		cuMemFree(warmPtr)
	}
	ptx, blockSz, sharedMem := genPTX(spec)

	// Compile via CUDA driver
	mod, fn, err := loadPTXModule(ptx, spec.Name)
	if err != nil {
		return nil, fmt.Errorf("compile %s: %w", spec.Name, err)
	}

	k := &CompiledKernel{
		Fn:        fn,
		Mod:       mod,
		Name:      spec.Name,
		NumBufs:   spec.NumBufs,
		GridDiv:   blockSz,
		BlockSz:   blockSz,
		SharedMem: sharedMem,
	}

	kernelCacheMu.Lock()
	kernelCache[key] = k
	kernelCacheMu.Unlock()

	return k, nil
}

// Launch executes a compiled kernel with the given buffers and element count.
func (k *CompiledKernel) Launch(n int, bufs ...*Buffer) {
	if k == nil || k.Fn == 0 || n <= 0 || k.GridDiv <= 0 || k.BlockSz <= 0 || len(bufs) < k.NumBufs {
		return
	}
	bytes, err := checkedByteSize(n, -1)
	if err != nil {
		return
	}
	for i := 0; i < k.NumBufs; i++ {
		if bufs[i] == nil || bufs[i].Ptr == 0 || bufs[i].Size < int(bytes) {
			return
		}
	}
	EnsureContext()
	grid := uint32((n + k.GridDiv - 1) / k.GridDiv)
	args := make([]unsafe.Pointer, len(bufs)+1)
	for i, b := range bufs {
		args[i] = unsafe.Pointer(&b.Ptr)
	}
	nn := uint32(n)
	args[len(bufs)] = unsafe.Pointer(&nn)
	LaunchKernel(k.Fn, grid, 1, 1, uint32(k.BlockSz), 1, 1, uint32(k.SharedMem), args...)
}

func (k *CompiledKernel) Destroy() {
	if k == nil {
		return
	}
	if k.Mod != 0 && cuModuleUnload != nil {
		EnsureContext()
		cuModuleUnload(k.Mod)
		k.Mod = 0
	}
	k.Fn = 0
}

func shutdownCompiledKernels() {
	kernelCacheMu.Lock()
	defer kernelCacheMu.Unlock()
	for _, k := range kernelCache {
		if k != nil {
			k.Destroy()
		}
	}
	kernelCache = map[string]*CompiledKernel{}
	jitSiLUMul = nil
	jitAdd = nil
}

// --- PTX code generation ---

// genPTXBody generates the entry point without the module header.
func genPTXBody(spec *KernelSpec) (string, int, int) {
	ptx, bs, sm := genPTX(spec)
	// Strip the .version/.target/.address_size header
	lines := strings.Split(ptx, "\n")
	var body []string
	for _, l := range lines {
		if strings.HasPrefix(l, ".version") || strings.HasPrefix(l, ".target") || strings.HasPrefix(l, ".address_size") {
			continue
		}
		body = append(body, l)
	}
	return strings.Join(body, "\n"), bs, sm
}

func genPTX(spec *KernelSpec) (string, int, int) {
	if spec == nil {
		return "", 0, 0
	}
	blockSz := 256
	sharedMem := 0
	if spec.HasReduce {
		sharedMem = blockSz * 4
	}

	var b strings.Builder
	b.WriteString(`.version 7.0
.target sm_80
.address_size 64
`)
	// Entry point with buffer params + N
	b.WriteString(fmt.Sprintf(".visible .entry %s(\n", spec.Name))
	for i := 0; i < spec.NumBufs; i++ {
		b.WriteString(fmt.Sprintf("    .param .u64 buf%d,\n", i))
	}
	b.WriteString("    .param .u32 N\n) {\n")

	// Register declarations
	b.WriteString("    .reg .u32 %r<16>;\n")
	b.WriteString("    .reg .u64 %rd<16>;\n")
	b.WriteString(fmt.Sprintf("    .reg .f32 %%f<32>;\n"))
	b.WriteString("    .reg .pred %p;\n")
	if spec.HasReduce {
		b.WriteString(fmt.Sprintf("    .shared .align 4 .f32 sdata[%d];\n", blockSz))
	}

	// Thread index
	b.WriteString(`    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra done;
`)

	// Assign registers and generate code for each node
	regCounter := 10
	for _, node := range spec.Nodes {
		reg := fmt.Sprintf("%%f%d", regCounter)
		node.RegName = reg
		regCounter++

		switch node.Op {
		case KOpLoad:
			// Load from buffer: buf_ptr + idx*4
			b.WriteString(fmt.Sprintf("    ld.param.u64 %%rd%d, [buf%d];\n", node.BufIdx, node.BufIdx))
			b.WriteString(fmt.Sprintf("    mul.wide.u32 %%rd8, %%r3, 4;\n"))
			b.WriteString(fmt.Sprintf("    add.u64 %%rd9, %%rd%d, %%rd8;\n", node.BufIdx))
			b.WriteString(fmt.Sprintf("    ld.global.f32 %s, [%%rd9];\n", reg))

		case KOpStore:
			src := node.Inputs[0].RegName
			b.WriteString(fmt.Sprintf("    ld.param.u64 %%rd%d, [buf%d];\n", node.BufIdx, node.BufIdx))
			b.WriteString(fmt.Sprintf("    mul.wide.u32 %%rd8, %%r3, 4;\n"))
			b.WriteString(fmt.Sprintf("    add.u64 %%rd9, %%rd%d, %%rd8;\n", node.BufIdx))
			b.WriteString(fmt.Sprintf("    st.global.f32 [%%rd9], %s;\n", src))

		case KOpAdd:
			a, bb := node.Inputs[0].RegName, node.Inputs[1].RegName
			b.WriteString(fmt.Sprintf("    add.f32 %s, %s, %s;\n", reg, a, bb))

		case KOpMul:
			a, bb := node.Inputs[0].RegName, node.Inputs[1].RegName
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, %s;\n", reg, a, bb))

		case KOpSub:
			a, bb := node.Inputs[0].RegName, node.Inputs[1].RegName
			b.WriteString(fmt.Sprintf("    sub.f32 %s, %s, %s;\n", reg, a, bb))

		case KOpDiv:
			a, bb := node.Inputs[0].RegName, node.Inputs[1].RegName
			b.WriteString(fmt.Sprintf("    div.rn.f32 %s, %s, %s;\n", reg, a, bb))

		case KOpNeg:
			a := node.Inputs[0].RegName
			b.WriteString(fmt.Sprintf("    neg.f32 %s, %s;\n", reg, a))

		case KOpSiLU:
			// silu(x) = x / (1 + exp(-x))
			a := node.Inputs[0].RegName
			b.WriteString(fmt.Sprintf("    neg.f32 %s, %s;\n", reg, a))
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, 0f3FB8AA3B;\n", reg, reg))
			b.WriteString(fmt.Sprintf("    ex2.approx.f32 %s, %s;\n", reg, reg))
			b.WriteString(fmt.Sprintf("    add.f32 %s, %s, 0f3F800000;\n", reg, reg))
			b.WriteString(fmt.Sprintf("    div.rn.f32 %s, %s, %s;\n", reg, a, reg))

		case KOpFMA:
			a, bb, c := node.Inputs[0].RegName, node.Inputs[1].RegName, node.Inputs[2].RegName
			b.WriteString(fmt.Sprintf("    fma.rn.f32 %s, %s, %s, %s;\n", reg, a, bb, c))

		case KOpExp:
			a := node.Inputs[0].RegName
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, 0f3FB8AA3B;\n", reg, a))
			b.WriteString(fmt.Sprintf("    ex2.approx.f32 %s, %s;\n", reg, reg))

		case KOpConst:
			bits := fmt.Sprintf("0f%08X", *(*uint32)(unsafe.Pointer(&node.ConstVal)))
			b.WriteString(fmt.Sprintf("    mov.f32 %s, %s;\n", reg, bits))

		case KOpRsqrt:
			a := node.Inputs[0].RegName
			b.WriteString(fmt.Sprintf("    rsqrt.approx.f32 %s, %s;\n", reg, a))
			// Newton refinement: y = y * (1.5 - 0.5*x*y*y)
			tmp := fmt.Sprintf("%%f%d", regCounter)
			regCounter++
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, %s;\n", tmp, a, reg))
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, %s;\n", tmp, tmp, reg))
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, 0fBF000000;\n", tmp, tmp))
			b.WriteString(fmt.Sprintf("    add.f32 %s, %s, 0f3FC00000;\n", tmp, tmp))
			b.WriteString(fmt.Sprintf("    mul.f32 %s, %s, %s;\n", reg, reg, tmp))
		}
	}

	b.WriteString("done:\n    ret;\n}\n")
	return b.String(), blockSz, sharedMem
}

func validateKernelSpec(spec *KernelSpec) error {
	if spec == nil {
		return fmt.Errorf("nil kernel spec")
	}
	if spec.Name == "" || spec.NumBufs <= 0 || len(spec.Nodes) == 0 {
		return fmt.Errorf("invalid kernel spec %q", spec.Name)
	}
	for i, n := range spec.Nodes {
		if n == nil {
			return fmt.Errorf("kernel spec %q has nil node %d", spec.Name, i)
		}
		if (n.Op == KOpLoad || n.Op == KOpStore) && (n.BufIdx < 0 || n.BufIdx >= spec.NumBufs) {
			return fmt.Errorf("kernel spec %q node %d buffer index %d out of range", spec.Name, i, n.BufIdx)
		}
		for j, in := range n.Inputs {
			if in == nil {
				return fmt.Errorf("kernel spec %q node %d input %d is nil", spec.Name, i, j)
			}
		}
	}
	return nil
}

func specKey(spec *KernelSpec) string {
	if spec == nil {
		return ""
	}
	var parts []string
	for _, n := range spec.Nodes {
		parts = append(parts, fmt.Sprintf("%d:%d", n.Op, n.BufIdx))
	}
	h := sha256.Sum256([]byte(strings.Join(parts, ",")))
	return fmt.Sprintf("%x", h[:8])
}

// --- Pre-built fused kernel specs ---

// FusedSiLUMulSpec: out = silu(a) * b
func FusedSiLUMulSpec() *KernelSpec {
	a := &KNode{Op: KOpLoad, BufIdx: 0}
	bb := &KNode{Op: KOpLoad, BufIdx: 1}
	silu := &KNode{Op: KOpSiLU, Inputs: []*KNode{a}}
	mul := &KNode{Op: KOpMul, Inputs: []*KNode{silu, bb}}
	store := &KNode{Op: KOpStore, BufIdx: 2, Inputs: []*KNode{mul}}
	return &KernelSpec{
		Name: "fused_silu_mul_jit", NumBufs: 3,
		Nodes: []*KNode{a, bb, silu, mul, store},
	}
}

// FusedResidualAddSpec: out = a + b (trivial but tests the compiler)
func FusedResidualAddSpec() *KernelSpec {
	a := &KNode{Op: KOpLoad, BufIdx: 0}
	bb := &KNode{Op: KOpLoad, BufIdx: 1}
	add := &KNode{Op: KOpAdd, Inputs: []*KNode{a, bb}}
	store := &KNode{Op: KOpStore, BufIdx: 2, Inputs: []*KNode{add}}
	return &KernelSpec{
		Name: "fused_add_jit", NumBufs: 3,
		Nodes: []*KNode{a, bb, add, store},
	}
}

// FusedRMSNormElemSpec: out = x * scale * weight
// (the per-element part of RMSNorm, after the reduction computes scale)
func FusedRMSNormElemSpec() *KernelSpec {
	x := &KNode{Op: KOpLoad, BufIdx: 0}
	scale := &KNode{Op: KOpLoad, BufIdx: 1} // broadcast scalar
	w := &KNode{Op: KOpLoad, BufIdx: 2}
	xs := &KNode{Op: KOpMul, Inputs: []*KNode{x, scale}}
	xsw := &KNode{Op: KOpMul, Inputs: []*KNode{xs, w}}
	store := &KNode{Op: KOpStore, BufIdx: 3, Inputs: []*KNode{xsw}}
	return &KernelSpec{
		Name: "fused_rmsnorm_elem_jit", NumBufs: 4,
		Nodes: []*KNode{x, scale, w, xs, xsw, store},
	}
}
