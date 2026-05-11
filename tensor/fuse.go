package tensor

import "math"

// tryFuse examines a UOp DAG and fuses chains of elementwise ops into a
// single kernel that evaluates the full expression tree per-element in one pass.
//
// Fusible: all unary and binary ALU ops (Add, Mul, Neg, Sqrt, etc.)
// Breaks fusion: Reduce, MatMul, Permute, realized Buffers (become leaves)

type fusedOp struct {
	op     Ops
	isLeaf bool
	srcIdx [2]int // inputs in the op list (-1 = none)
	bufIdx int    // index into buffers (for leaves)
}

type fusedKernel struct {
	ops  []fusedOp
	bufs []*Buffer
	n    int
}

func tryFuse(u *UOp, shape Shape) *fusedKernel {
	if u == nil || shape.Numel() < 0 || !canFuse(u) {
		return nil
	}
	k := &fusedKernel{n: shape.Numel()}
	visited := map[*UOp]int{}

	var walk func(node *UOp) int
	walk = func(node *UOp) int {
		if node == nil {
			panic("fuse: nil node")
		}
		if idx, ok := visited[node]; ok {
			return idx
		}
		// Leaf: already-realized buffer, non-fusible op, or broadcast op
		if node.buf != nil || !isFusible(node.Op) || !canFuse(node) {
			if node.buf == nil {
				node.buf = realize(node, shape)
			}
			bufIdx := len(k.bufs)
			k.bufs = append(k.bufs, node.buf)
			opIdx := len(k.ops)
			k.ops = append(k.ops, fusedOp{isLeaf: true, bufIdx: bufIdx, srcIdx: [2]int{-1, -1}})
			visited[node] = opIdx
			return opIdx
		}
		src := [2]int{-1, -1}
		for i, s := range node.Src {
			if i < 2 {
				src[i] = walk(s)
			}
		}
		opIdx := len(k.ops)
		k.ops = append(k.ops, fusedOp{op: node.Op, srcIdx: src})
		visited[node] = opIdx
		return opIdx
	}
	walk(u)
	return k
}

func (k *fusedKernel) execute() *Buffer {
	if k == nil || k.n < 0 || len(k.ops) == 0 {
		panic("fusedKernel: invalid kernel")
	}
	out := allocBuffer(Float32, k.n)
	outData := out.Float32Data()

	bufData := make([][]float32, len(k.bufs))
	for i, b := range k.bufs {
		if b == nil || b.Length < k.n {
			panic("fusedKernel: invalid leaf buffer")
		}
		bufData[i] = b.Float32Data()
	}

	nOps := len(k.ops)
	vals := make([]float32, nOps)

	for i := 0; i < k.n; i++ {
		for j := 0; j < nOps; j++ {
			op := &k.ops[j]
			if op.isLeaf {
				vals[j] = bufData[op.bufIdx][i]
				continue
			}
			a := vals[op.srcIdx[0]]
			switch op.op {
			case OpNeg:
				vals[j] = -a
			case OpSqrt:
				vals[j] = float32(math.Sqrt(float64(a)))
			case OpReciprocal:
				vals[j] = 1.0 / a
			case OpExp2:
				vals[j] = float32(math.Exp2(float64(a)))
			case OpLog2:
				vals[j] = float32(math.Log2(float64(a)))
			case OpAdd:
				vals[j] = a + vals[op.srcIdx[1]]
			case OpSub:
				vals[j] = a - vals[op.srcIdx[1]]
			case OpMul:
				vals[j] = a * vals[op.srcIdx[1]]
			case OpDiv:
				vals[j] = a / vals[op.srcIdx[1]]
			case OpMax:
				b := vals[op.srcIdx[1]]
				if b > a {
					vals[j] = b
				} else {
					vals[j] = a
				}
			}
		}
		outData[i] = vals[nOps-1]
	}
	return out
}

func isFusible(op Ops) bool {
	return op.IsUnary() || op.IsBinary()
}

// canFuse checks if a UOp tree can be fused (no broadcast, no shape mismatches).
func canFuse(u *UOp) bool {
	if u == nil || !isFusible(u.Op) {
		return false
	}
	// Broadcast ops carry BroadcastArg — not fusible with simple per-element indexing
	if _, ok := u.Arg.(BroadcastArg); ok {
		return false
	}
	return true
}
