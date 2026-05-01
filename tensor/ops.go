package tensor

// Ops enumerates all operations in the computation graph.
type Ops int

const (
	// Memory
	OpBuffer Ops = iota
	OpConst
	OpLoad
	OpStore
	OpCopy

	// Movement (shape-only, no data movement)
	OpReshape
	OpPermute
	OpExpand
	OpPad
	OpShrink
	OpFlip

	// Unary ALU
	OpExp2
	OpLog2
	OpSin
	OpSqrt
	OpReciprocal
	OpNeg
	OpCast

	// Binary ALU
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMod
	OpMax
	OpCmpLt
	OpCmpEq

	// Ternary
	OpWhere

	// Reduce
	OpReduceSum
	OpReduceMax

	// Control
	OpSink
)

var opNames = map[Ops]string{
	OpBuffer: "BUFFER", OpConst: "CONST", OpLoad: "LOAD", OpStore: "STORE", OpCopy: "COPY",
	OpReshape: "RESHAPE", OpPermute: "PERMUTE", OpExpand: "EXPAND", OpPad: "PAD", OpShrink: "SHRINK", OpFlip: "FLIP",
	OpExp2: "EXP2", OpLog2: "LOG2", OpSin: "SIN", OpSqrt: "SQRT", OpReciprocal: "RECIP", OpNeg: "NEG", OpCast: "CAST",
	OpAdd: "ADD", OpSub: "SUB", OpMul: "MUL", OpDiv: "DIV", OpMod: "MOD", OpMax: "MAX", OpCmpLt: "CMPLT", OpCmpEq: "CMPEQ",
	OpWhere: "WHERE",
	OpReduceSum: "REDUCE_SUM", OpReduceMax: "REDUCE_MAX",
	OpSink: "SINK",
}

func (o Ops) String() string {
	if s, ok := opNames[o]; ok {
		return s
	}
	return "UNKNOWN"
}

// IsMovement returns true for shape-only ops.
func (o Ops) IsMovement() bool {
	return o >= OpReshape && o <= OpFlip
}

// IsUnary returns true for unary ALU ops.
func (o Ops) IsUnary() bool {
	return o >= OpExp2 && o <= OpCast
}

// IsBinary returns true for binary ALU ops.
func (o Ops) IsBinary() bool {
	return o >= OpAdd && o <= OpCmpEq
}

// IsReduce returns true for reduce ops.
func (o Ops) IsReduce() bool {
	return o == OpReduceSum || o == OpReduceMax
}
