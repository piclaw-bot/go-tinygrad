package tensor

// DType represents an element data type.
type DType struct {
	Name    string
	Bits    int
	IsFloat bool
	IsInt   bool
	IsBool  bool
}

var (
	Bool    = DType{"bool", 1, false, false, true}
	Int8    = DType{"int8", 8, false, true, false}
	Int16   = DType{"int16", 16, false, true, false}
	Int32   = DType{"int32", 32, false, true, false}
	Int64   = DType{"int64", 64, false, true, false}
	Uint8   = DType{"uint8", 8, false, true, false}
	Uint16  = DType{"uint16", 16, false, true, false}
	Uint32  = DType{"uint32", 32, false, true, false}
	Float16 = DType{"float16", 16, true, false, false}
	Float32 = DType{"float32", 32, true, false, false}
	Float64 = DType{"float64", 64, true, false, false}
)

// ByteSize returns the size of one element in bytes.
func (d DType) ByteSize() int {
	if d.IsBool {
		return 1
	}
	return d.Bits / 8
}

func (d DType) String() string { return d.Name }
