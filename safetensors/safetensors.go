package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// TensorInfo describes a tensor stored in a safetensors file.
type TensorInfo struct {
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets [2]int  `json:"data_offsets"`
}

// File represents a loaded safetensors file.
type File struct {
	Tensors    map[string]TensorInfo
	data       []byte // raw mmap'd or loaded data (after header)
	headerSize int
}

// Open loads a safetensors file.
func Open(path string) (*File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read %s: %w", path, err)
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}

	headerLen := int(binary.LittleEndian.Uint64(raw[:8]))
	if 8+headerLen > len(raw) {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size", headerLen)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &header); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	for key, val := range header {
		if key == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(val, &info); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", key, err)
		}
		tensors[key] = info
	}

	return &File{
		Tensors:    tensors,
		data:       raw[8+headerLen:],
		headerSize: headerLen,
	}, nil
}

// Names returns all tensor names in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		names = append(names, k)
	}
	return names
}

// GetFloat32 returns a tensor's data as float32, converting from the stored dtype.
func (f *File) GetFloat32(name string) ([]float32, []int, error) {
	info, ok := f.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}

	raw := f.data[info.DataOffsets[0]:info.DataOffsets[1]]

	switch info.DType {
	case "F32":
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, info.Shape, nil

	case "F16":
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, info.Shape, nil

	case "BF16":
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, info.Shape, nil

	case "I64":
		n := len(raw) / 8
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, info.Shape, nil

	case "I32":
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out, info.Shape, nil

	default:
		return nil, nil, fmt.Errorf("safetensors: unsupported dtype %q for tensor %q", info.DType, name)
	}
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31) // ±0
		}
		// Subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		fallthrough
	case exp < 31:
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	default: // exp == 31: Inf or NaN
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7f800000) // ±Inf
		}
		return math.Float32frombits((sign << 31) | 0x7fc00000) // NaN
	}
}
