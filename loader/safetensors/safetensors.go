package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"syscall"

	"github.com/rcarmo/go-pherence/runtime/memory"
)

// TensorInfo describes a tensor stored in a safetensors file.
type TensorInfo struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// File represents a loaded safetensors file.
type File struct {
	Tensors    map[string]TensorInfo
	data       []byte // tensor data region (after header)
	headerSize int
	mmapData   []byte              // full mmap'd region (nil if not mmap'd)
	mmapFd     *os.File            // file handle for mmap'd file (nil if not mmap'd)
	Advisor    *memory.MmapAdvisor // madvise tracking (nil if not mmap'd)
}

var eagerLoadSink byte

// EagerLoad asks the OS to read the mmap'd file and then touches one byte per
// page to fault pages in now instead of during first-token inference.
// It returns the number of bytes covered. Safe to call on non-mmap'd files.
func (f *File) EagerLoad() (int64, error) {
	if f == nil || len(f.mmapData) == 0 {
		return 0, nil
	}
	if f.Advisor != nil {
		if err := f.Advisor.Prefetch(0, int64(len(f.mmapData))); err != nil {
			return 0, fmt.Errorf("safetensors: eager-load prefetch: %w", err)
		}
	}
	pageSize := syscall.Getpagesize()
	if pageSize <= 0 {
		pageSize = 4096
	}
	var sink byte
	for off := 0; off < len(f.mmapData); off += pageSize {
		sink ^= f.mmapData[off]
	}
	// Touch the final byte so short/non-page-aligned files are fully covered.
	sink ^= f.mmapData[len(f.mmapData)-1]
	eagerLoadSink ^= sink
	runtime.KeepAlive(f.mmapData)
	return int64(len(f.mmapData)), nil
}

// Close releases mmap resources. Safe to call on non-mmap'd files.
func (f *File) Close() error {
	if f.mmapData != nil {
		syscall.Munmap(f.mmapData)
		f.mmapData = nil
		f.data = nil
	}
	if f.mmapFd != nil {
		f.mmapFd.Close()
		f.mmapFd = nil
	}
	return nil
}

// Open loads a safetensors file using mmap for zero-copy access.
func Open(path string) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %s: %w", path, err)
	}

	fi, err := fd.Stat()
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("safetensors: stat %s: %w", path, err)
	}
	size := int(fi.Size())
	if size < 8 {
		fd.Close()
		return nil, fmt.Errorf("safetensors: file too short")
	}

	mapped, err := syscall.Mmap(int(fd.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("safetensors: mmap %s: %w", path, err)
	}

	headerLen64 := binary.LittleEndian.Uint64(mapped[:8])
	if headerLen64 > uint64(size-8) {
		syscall.Munmap(mapped)
		fd.Close()
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size", headerLen64)
	}
	headerLen := int(headerLen64)

	var header map[string]json.RawMessage
	if err := json.Unmarshal(mapped[8:8+headerLen], &header); err != nil {
		syscall.Munmap(mapped)
		fd.Close()
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	for key, val := range header {
		if key == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(val, &info); err != nil {
			syscall.Munmap(mapped)
			fd.Close()
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", key, err)
		}
		if err := validateTensorInfo(key, info, size-8-headerLen); err != nil {
			syscall.Munmap(mapped)
			fd.Close()
			return nil, err
		}
		tensors[key] = info
	}

	return &File{
		Tensors:    tensors,
		data:       mapped[8+headerLen:],
		headerSize: headerLen,
		mmapData:   mapped,
		mmapFd:     fd,
		Advisor:    memory.NewMmapAdvisor(mapped),
	}, nil
}

func checkedAddInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a > math.MaxInt64-b {
		return 0, false
	}
	return a + b, true
}

func shapeNumel(shape []int) (int, bool) {
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d < 0 || (d != 0 && n > maxInt/d) {
			return 0, false
		}
		n *= d
	}
	return n, true
}

func dtypeByteSize(dtype string) int {
	switch dtype {
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "I64", "U64":
		return 8
	case "I8", "U8", "BOOL":
		return 1
	default:
		return 0
	}
}

func validateTensorInfo(name string, info TensorInfo, dataLen int) error {
	start, end := info.DataOffsets[0], info.DataOffsets[1]
	if start < 0 || end < start || end > dataLen {
		return fmt.Errorf("safetensors: tensor %q invalid data offsets [%d,%d] for data length %d", name, start, end, dataLen)
	}
	numel, ok := shapeNumel(info.Shape)
	if !ok {
		return fmt.Errorf("safetensors: tensor %q invalid shape %v", name, info.Shape)
	}
	if elemSize := dtypeByteSize(info.DType); elemSize > 0 {
		byteLen := end - start
		maxInt := int(^uint(0) >> 1)
		if numel > maxInt/elemSize || numel*elemSize != byteLen {
			return fmt.Errorf("safetensors: tensor %q shape %v dtype %s expects %d bytes, got %d", name, info.Shape, info.DType, numel*elemSize, byteLen)
		}
	}
	return nil
}

func (f *File) rawTensor(name string) (TensorInfo, []byte, error) {
	if f == nil {
		return TensorInfo{}, nil, fmt.Errorf("safetensors: nil file")
	}
	info, ok := f.Tensors[name]
	if !ok {
		return TensorInfo{}, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if err := validateTensorInfo(name, info, len(f.data)); err != nil {
		return TensorInfo{}, nil, err
	}
	return info, f.data[info.DataOffsets[0]:info.DataOffsets[1]], nil
}

// Names returns all tensor names in sorted order.
func (f *File) Names() []string {
	if f == nil {
		return nil
	}
	names := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// GetFloat32 returns a tensor's data as float32, converting from the stored dtype.
func (f *File) GetFloat32(name string) ([]float32, []int, error) {
	info, raw, err := f.rawTensor(name)
	if err != nil {
		return nil, nil, err
	}

	switch info.DType {
	case "F32":
		if len(raw)%4 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q F32 byte length %d is not divisible by 4", name, len(raw))
		}
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, info.Shape, nil

	case "F16":
		if len(raw)%2 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q F16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, info.Shape, nil

	case "BF16":
		if len(raw)%2 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q BF16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, info.Shape, nil

	case "I64":
		if len(raw)%8 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q I64 byte length %d is not divisible by 8", name, len(raw))
		}
		n := len(raw) / 8
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, info.Shape, nil

	case "I32":
		if len(raw)%4 != 0 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q I32 byte length %d is not divisible by 4", name, len(raw))
		}
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

// ShardedFile represents a sharded safetensors model (multiple files).
type ShardedFile struct {
	shards  map[string]*File  // filename → loaded shard
	mapping map[string]string // tensor name → filename
}

// EagerLoad pre-faults all mapped shards and returns total bytes covered.
func (sf *ShardedFile) EagerLoad() (int64, error) {
	if sf == nil {
		return 0, nil
	}
	var total int64
	for name, f := range sf.shards {
		n, err := f.EagerLoad()
		if err != nil {
			return total, fmt.Errorf("eager load shard %s: %w", name, err)
		}
		var ok bool
		total, ok = checkedAddInt64(total, n)
		if !ok {
			return total, fmt.Errorf("eager load shard %s: byte count overflows", name)
		}
	}
	return total, nil
}

// Close releases all shard resources.
func (sf *ShardedFile) Close() error {
	if sf == nil {
		return nil
	}
	var firstErr error
	for _, f := range sf.shards {
		if f == nil {
			continue
		}
		if err := f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// OpenSharded loads a sharded safetensors model from an index file.
func OpenSharded(indexPath string) (*ShardedFile, error) {
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("read index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("parse index: %w", err)
	}

	// Determine directory
	dir := filepath.Dir(indexPath)

	sf := &ShardedFile{
		shards:  map[string]*File{},
		mapping: index.WeightMap,
	}

	// Load each unique shard file
	shardFiles := map[string]bool{}
	for _, filename := range index.WeightMap {
		shardFiles[filename] = true
	}
	for filename := range shardFiles {
		path := filepath.Join(dir, filename)
		f, err := Open(path)
		if err != nil {
			_ = sf.Close()
			return nil, fmt.Errorf("open shard %s: %w", filename, err)
		}
		sf.shards[filename] = f
	}

	return sf, nil
}

// GetFloat32 returns a tensor's data, looking up the correct shard.
func (sf *ShardedFile) GetFloat32(name string) ([]float32, []int, error) {
	if sf == nil {
		return nil, nil, fmt.Errorf("safetensors: nil sharded file")
	}
	filename, ok := sf.mapping[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not in weight map", name)
	}
	shard, ok := sf.shards[filename]
	if !ok {
		return nil, nil, fmt.Errorf("shard %q not loaded", filename)
	}
	return shard.GetFloat32(name)
}

// Tensors returns all tensor names.
func (sf *ShardedFile) Names() []string {
	if sf == nil {
		return nil
	}
	names := make([]string, 0, len(sf.mapping))
	for k := range sf.mapping {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// GetRaw returns raw bytes and shape for a tensor without conversion.
func (f *File) GetRaw(name string) ([]byte, string, []int, error) {
	t, data, err := f.rawTensor(name)
	if err != nil {
		return nil, "", nil, err
	}
	return data, t.DType, t.Shape, nil
}

// GetInt32 returns a tensor's data as []int32.
func (f *File) GetInt32(name string) ([]int32, []int, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype != "I32" {
		return nil, nil, fmt.Errorf("tensor %q is %s, not I32", name, dtype)
	}
	if len(raw)%4 != 0 {
		return nil, nil, fmt.Errorf("tensor %q I32 byte length %d is not divisible by 4", name, len(raw))
	}
	n := len(raw) / 4
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, shape, nil
}

// ShardedFile GetRaw/GetInt32
func (sf *ShardedFile) GetRaw(name string) ([]byte, string, []int, error) {
	if sf == nil {
		return nil, "", nil, fmt.Errorf("safetensors: nil sharded file")
	}
	filename, ok := sf.mapping[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("tensor %q not in weight map", name)
	}
	shard, ok := sf.shards[filename]
	if !ok || shard == nil {
		return nil, "", nil, fmt.Errorf("shard %q not loaded", filename)
	}
	return shard.GetRaw(name)
}

func (sf *ShardedFile) GetInt32(name string) ([]int32, []int, error) {
	if sf == nil {
		return nil, nil, fmt.Errorf("safetensors: nil sharded file")
	}
	filename, ok := sf.mapping[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not in weight map", name)
	}
	shard, ok := sf.shards[filename]
	if !ok || shard == nil {
		return nil, nil, fmt.Errorf("shard %q not loaded", filename)
	}
	return shard.GetInt32(name)
}

// GetBF16 returns BF16 tensor data as []uint16 without F32 conversion.
// For BF16 dtype, returns raw uint16 values.
// For F32 dtype, converts F32→BF16 (truncation).
// For F16 dtype, converts F16→BF16 via F32 intermediate.
func (f *File) GetBF16(name string) ([]uint16, []int, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		return nil, nil, err
	}

	switch dtype {
	case "BF16":
		if len(raw)%2 != 0 {
			return nil, nil, fmt.Errorf("tensor %q BF16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			out[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		return out, shape, nil

	case "F32":
		if len(raw)%4 != 0 {
			return nil, nil, fmt.Errorf("tensor %q F32 byte length %d is not divisible by 4", name, len(raw))
		}
		n := len(raw) / 4
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4:])
			out[i] = uint16(bits >> 16) // truncate to BF16
		}
		return out, shape, nil

	case "F16":
		if len(raw)%2 != 0 {
			return nil, nil, fmt.Errorf("tensor %q F16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			f32 := float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
			out[i] = uint16(math.Float32bits(f32) >> 16)
		}
		return out, shape, nil

	default:
		return nil, nil, fmt.Errorf("tensor %q is %s, not BF16/F32/F16", name, dtype)
	}
}

// GetBF16 for sharded files
func (sf *ShardedFile) GetBF16(name string) ([]uint16, []int, error) {
	if sf == nil {
		return nil, nil, fmt.Errorf("safetensors: nil sharded file")
	}
	filename, ok := sf.mapping[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not in weight map", name)
	}
	shard, ok := sf.shards[filename]
	if !ok || shard == nil {
		return nil, nil, fmt.Errorf("shard %q not loaded", filename)
	}
	return shard.GetBF16(name)
}
