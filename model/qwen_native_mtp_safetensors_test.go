package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestSafetensorsQwenNativeMTPTensorSource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := writeTinySafetensors(path, map[string]*tensor.Tensor{
		"mtp.fc.weight": tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2}),
	}); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()
	src := SafetensorsQwenNativeMTPTensorSource{File: f}
	got, err := src.Get("mtp.fc.weight", []int{2, 2})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Shape()[0] != 2 || got.Shape()[1] != 2 || got.Data()[2] != 3 {
		t.Fatalf("got shape=%v data=%v", got.Shape(), got.Data())
	}
	if _, err := src.Get("mtp.fc.weight", []int{1, 4}); err == nil {
		t.Fatal("shape mismatch returned nil error")
	}
}

func writeTinySafetensors(path string, tensors map[string]*tensor.Tensor) error {
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for name, t := range tensors {
		start := len(data)
		for _, v := range t.Data() {
			bits := math.Float32bits(v)
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], bits)
			data = append(data, buf[:]...)
		}
		header[name] = entry{DType: "F32", Shape: t.Shape(), DataOffsets: [2]int{start, len(data)}}
	}
	h, err := json.Marshal(header)
	if err != nil {
		return err
	}
	out := make([]byte, 8+len(h)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(h)))
	copy(out[8:], h)
	copy(out[8+len(h):], data)
	return os.WriteFile(path, out, 0o644)
}
