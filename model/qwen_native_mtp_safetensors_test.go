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

func TestLoadQwenNativeMTPHeadFromTinySafetensors(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	srcMap := fakeQwenMTPTensorSourceFromHead(head)
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := writeTinySafetensors(path, mapFromFakeSource(srcMap)); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()
	loaded, err := LoadQwenNativeMTPHead(SafetensorsQwenNativeMTPTensorSource{File: f}, meta)
	if err != nil {
		t.Fatalf("LoadQwenNativeMTPHead: %v", err)
	}
	if err := ValidateQwenNativeMTPHead(loaded, meta); err != nil {
		t.Fatalf("ValidateQwenNativeMTPHead: %v", err)
	}
}

func TestLoadQwenNativeMTPHeadFromSafetensorsDir(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	dir := t.TempDir()
	if err := writeTinySafetensors(filepath.Join(dir, "model.safetensors"), mapFromFakeSource(fakeQwenMTPTensorSourceFromHead(head))); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	loaded, err := LoadQwenNativeMTPHeadFromSafetensorsDir(dir, meta)
	if err != nil {
		t.Fatalf("LoadQwenNativeMTPHeadFromSafetensorsDir: %v", err)
	}
	if err := ValidateQwenNativeMTPHead(loaded, meta); err != nil {
		t.Fatalf("ValidateQwenNativeMTPHead: %v", err)
	}
}

func TestOpenQwenNativeMTPSafetensorsSource(t *testing.T) {
	dir := t.TempDir()
	if err := writeTinySafetensors(filepath.Join(dir, "model.safetensors"), map[string]*tensor.Tensor{
		"mtp.fc.weight": tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2}),
	}); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	src, err := OpenQwenNativeMTPSafetensorsSource(dir)
	if err != nil {
		t.Fatalf("OpenQwenNativeMTPSafetensorsSource: %v", err)
	}
	defer src.Close()
	got, err := src.Get("mtp.fc.weight", []int{2, 2})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Data()[0] != 1 {
		t.Fatalf("got=%v", got.Data())
	}
}

func TestOpenQwenNativeMTPSafetensorsSourcePrefersSharded(t *testing.T) {
	dir := t.TempDir()
	shard := "model-00001-of-00001.safetensors"
	if err := writeTinySafetensors(filepath.Join(dir, shard), map[string]*tensor.Tensor{
		"mtp.fc.weight": tensor.FromFloat32([]float32{5, 6, 7, 8}, []int{2, 2}),
	}); err != nil {
		t.Fatalf("write shard: %v", err)
	}
	if err := writeTinySafetensors(filepath.Join(dir, "model.safetensors"), map[string]*tensor.Tensor{
		"mtp.fc.weight": tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2}),
	}); err != nil {
		t.Fatalf("write single: %v", err)
	}
	index := map[string]any{"weight_map": map[string]string{"mtp.fc.weight": shard}}
	data, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), data, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	src, err := OpenQwenNativeMTPSafetensorsSource(dir)
	if err != nil {
		t.Fatalf("OpenQwenNativeMTPSafetensorsSource: %v", err)
	}
	defer src.Close()
	got, err := src.Get("mtp.fc.weight", []int{2, 2})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Data()[0] != 5 {
		t.Fatalf("expected sharded source, got data=%v", got.Data())
	}
}

func TestShardedSafetensorsQwenNativeMTPTensorSource(t *testing.T) {
	dir := t.TempDir()
	shard := "model-00001-of-00001.safetensors"
	path := filepath.Join(dir, shard)
	if err := writeTinySafetensors(path, map[string]*tensor.Tensor{
		"mtp.fc.weight": tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2}),
	}); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	index := map[string]any{"weight_map": map[string]string{"mtp.fc.weight": shard}}
	data, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), data, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	f, err := safetensors.OpenSharded(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatalf("OpenSharded: %v", err)
	}
	defer f.Close()
	src := ShardedSafetensorsQwenNativeMTPTensorSource{File: f}
	got, err := src.Get("mtp.fc.weight", []int{2, 2})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Shape()[0] != 2 || got.Shape()[1] != 2 || got.Data()[3] != 4 {
		t.Fatalf("got shape=%v data=%v", got.Shape(), got.Data())
	}
	if _, err := src.Get("mtp.fc.weight", []int{1, 4}); err == nil {
		t.Fatal("shape mismatch returned nil error")
	}
}

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

func mapFromFakeSource(src fakeQwenMTPTensorSource) map[string]*tensor.Tensor {
	out := make(map[string]*tensor.Tensor, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
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
