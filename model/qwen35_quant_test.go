package model

import (
	"encoding/binary"
	"fmt"
	"testing"
)

type fakeRawTensorSource map[string]struct {
	raw   []byte
	dtype string
	shape []int
}

func (f fakeRawTensorSource) GetRaw(name string) ([]byte, string, []int, error) {
	v, ok := f[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("missing %s", name)
	}
	return v.raw, v.dtype, v.shape, nil
}

func TestLoadQwen35NVFP4WeightCandidates(t *testing.T) {
	var scale2 [4]byte
	binary.LittleEndian.PutUint32(scale2[:], 0x3f800000)
	src := fakeRawTensorSource{
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight":         {raw: make([]byte, 16), dtype: "U8", shape: []int{2, 8}},
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale":   {raw: []byte{0x38, 0x38}, dtype: "F8_E4M3", shape: []int{2, 1}},
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_2": {raw: scale2[:], dtype: "F32", shape: nil},
	}
	got, err := LoadQwen35NVFP4WeightCandidates(src, "model.layers.0.linear_attn.in_proj_qkvz.weight", []int{2, 16})
	if err != nil {
		t.Fatalf("LoadQwen35NVFP4WeightCandidates: %v", err)
	}
	if got.Name != "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" {
		t.Fatalf("name=%q", got.Name)
	}
}

func TestLoadQwen35NVFP4Weight(t *testing.T) {
	var scale2 [4]byte
	binary.LittleEndian.PutUint32(scale2[:], 0x3f800000)
	src := fakeRawTensorSource{
		"w.weight":         {raw: make([]byte, 16), dtype: "U8", shape: []int{2, 8}},
		"w.weight_scale":   {raw: []byte{0x38, 0x38}, dtype: "F8_E4M3", shape: []int{2, 1}},
		"w.weight_scale_2": {raw: scale2[:], dtype: "F32", shape: nil},
	}
	got, err := LoadQwen35NVFP4Weight(src, "w.weight", []int{2, 16})
	if err != nil {
		t.Fatalf("LoadQwen35NVFP4Weight: %v", err)
	}
	if got.W.OutDim != 2 || got.W.InDim != 16 || got.W.Groups != 1 || got.W.GroupSize != 16 {
		t.Fatalf("weight=%+v", got.W)
	}
	if got.W.WeightScale2 != 1 {
		t.Fatalf("scale2=%v", got.W.WeightScale2)
	}
	if _, err := LoadQwen35NVFP4Weight(src, "w.weight", []int{2, 8}); err == nil {
		t.Fatal("bad wanted shape returned nil error")
	}
}
