package quant

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

type fakeMLXTensor struct {
	f32   []float32
	raw   []byte
	dtype string
	shape []int
}

type fakeMLXSource map[string]fakeMLXTensor

func (f fakeMLXSource) GetFloat32(name string) ([]float32, []int, error) {
	t, ok := f[name]
	if !ok || t.f32 == nil {
		return nil, nil, fmt.Errorf("missing f32 %s", name)
	}
	return t.f32, t.shape, nil
}

func (f fakeMLXSource) GetRaw(name string) ([]byte, string, []int, error) {
	t, ok := f[name]
	if !ok || t.raw == nil {
		return nil, "", nil, fmt.Errorf("missing raw %s", name)
	}
	return t.raw, t.dtype, t.shape, nil
}

func packedU32(vals ...uint32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], v)
	}
	return out
}

func packedF32(vals ...float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func validMLXSource() fakeMLXSource {
	return fakeMLXSource{
		"proj.weight": {raw: packedU32(0x76543210, 0xfedcba98), dtype: "U32", shape: []int{2, 1}},
		"proj.scales": {raw: packedF32(0.1, 0.2), dtype: "F32", shape: []int{2, 1}},
		"proj.biases": {raw: packedF32(-0.1, -0.2), dtype: "F32", shape: []int{2, 1}},
	}
}

func TestLoadMLXWeightValidatesAndInfersShape(t *testing.T) {
	qw, err := LoadMLXWeight(validMLXSource(), "proj", 999, 999, 8, 4)
	if err != nil {
		t.Fatalf("LoadMLXWeight: %v", err)
	}
	if qw.OutDim != 2 || qw.InDim != 8 || qw.Groups != 1 || len(qw.Weight) != 2 {
		t.Fatalf("unexpected inferred weight: %+v", qw)
	}
}

func TestLoadMLXWeightRejectsBadConfig(t *testing.T) {
	src := validMLXSource()
	cases := []struct {
		name      string
		outDim    int
		inDim     int
		groupSize int
		bits      int
		want      string
	}{
		{name: "zero bits", outDim: 2, inDim: 8, groupSize: 8, bits: 0, want: "bits"},
		{name: "non divisor bits", outDim: 2, inDim: 8, groupSize: 8, bits: 5, want: "bits"},
		{name: "zero group", outDim: 2, inDim: 8, groupSize: 0, bits: 4, want: "groupSize"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadMLXWeight(src, "proj", tc.outDim, tc.inDim, tc.groupSize, tc.bits)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestLoadMLXWeightRejectsBadInDimWithoutShape(t *testing.T) {
	src := validMLXSource()
	w := src["proj.weight"]
	w.shape = nil
	src["proj.weight"] = w
	_, err := LoadMLXWeight(src, "proj", 2, 7, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "inDim") {
		t.Fatalf("err=%v, want inDim error", err)
	}
}

func TestLoadMLXWeightRejectsBadScaleLength(t *testing.T) {
	src := validMLXSource()
	src["proj.scales"] = fakeMLXTensor{raw: packedF32(0.1), dtype: "F32", shape: []int{1, 1}}
	_, err := LoadMLXWeight(src, "proj", 2, 8, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Fatalf("err=%v, want length mismatch", err)
	}
}

func TestLoadMLXWeightRejectsBadFloatShape(t *testing.T) {
	src := validMLXSource()
	src["proj.biases"] = fakeMLXTensor{raw: packedF32(-0.1, -0.2), dtype: "F32", shape: []int{3, 1}}
	_, err := LoadMLXWeight(src, "proj", 2, 8, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "shape") {
		t.Fatalf("err=%v, want shape error", err)
	}
}

func TestLoadMLXWeightRejectsIntegerScales(t *testing.T) {
	src := validMLXSource()
	src["proj.scales"] = fakeMLXTensor{raw: packedU32(1, 2), dtype: "I32", shape: []int{2, 1}}
	_, err := LoadMLXWeight(src, "proj", 2, 8, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "unsupported dtype I32") {
		t.Fatalf("err=%v, want unsupported I32 dtype", err)
	}
}

func TestLoadMLXWeightRejectsOverflowingShapes(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	src := validMLXSource()
	w := src["proj.weight"]
	w.shape = []int{2, maxInt/2 + 1}
	src["proj.weight"] = w
	_, err := LoadMLXWeight(src, "proj", 2, 8, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("err=%v, want overflow error", err)
	}

	src = validMLXSource()
	src["proj.scales"] = fakeMLXTensor{raw: packedF32(0.1, 0.2), dtype: "F32", shape: []int{maxInt/2 + 1, 3}}
	_, err = LoadMLXWeight(src, "proj", 2, 8, 8, 4)
	if err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("err=%v, want scale shape overflow error", err)
	}
}

func TestValidateMLXQuantWeightAndMalformedUse(t *testing.T) {
	if err := ValidateMLXQuantWeight(nil); err == nil {
		t.Fatal("expected nil weight error")
	}
	bad := &MLXQuantWeight{Bits: 4, OutDim: 2, InDim: 8, GroupSize: 4, Groups: 2}
	if err := ValidateMLXQuantWeight(bad); err == nil {
		t.Fatal("expected missing tensor data error")
	}
	if got := DequantMLX(bad); got != nil {
		t.Fatalf("DequantMLX malformed = %v, want nil", got)
	}
	out := []float32{123}
	GemvMLQ(out, []float32{1}, bad)
	if out[0] != 123 {
		t.Fatalf("GemvMLQ malformed changed output to %f", out[0])
	}
	overflow := &MLXQuantWeight{Weight: []uint32{1}, Scales: []float32{1}, Biases: []float32{0}, OutDim: maxIntForTest(), InDim: 16, Groups: 2, GroupSize: 8, Bits: 4}
	if err := ValidateMLXQuantWeight(overflow); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("overflow ValidateMLXQuantWeight err=%v", err)
	}
	if got := DequantMLX(overflow); got != nil {
		t.Fatalf("DequantMLX overflow len=%d, want nil", len(got))
	}

	good := &MLXQuantWeight{
		Weight:    []uint32{0x11111111, 0x22222222},
		Scales:    []float32{1, 1},
		Biases:    []float32{0, 0},
		OutDim:    2,
		InDim:     8,
		Groups:    1,
		GroupSize: 8,
		Bits:      4,
	}
	if err := ValidateMLXQuantWeight(good); err != nil {
		t.Fatalf("ValidateMLXQuantWeight good: %v", err)
	}
}

func maxIntForTest() int { return int(^uint(0) >> 1) }
