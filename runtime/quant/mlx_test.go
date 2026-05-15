package quant

import (
	"math"
	"testing"
)

func TestDequantMLX(t *testing.T) {
	// Test MLX affine 4-bit dequantization
	// Create a small 4×8 weight matrix with known values
	outDim, inDim := 4, 8
	groupSize := 4
	bits := 4
	packFactor := 32 / bits // 8 values per uint32

	// Pack values: for each row, pack 8 values into 1 uint32
	// Values: row0=[0,1,2,3,4,5,6,7], row1=[8,9,10,11,12,13,14,15] (mod 16)
	weight := make([]uint32, outDim*(inDim/packFactor))
	for row := 0; row < outDim; row++ {
		var packed uint32
		for i := 0; i < inDim; i++ {
			val := uint32((row*inDim + i) % 16)
			packed |= val << (uint(i) * uint(bits))
		}
		weight[row] = packed
	}

	// Scales and biases: 2 groups per row (groupSize=4, inDim=8)
	numGroups := inDim / groupSize
	scales := make([]float32, outDim*numGroups)
	biases := make([]float32, outDim*numGroups)
	for i := range scales {
		scales[i] = 0.1
		biases[i] = -0.5
	}

	qw := &MLXQuantWeight{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		OutDim:    outDim,
		InDim:     inDim,
		Groups:    numGroups,
		GroupSize: groupSize,
		Bits:      bits,
	}

	out := DequantMLX(qw)

	// Verify: out[row][col] = val * scale + bias
	for row := 0; row < outDim; row++ {
		for col := 0; col < inDim; col++ {
			val := float32((row*inDim + col) % 16)
			expected := val*0.1 + (-0.5)
			got := out[row*inDim+col]
			if math.Abs(float64(got-expected)) > 1e-6 {
				t.Fatalf("row=%d col=%d: got %f, want %f (val=%f)", row, col, got, expected, val)
			}
		}
	}
	t.Log("DequantMLX: OK")
}

func TestGemvMLQ4FastPathMatchesGeneric(t *testing.T) {
	outDim, inDim := 3, 16
	groupSize := 8
	bits := 4
	weight := make([]uint32, outDim*(inDim/8))
	for row := 0; row < outDim; row++ {
		for p := 0; p < inDim/8; p++ {
			var packed uint32
			for i := 0; i < 8; i++ {
				val := uint32((row*11 + p*7 + i*3) & 0xF)
				packed |= val << (uint(i) * 4)
			}
			weight[row*(inDim/8)+p] = packed
		}
	}
	numGroups := inDim / groupSize
	scales := make([]float32, outDim*numGroups)
	biases := make([]float32, outDim*numGroups)
	for i := range scales {
		scales[i] = 0.03 * float32(i+1)
		biases[i] = -0.2 + 0.01*float32(i)
	}
	x := []float32{1, -2, 3, -4, 5, -6, 7, -8, 0.5, -0.25, 1.5, -1.25, 2.5, -2.25, 3.5, -3.25}
	fast := &MLXQuantWeight{Weight: weight, Scales: scales, Biases: biases, OutDim: outDim, InDim: inDim, Groups: numGroups, GroupSize: groupSize, Bits: bits}
	generic := *fast
	generic.GroupSize = 4
	generic.Groups = inDim / generic.GroupSize
	generic.Scales = make([]float32, outDim*generic.Groups)
	generic.Biases = make([]float32, outDim*generic.Groups)
	for row := 0; row < outDim; row++ {
		for g := 0; g < generic.Groups; g++ {
			src := row*numGroups + g/2
			generic.Scales[row*generic.Groups+g] = scales[src]
			generic.Biases[row*generic.Groups+g] = biases[src]
		}
	}
	got := make([]float32, outDim)
	want := make([]float32, outDim)
	GemvMLQ(got, x, fast)
	GemvMLQ(want, x, &generic)
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5 {
			t.Fatalf("row %d: fast=%f generic=%f diff=%g", i, got[i], want[i], diff)
		}
	}
}

func TestGemvMLQ(t *testing.T) {
	// Test MLX on-the-fly GEMV
	outDim, inDim := 4, 8
	groupSize := 4
	bits := 4
	packFactor := 32 / bits

	// Same weight setup as above
	weight := make([]uint32, outDim*(inDim/packFactor))
	for row := 0; row < outDim; row++ {
		var packed uint32
		for i := 0; i < inDim; i++ {
			val := uint32((row*inDim + i) % 16)
			packed |= val << (uint(i) * uint(bits))
		}
		weight[row] = packed
	}

	numGroups := inDim / groupSize
	scales := make([]float32, outDim*numGroups)
	biases := make([]float32, outDim*numGroups)
	for i := range scales {
		scales[i] = 0.1
		biases[i] = -0.5
	}

	qw := &MLXQuantWeight{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		OutDim:    outDim,
		InDim:     inDim,
		Groups:    numGroups,
		GroupSize: groupSize,
		Bits:      bits,
	}

	// Input vector
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	// CPU reference: dequant then multiply
	deq := DequantMLX(qw)
	cpuOut := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		sum := float32(0)
		for col := 0; col < inDim; col++ {
			sum += deq[row*inDim+col] * x[col]
		}
		cpuOut[row] = sum
	}

	// On-the-fly GEMV
	mlqOut := make([]float32, outDim)
	GemvMLQ(mlqOut, x, qw)

	// Compare
	for i := 0; i < outDim; i++ {
		diff := math.Abs(float64(mlqOut[i] - cpuOut[i]))
		if diff > 1e-4 {
			t.Fatalf("row %d: gemvMLQ=%f, reference=%f, diff=%e", i, mlqOut[i], cpuOut[i], diff)
		}
	}
	t.Logf("GemvMLQ vs dequant reference: max diff OK")
}
