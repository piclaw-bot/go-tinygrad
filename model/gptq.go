package model

// GPTQ INT4 dequantization for quantized model inference.
//
// GPTQ format (as used by AutoRound/AutoGPTQ):
//   qweight [in/8, out]  — 8 INT4 values packed per int32
//   scales  [groups, out] — F16 scale per group
//   qzeros  [groups, out/8] — packed INT4 zero points per group
//   g_idx   [in]          — maps each input row to its quantization group
//
// Dequantization: w[i,j] = scale[g,j] * (qw[i,j] - qz[g,j])
// where g = g_idx[i], qw is the 4-bit value, qz is the 4-bit zero point.

import (
	"math"
	"runtime"
	"sync"
)

// DequantGPTQ dequantizes a GPTQ INT4 weight tensor to float32.
//
// Parameters:
//   qweight: [inFeatures/8, outFeatures] packed int32 (8 x 4-bit per int32)
//   scales:  [numGroups, outFeatures] float32 (already converted from F16)
//   qzeros:  [numGroups, outFeatures/8] packed int32
//   gIdx:    [inFeatures] int32 group index per input
//   inFeatures, outFeatures: weight dimensions
//   sym: true if symmetric quantization (zero point = 8 for 4-bit)
//
// Returns: [outFeatures, inFeatures] float32 weight matrix (row-major, row=output)
func DequantGPTQ(qweight, qzeros, gIdx []int32, scales []float32,
	inFeatures, outFeatures int, sym bool) []float32 {

	out := make([]float32, outFeatures*inFeatures)

	for i := 0; i < inFeatures; i++ {
		g := int(gIdx[i]) // group for this input row

		// Extract zero point for each output from qzeros
		// qzeros is [numGroups, outFeatures/8] with 8 x 4-bit per int32
		for j := 0; j < outFeatures; j++ {
			// Extract 4-bit quantized weight
			packIdx := i / 8           // which int32 in qweight
			bitIdx := uint(i%8) * 4    // bit offset within int32
			qw := (qweight[packIdx*outFeatures+j] >> bitIdx) & 0xF

			// Extract 4-bit zero point
			var qz int32
			if sym {
				qz = 8 // symmetric: zero point is always 2^(bits-1)
			} else {
				zPackIdx := j / 8
				zBitIdx := uint(j%8) * 4
				qz = (qzeros[g*(outFeatures/8)+zPackIdx] >> zBitIdx) & 0xF
			}

			// Dequantize: w = scale * (qw - qz)
			scale := scales[g*outFeatures+j]
			out[j*inFeatures+i] = scale * float32(qw-qz)
		}
	}

	return out
}

// DequantGPTQSym is an optimized parallel symmetric dequantization (zero point = 8).
func DequantGPTQSym(qweight, gIdx []int32, scales []float32,
	inFeatures, outFeatures int) []float32 {

	out := make([]float32, outFeatures*inFeatures)
	nPacks := inFeatures / 8

	// Parallelize across output rows
	nWorkers := runtime.NumCPU()
	if nWorkers > outFeatures {
		nWorkers = outFeatures
	}
	var wg sync.WaitGroup
	chunkSize := (outFeatures + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		jStart := w * chunkSize
		jEnd := jStart + chunkSize
		if jEnd > outFeatures {
			jEnd = outFeatures
		}
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
			for packIdx := 0; packIdx < nPacks; packIdx++ {
				qwRow := qweight[packIdx*outFeatures : (packIdx+1)*outFeatures]
				for bit := 0; bit < 8; bit++ {
					i := packIdx*8 + bit
					g := int(gIdx[i])
					bitIdx := uint(bit) * 4
					scaleRow := scales[g*outFeatures : (g+1)*outFeatures]

					for j := jStart; j < jEnd; j++ {
						qw := (qwRow[j] >> bitIdx) & 0xF
						out[j*inFeatures+i] = scaleRow[j] * float32(qw-8)
					}
				}
			}
		}(jStart, jEnd)
	}
	wg.Wait()

	return out
}

// float16ToFloat32 converts a uint16 IEEE 754 half-precision to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
		exp++
		exp += 127 - 15
	} else if exp == 0x1F {
		exp = 0xFF
	} else {
		exp += 127 - 15
	}

	return math.Float32frombits((sign << 31) | (exp << 23) | (frac << 13))
}
