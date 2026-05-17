package model

import (
	"math"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func Qwen35RotaryHalf(meta loaderconfig.QwenNativeMTPMetadata) int {
	headDim := meta.HeadDim
	if headDim <= 0 {
		return 0
	}
	factor := meta.PartialRotaryFactor
	if factor <= 0 || factor > 1 {
		factor = 1
	}
	rotDims := int(float64(headDim) * factor)
	if rotDims < 2 {
		rotDims = headDim
	}
	if rotDims%2 != 0 {
		rotDims--
	}
	return rotDims / 2
}

func NewQwen35RoPEFreqs(meta loaderconfig.QwenNativeMTPMetadata, maxSeq int) []float32 {
	rotHalf := Qwen35RotaryHalf(meta)
	if maxSeq <= 0 || rotHalf <= 0 {
		return nil
	}
	theta := meta.RopeTheta
	if theta == 0 {
		theta = 10000
	}
	freqs := make([]float32, maxSeq*rotHalf*2)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < rotHalf; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(meta.HeadDim))
			angle := float64(pos) * freq
			off := (pos*rotHalf + i) * 2
			freqs[off] = float32(math.Cos(angle))
			freqs[off+1] = float32(math.Sin(angle))
		}
	}
	return freqs
}
