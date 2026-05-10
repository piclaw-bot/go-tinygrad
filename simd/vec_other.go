//go:build !amd64 && !arm64

package simd

func Snrm2(x []float32) float32 { return snrm2Go(x) }

func VecAdd(dst, a, b []float32) { vecAddGo(dst, a, b) }
func VecMul(dst, a, b []float32) { vecMulGo(dst, a, b) }

func VecScaleAdd(dst, a, b []float32, scale float32) { vecScaleAddGo(dst, a, b, scale) }

func VecScale(dst, a []float32, scale float32) { vecScaleGo(dst, a, scale) }

func VecSiLUMul(dst, a, b []float32) { vecSiLUMulGo(dst, a, b) }

func RMSNorm(x, w []float32, eps float32) { rmsNormGo(x, w, eps) }

func RMSNormNoScale(x []float32, eps float32) { rmsNormNoScaleGo(x, eps) }

func GELUTanhMul(dst, a, b []float32) { geluTanhMulGo(dst, a, b) }

func RMSNormBF16(x, w []float32, eps float32) { rmsNormBF16Go(x, w, eps) }

func ToBF16(x []float32) { toBF16Go(x) }

func init() { HasVecAsm = false }

func BF16DotAsm(x, y []uint16) float32              { return BF16Dot(x, y) }
func BF16RMSNormAsm(x, w []uint16, eps float32)     { BF16RMSNorm(x, w, eps) }
func BF16VecAddAsm(dst, a, b []uint16)              { BF16VecAdd(dst, a, b) }
func BF16WidenToF32(dst []float32, src []uint16)    { bf16WidenToF32Go(dst, src) }
func BF16NarrowFromF32(dst []uint16, src []float32) { bf16NarrowFromF32Go(dst, src) }
