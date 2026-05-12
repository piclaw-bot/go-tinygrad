package simd

func sdotScalar(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	sum := float32(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		sum += x[i]*y[i] + x[i+1]*y[i+1] + x[i+2]*y[i+2] + x[i+3]*y[i+3] +
			x[i+4]*y[i+4] + x[i+5]*y[i+5] + x[i+6]*y[i+6] + x[i+7]*y[i+7]
	}
	for ; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}

func saxpyScalar(alpha float32, x []float32, y []float32) {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	for i := 0; i < n; i++ {
		y[i] += alpha * x[i]
	}
}
