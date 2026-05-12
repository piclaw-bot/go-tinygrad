package simd

func init() {
	HasVecAsm = RuntimeCapabilities().HasVec
}
