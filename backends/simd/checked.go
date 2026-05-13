package simd

const int32Max = int32(^uint32(0) >> 1)

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}

func checkedFloat32ByteOffset(index int) (uintptr, bool) {
	off, ok := checkedMulInt(index, 4)
	if !ok {
		return 0, false
	}
	return uintptr(off), true
}
