package tensor

import "unsafe"

// byteSliceToFloat32 reinterprets a byte slice as float32 without copying.
func byteSliceToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	if len(b)%4 != 0 {
		panic("byteSliceToFloat32: byte length is not a multiple of 4")
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// float32ToByteSlice reinterprets a float32 slice as bytes without copying.
func float32ToByteSlice(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}
