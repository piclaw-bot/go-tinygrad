package gpu

import (
	"testing"
	"unsafe"
)

func TestLaunchKernelValidation(t *testing.T) {
	oldLaunch := cuLaunchKernel
	defer func() { cuLaunchKernel = oldLaunch }()
	cuLaunchKernel = nil
	if err := LaunchKernel(1, 1, 1, 1, 1, 1, 1, 0); err == nil {
		t.Fatal("LaunchKernel accepted unavailable CUDA launcher")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	if err := LaunchKernel(0, 1, 1, 1, 1, 1, 1, 0); err == nil {
		t.Fatal("LaunchKernel accepted nil function")
	}
	if err := LaunchKernel(1, 0, 1, 1, 1, 1, 1, 0); err == nil {
		t.Fatal("LaunchKernel accepted zero grid dimension")
	}
}
