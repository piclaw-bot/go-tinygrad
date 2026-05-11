package gpu

import "testing"

func TestLaunchKernelOnStreamValidation(t *testing.T) {
	if err := LaunchKernelOnStream(0, 1, 1, 1, 1, 1, 1, 0, 0); err == nil {
		t.Fatalf("expected nil function error")
	}
	if err := LaunchKernelOnStream(1, 0, 1, 1, 1, 1, 1, 0, 0); err == nil {
		t.Fatalf("expected invalid dimensions error")
	}
}

func TestCopyDtoDValidation(t *testing.T) {
	if err := CopyDtoD(0, 1, 4); err != nil {
		t.Fatalf("zero destination should be a no-op: %v", err)
	}
	if err := CopyDtoD(1, 0, 4); err != nil {
		t.Fatalf("zero source should be a no-op: %v", err)
	}
	if err := CopyDtoD(1, 1, 0); err != nil {
		t.Fatalf("zero bytes should be a no-op: %v", err)
	}
}
