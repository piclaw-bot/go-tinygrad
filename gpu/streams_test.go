package gpu

import "testing"

func TestLaunchKernelOnStreamValidation(t *testing.T) {
	if err := LaunchKernelOnStream(0, 1, 1, 1, 1, 1, 1, 0, 0); err == nil {
		t.Fatalf("expected nil function error")
	}
	if err := LaunchKernelOnStream(1, 0, 1, 1, 1, 1, 1, 0, 0); err == nil {
		t.Fatalf("expected invalid dimensions error")
	}
	if err := LaunchKernelOnStream(1, 1, 1, 1, 1, 1, 1, 0, 0, nil); err == nil {
		t.Fatalf("expected nil argument error")
	}
}

func TestCapturedGraphNilSafety(t *testing.T) {
	var cg *CapturedGraph
	if err := cg.Launch(); err == nil {
		t.Fatal("nil graph launch returned nil error")
	}
	cg.Destroy()
	cg = &CapturedGraph{}
	if err := cg.Launch(); err == nil {
		t.Fatal("empty graph launch returned nil error")
	}
	cg.Destroy()
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
