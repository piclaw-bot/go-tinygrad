package gpu

import (
	"testing"
	"unsafe"
)

func TestNVDeviceHelperValidation(t *testing.T) {
	var dev *NVDevice
	if got := dev.nextHandle(); got != 0 {
		t.Fatalf("nil nextHandle=%d want 0", got)
	}
	if got := dev.allocVA(4096); got != 0 {
		t.Fatalf("nil allocVA=%x want 0", got)
	}
	if _, err := dev.rmAlloc(0, NV01_ROOT_CLIENT, nil, 0); err == nil {
		t.Fatal("nil rmAlloc returned nil error")
	}
	if err := dev.rmControl(0, 0, nil, 0); err == nil {
		t.Fatal("nil rmControl returned nil error")
	}
	dev.Close()
	if err := dev.SetupDevice(0); err == nil {
		t.Fatal("nil SetupDevice returned nil error")
	}

	dev = &NVDevice{vaAllocator: ^uint64(0) - 1024}
	dummy := 0
	dummyPtr := unsafe.Pointer(&dummy)
	if got := dev.allocVA(4096); got != 0 {
		t.Fatalf("overflow allocVA=%x want 0", got)
	}
	if _, err := dev.rmAlloc(0, NV01_ROOT_CLIENT, dummyPtr, 0); err == nil {
		t.Fatal("rmAlloc with invalid fd unexpectedly succeeded")
	}
	if _, err := dev.rmAlloc(0, NV01_ROOT_CLIENT, nil, 4); err == nil {
		t.Fatal("rmAlloc accepted nil params with non-zero size")
	}
	if err := dev.rmControl(0, 0, nil, 4); err == nil {
		t.Fatal("rmControl accepted nil params with non-zero size")
	}
	if err := dev.nvIoctl(-1, NV_ESC_CARD_INFO, dummyPtr, 1); err == nil {
		t.Fatal("nvIoctl accepted negative fd")
	}
	if err := dev.nvIoctl(0, NV_ESC_CARD_INFO, nil, 1); err == nil {
		t.Fatal("nvIoctl accepted nil arg with non-zero size")
	}
	if err := dev.uvmIoctl(-1, UVM_INITIALIZE, dummyPtr, 1); err == nil {
		t.Fatal("uvmIoctl accepted negative fd")
	}
	if err := dev.uvmIoctl(0, UVM_INITIALIZE, nil, 1); err == nil {
		t.Fatal("uvmIoctl accepted nil arg with non-zero size")
	}
}
