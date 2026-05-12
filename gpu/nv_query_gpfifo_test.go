package gpu

import "testing"

func TestNVQueryAndGPFifoValidation(t *testing.T) {
	var dev *NVDevice
	if _, err := dev.QueryGRInfo(GR_INFO_SM_VERSION); err == nil {
		t.Fatal("nil QueryGRInfo returned nil error")
	}
	if _, err := dev.QueryGPUClassList(); err == nil {
		t.Fatal("nil QueryGPUClassList returned nil error")
	}
	if _, err := dev.QueryGPUInfo(); err == nil {
		t.Fatal("nil QueryGPUInfo returned nil error")
	}
	if _, err := dev.SetupChannelGroup(); err == nil {
		t.Fatal("nil SetupChannelGroup returned nil error")
	}
	if _, err := dev.SetupContextShare(nil, 0); err == nil {
		t.Fatal("nil SetupContextShare returned nil error")
	}
	if _, err := dev.SetupGPFifo(nil, 0, nil); err == nil {
		t.Fatal("nil SetupGPFifo returned nil error")
	}

	dev = &NVDevice{}
	if _, err := dev.QueryGRInfo(GR_INFO_SM_VERSION); err == nil {
		t.Fatal("uninitialized QueryGRInfo returned nil error")
	}
	if _, err := dev.QueryGPUClassList(); err == nil {
		t.Fatal("uninitialized QueryGPUClassList returned nil error")
	}
	if _, err := dev.SetupContextShare(&ChannelGroup{}, 0); err == nil {
		t.Fatal("zero handle SetupContextShare returned nil error")
	}
	if _, err := dev.SetupGPFifo(&ChannelGroup{handle: 1}, 0, &GPUInfo{GPFifoClass: 1}); err == nil {
		t.Fatal("zero ctxShare SetupGPFifo returned nil error")
	}
	if _, err := dev.SetupGPFifo(&ChannelGroup{handle: 1}, 1, &GPUInfo{}); err == nil {
		t.Fatal("zero GPFifoClass SetupGPFifo returned nil error")
	}
}
