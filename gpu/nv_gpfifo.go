package gpu

// NV GPFifo: GPU command submission queue via direct ioctl.
//
// The GPFifo (Graphics Processing First-In First-Out) is NVIDIA's mechanism
// for submitting work to the GPU. It's a ring buffer of 8-byte entries,
// each pointing to a command buffer + length. The GPU reads entries via
// MMIO doorbell writes.

import (
	"fmt"
	"unsafe"
)

// GPFifo represents a GPU command submission queue.
type GPFifo struct {
	dev      *NVDevice
	handle   uint32
	ring     []uint64 // ring buffer entries (on GPU-visible memory)
	ringBuf  *NVBuffer
	entries  int
	putValue uint64
	token    uint32 // work submit token for MMIO doorbell
}

// Channel group for organizing GPFifos
type ChannelGroup struct {
	handle uint32
}

// SetupChannelGroup creates a Kepler channel group.
func (d *NVDevice) SetupChannelGroup() (*ChannelGroup, error) {
	if d == nil {
		return nil, fmt.Errorf("nil NVDevice")
	}
	type channelGroupParams struct {
		HObjectError                uint32
		HObjectEccError             uint32
		HVASpace                    uint32
		EngineType                  uint32
		BIsCallingContextVgpuPlugin uint8
		_pad                        [3]byte
	}
	cgp := channelGroupParams{
		EngineType: 1, // NV2080_ENGINE_TYPE_GRAPHICS
	}
	handle, err := d.rmAlloc(d.device, KEPLER_CHANNEL_GROUP_A,
		unsafe.Pointer(&cgp), uint32(unsafe.Sizeof(cgp)))
	if err != nil {
		return nil, fmt.Errorf("channel group: %w", err)
	}
	return &ChannelGroup{handle: handle}, nil
}

// SetupContextShare creates a context share for a channel group.
func (d *NVDevice) SetupContextShare(cg *ChannelGroup, vaspace uint32) (uint32, error) {
	if d == nil || cg == nil || cg.handle == 0 {
		return 0, fmt.Errorf("invalid context share target")
	}
	type ctxShareParams struct {
		HVASpace uint32
		Flags    uint32
		SubctxID uint32
		_pad     uint32
	}
	csp := ctxShareParams{
		HVASpace: vaspace,
		Flags:    1, // NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC
	}
	handle, err := d.rmAlloc(cg.handle, FERMI_CONTEXT_SHARE_A,
		unsafe.Pointer(&csp), uint32(unsafe.Sizeof(csp)))
	if err != nil {
		return 0, fmt.Errorf("context share: %w", err)
	}
	return handle, nil
}

// SetupGPFifo creates a GPFifo for compute command submission.
func (d *NVDevice) SetupGPFifo(cg *ChannelGroup, ctxShare uint32, gpuInfo *GPUInfo) (*GPFifo, error) {
	if d == nil || cg == nil || cg.handle == 0 || ctxShare == 0 || gpuInfo == nil || gpuInfo.GPFifoClass == 0 {
		return nil, fmt.Errorf("invalid GPFifo setup inputs")
	}
	entries := 0x10000 // 64K entries

	// Allocate ring buffer memory (GPU-visible, CPU-accessible)
	ringSize := uint64(entries*8 + 0x1000) // ring + control area
	ringBuf, err := d.AllocHostMem(ringSize)
	if err != nil {
		return nil, fmt.Errorf("alloc ring: %w", err)
	}

	// Allocate error notifier
	notifier, err := d.AllocHostMem(48 << 20) // 48MB for notifier
	if err != nil {
		ringBuf.Free()
		return nil, fmt.Errorf("alloc notifier: %w", err)
	}

	// NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS
	type gpfifoParams struct {
		GPFifoOffset  uint64
		GPFifoEntries uint32
		Flags         uint32
		HContextShare uint32
		HVASpace      uint32
		HObjectError  uint32
		HObjectBuffer uint32
		_pad          [256]byte // rest of params
	}
	gfp := gpfifoParams{
		GPFifoOffset:  uint64(ringBuf.cpuAddr),
		GPFifoEntries: uint32(entries),
		HContextShare: ctxShare,
		HObjectError:  notifier.hMemory,
		HObjectBuffer: ringBuf.hMemory,
	}

	gpfifoHandle, err := d.rmAlloc(cg.handle, gpuInfo.GPFifoClass,
		unsafe.Pointer(&gfp), uint32(unsafe.Sizeof(gfp)))
	if err != nil {
		ringBuf.Free()
		notifier.Free()
		return nil, fmt.Errorf("gpfifo alloc: %w", err)
	}

	// Get work submit token
	type getTokenParams struct {
		WorkSubmitToken uint32
	}
	tp := getTokenParams{WorkSubmitToken: 0xFFFFFFFF}
	if err := d.rmControl(gpfifoHandle, 0xC36F0108, // NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN
		unsafe.Pointer(&tp), uint32(unsafe.Sizeof(tp))); err != nil {
		ringBuf.Free()
		notifier.Free()
		return nil, fmt.Errorf("get submit token: %w", err)
	}

	gf := &GPFifo{
		dev:     d,
		handle:  gpfifoHandle,
		ringBuf: ringBuf,
		entries: entries,
		token:   tp.WorkSubmitToken,
	}

	// Ring buffer view (uses pre-allocated cpuMem)
	_ = ringBuf // ring buffer access via cpuMem if needed

	fmt.Printf("[nv] GPFifo created: handle=0x%x, entries=%d, token=%d\n",
		gpfifoHandle, entries, tp.WorkSubmitToken)

	return gf, nil
}
