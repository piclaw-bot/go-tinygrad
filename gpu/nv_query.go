package gpu

// NV GPU capability queries via RM control ioctls.

import (
	"fmt"
	"unsafe"
)

// GR info indices (from NV2080_CTRL_GR_INFO_INDEX_*)
const (
	GR_INFO_SM_VERSION       = 12
	GR_INFO_MAX_WARPS_PER_SM = 13
	GR_INFO_NUM_GPCS         = 20
	GR_INFO_NUM_TPC_PER_GPC  = 23
	GR_INFO_NUM_SM_PER_TPC   = 32
)

// NV2080_CTRL_GR_INFO: index(4) + data(4) = 8 bytes
type grInfo struct {
	Index uint32
	Data  uint32
}

// QueryGRInfo queries GPU graphics info using NV2080_CTRL_CMD_GR_GET_INFO.
func (d *NVDevice) QueryGRInfo(indices ...uint32) (map[uint32]uint32, error) {
	if d == nil {
		return nil, fmt.Errorf("nil NVDevice")
	}
	if d.subdevice == 0 {
		return nil, fmt.Errorf("subdevice not initialized")
	}

	n := len(indices)
	if n > int(^uint(0)>>1)/8 {
		return nil, fmt.Errorf("too many GR info indices: %d", n)
	}
	if n == 0 {
		return nil, nil
	}

	// NV2080_CTRL_GR_GET_INFO_PARAMS: grInfoListSize(4) + pad(4) + grInfoList(ptr, 8) + ...
	// But tinygrad uses the inline array approach. Let me use the pointer-based approach.
	infos := make([]grInfo, n)
	for i, idx := range indices {
		infos[i].Index = idx
	}

	type grGetInfoParams struct {
		GrInfoListSize uint32
		_pad           uint32
		GrInfoList     uint64 // pointer to grInfo array
		GrRouteInfo    [16]byte
	}

	params := grGetInfoParams{
		GrInfoListSize: uint32(n),
		GrInfoList:     uint64(uintptr(unsafe.Pointer(&infos[0]))),
	}

	if err := d.rmControl(d.subdevice, NV2080_CTRL_CMD_GR_GET_INFO,
		unsafe.Pointer(&params), uint32(unsafe.Sizeof(params))); err != nil {
		return nil, err
	}

	result := make(map[uint32]uint32, n)
	for _, info := range infos {
		result[info.Index] = info.Data
	}
	return result, nil
}

// QueryGPUClassList queries available GPU classes.
func (d *NVDevice) QueryGPUClassList() ([]uint32, error) {
	if d == nil {
		return nil, fmt.Errorf("nil NVDevice")
	}
	if d.device == 0 {
		return nil, fmt.Errorf("device not initialized")
	}
	// First call to get count
	type classListParams struct {
		NumClasses uint32
		_pad       uint32
		ClassList  uint64 // pointer
	}

	params := classListParams{}
	if err := d.rmControl(d.device, NV0080_CTRL_CMD_GPU_GET_CLASSLIST,
		unsafe.Pointer(&params), uint32(unsafe.Sizeof(params))); err != nil {
		return nil, err
	}

	if params.NumClasses == 0 {
		return nil, nil
	}

	// Second call to get list
	if params.NumClasses > 1<<20 {
		return nil, fmt.Errorf("class list too large: %d", params.NumClasses)
	}
	classes := make([]uint32, params.NumClasses)
	params.ClassList = uint64(uintptr(unsafe.Pointer(&classes[0])))
	if err := d.rmControl(d.device, NV0080_CTRL_CMD_GPU_GET_CLASSLIST,
		unsafe.Pointer(&params), uint32(unsafe.Sizeof(params))); err != nil {
		return nil, err
	}

	return classes[:params.NumClasses], nil
}

// GPUInfo holds queried GPU capabilities.
type GPUInfo struct {
	NumGPCs       int
	NumTPCPerGPC  int
	NumSMPerTPC   int
	MaxWarpsPerSM int
	SMVersion     uint32
	TotalSMs      int
	Arch          string

	ComputeClass uint32
	DMAClass     uint32
	GPFifoClass  uint32
}

// QueryGPUInfo retrieves all GPU capability info.
func (d *NVDevice) QueryGPUInfo() (*GPUInfo, error) {
	if d == nil {
		return nil, fmt.Errorf("nil NVDevice")
	}
	info := &GPUInfo{}

	// Query GR info
	grData, err := d.QueryGRInfo(
		GR_INFO_NUM_GPCS,
		GR_INFO_NUM_TPC_PER_GPC,
		GR_INFO_NUM_SM_PER_TPC,
		GR_INFO_MAX_WARPS_PER_SM,
		GR_INFO_SM_VERSION,
	)
	if err != nil {
		return nil, fmt.Errorf("GR info: %w", err)
	}

	info.NumGPCs = int(grData[GR_INFO_NUM_GPCS])
	info.NumTPCPerGPC = int(grData[GR_INFO_NUM_TPC_PER_GPC])
	info.NumSMPerTPC = int(grData[GR_INFO_NUM_SM_PER_TPC])
	info.MaxWarpsPerSM = int(grData[GR_INFO_MAX_WARPS_PER_SM])
	info.SMVersion = grData[GR_INFO_SM_VERSION]
	info.TotalSMs = info.NumGPCs * info.NumTPCPerGPC * info.NumSMPerTPC

	// Compute arch string (e.g. "sm_86")
	major := (info.SMVersion >> 8) & 0xFF
	minor := info.SMVersion & 0xFF
	if minor > 0xF {
		info.Arch = fmt.Sprintf("sm_%d%d", major, minor>>4)
	} else {
		info.Arch = fmt.Sprintf("sm_%d%d", major, minor)
	}

	// Query class list for compute/DMA/GPFifo classes
	classes, err := d.QueryGPUClassList()
	if err == nil {
		classSet := make(map[uint32]bool, len(classes))
		for _, c := range classes {
			classSet[c] = true
		}

		// Find compute class (prefer newer)
		for _, c := range []uint32{0xCEC0, 0xC7C0, 0xC6C0} { // Blackwell, Ampere, Turing
			if classSet[c] {
				info.ComputeClass = c
				break
			}
		}
		// DMA class
		for _, c := range []uint32{0xCEB5, 0xC7B5, 0xC6B5} {
			if classSet[c] {
				info.DMAClass = c
				break
			}
		}
		// GPFifo class
		for _, c := range []uint32{0xCE6F, 0xC56F, 0xC46F} {
			if classSet[c] {
				info.GPFifoClass = c
				break
			}
		}
	}

	return info, nil
}
