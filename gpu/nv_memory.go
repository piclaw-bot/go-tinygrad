package gpu

// NV memory management and compute dispatch via direct ioctl.
// Continues from nv_ioctl.go (device init).

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/unix"
)

// UVM ioctl command numbers (UVM_IOCTL_BASE(i) = i on Linux, except INITIALIZE)
const (
	UVM_REGISTER_GPU              = 37
	UVM_REGISTER_GPU_VASPACE      = 25
	UVM_CREATE_EXTERNAL_RANGE     = 73
	UVM_MAP_EXTERNAL_ALLOCATION   = 33
	UVM_FREE_CMD                  = 34
	UVM_ALLOC_SEMAPHORE_POOL      = 69
)

// UVM_REGISTER_GPU_PARAMS
type uvmRegGPUParams struct {
	GpuUUID     [16]byte // NvProcessorUuid
	NumaEnabled uint32   // NvBool
	NumaNodeID  int32
	RmCtrlFd    int32
	HClient     uint32
	HSmcPartRef uint32
	RmStatus    uint32
}

// NVOS02_PARAMETERS — RM memory allocation
// nvos02WithFD matches nv_ioctl_nvos02_parameters_with_fd (56 bytes)
// Layout: NVOS02_PARAMETERS (48 bytes) + fd (4 bytes) + pad (4 bytes)
type nvos02WithFD struct {
	// NVOS02_PARAMETERS at offset 0
	HRoot         uint32 // 0
	HObjectParent uint32 // 4
	HObjectNew    uint32 // 8
	HClass        uint32 // 12
	Flags         uint32 // 16
	_pad1         uint32 // 20
	PMemory       uint64 // 24
	Limit         uint64 // 32
	Status        uint32 // 40
	_pad2         uint32 // 44
	// fd at offset 48
	FD            int32  // 48
	_pad3         uint32 // 52
}

// nv_ioctl_nvos33_parameters_with_fd — map GPU memory to CPU
// nvos33WithFD matches nv_ioctl_nvos33_parameters_with_fd
type nvos33WithFD struct {
	FD      int32
	_pad    uint32
	HClient uint32
	HDevice uint32
	HMemory uint32
	_pad2   uint32
	Offset  uint64
	Length  uint64
	PLinAddr uint64
	Status  uint32
	Flags   uint32
}

// NVBuffer represents GPU-accessible memory.
type NVBuffer struct {
	dev      *NVDevice
	va       uint64 // virtual address (GPU-accessible)
	size     uint64
	cpuAddr  uintptr
	cpuMem   []byte  // keep mmap alive // mmap'd CPU address (if mapped)
	hMemory  uint32  // RM memory handle
}

// AllocGPUMem allocates GPU memory and optionally maps it to CPU.
func (d *NVDevice) AllocHostMem(size uint64) (*NVBuffer, error) {
	// Align to page
	size = (size + 0xFFF) &^ 0xFFF

	// mmap anonymous page-aligned memory (outside Go heap to avoid GC issues)
	mem, err := unix.Mmap(-1, 0, int(size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_PRIVATE|unix.MAP_ANONYMOUS)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}
	hostAddr := uintptr(unsafe.Pointer(&mem[0]))

	// Register with GPU via NV_ESC_RM_ALLOC_MEMORY (NV01_MEMORY_SYSTEM_OS_DESCRIPTOR)
	handle := d.nextHandle()
	params := nvos02WithFD{
		HRoot:         d.root,
		HObjectParent: d.device,
		HObjectNew:    handle,
		HClass:        NV01_MEMORY_SYSTEM, // 0x003E
		Flags: (0x1 << 4) |  // PHYSICALITY_NONCONTIGUOUS
			(0x1 << 12) |    // COHERENCY_CACHED
			(0x1 << 30),     // MAPPING_NO_MAP
		PMemory: uint64(hostAddr),
		Limit:   size - 1,
		FD:      int32(d.fdDev),
	}

	if err := d.nvIoctl(d.fdDev, NV_ESC_RM_ALLOC_MEMORY, unsafe.Pointer(&params), unsafe.Sizeof(params)); err != nil {
		return nil, fmt.Errorf("register host memory: %w", err)
	}
	if params.Status != 0 {
		return nil, fmt.Errorf("register host memory status: 0x%X", params.Status)
	}

	buf := &NVBuffer{
		dev:     d,
		size:    size,
		cpuAddr: hostAddr,
		hMemory: handle,
	}
	return buf, nil
}

// mapToCPU maps GPU memory to CPU virtual address space.
func (d *NVDevice) mapToCPU(buf *NVBuffer) error {
	// Open a new GPU fd for the mapping
	devPath := fmt.Sprintf("/dev/nvidia%d", 0)
	fd, err := unix.Open(devPath, unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("open %s: %w", devPath, err)
	}

	// Register with control fd
	regFD := nvRegisterFD{CtlFD: int32(d.fdCtl)}
	if err := d.nvIoctl(fd, NV_ESC_REGISTER_FD, unsafe.Pointer(&regFD), unsafe.Sizeof(regFD)); err != nil {
		unix.Close(fd)
		return fmt.Errorf("register fd: %w", err)
	}

	// NV_ESC_RM_MAP_MEMORY
	params := nvos33WithFD{
		FD:      int32(fd),
		HClient: d.root,
		HDevice: d.device,
		HMemory: buf.hMemory,
		Length:  buf.size,
	}

	if err := d.nvIoctl(d.fdCtl, NV_ESC_RM_MAP_MEMORY, unsafe.Pointer(&params), unsafe.Sizeof(params)); err != nil {
		unix.Close(fd)
		return fmt.Errorf("map memory: %w", err)
	}
	if params.Status != 0 {
		unix.Close(fd)
		return fmt.Errorf("map memory status: 0x%X", params.Status)
	}

	// mmap the fd
	addr, err := unix.Mmap(fd, 0, int(buf.size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		unix.Close(fd)
		return fmt.Errorf("mmap: %w", err)
	}

	buf.cpuAddr = uintptr(unsafe.Pointer(&addr[0]))
	return nil
}

// Upload copies host data to GPU buffer.
func (buf *NVBuffer) Upload(data []float32) error {
	if buf.cpuAddr == 0 {
		return fmt.Errorf("buffer not CPU-mapped")
	}
	bytes := len(data) * 4
	if uint64(bytes) > buf.size {
		return fmt.Errorf("data too large: %d > %d", bytes, buf.size)
	}
	// Use the cpuMem slice directly (backed by mmap)
	if buf.cpuMem == nil {
		return fmt.Errorf("buffer not CPU-mapped")
	}
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	copy(buf.cpuMem[:bytes], srcBytes)
	return nil
}

// Download copies GPU buffer to host.
func (buf *NVBuffer) Download(data []float32) error {
	if buf.cpuMem == nil {
		return fmt.Errorf("buffer not CPU-mapped")
	}
	bytes := len(data) * 4
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	copy(dstBytes, buf.cpuMem[:bytes])
	return nil
}

// Free releases GPU memory.
func (buf *NVBuffer) Free() {
	if buf.hMemory != 0 {
		params := nvos00Params{
			HRoot:         buf.dev.root,
			HObjectParent: buf.dev.device,
			HObjectOld:    buf.hMemory,
		}
		buf.dev.nvIoctl(buf.dev.fdCtl, NV_ESC_RM_FREE, unsafe.Pointer(&params), unsafe.Sizeof(params))
		buf.hMemory = 0
	}
}

// SetupVASpace creates VA space and registers GPU with UVM.
func (d *NVDevice) SetupVASpace() error {
	// Allocate FERMI_VASPACE_A
	type vaspaceParams struct {
		Index           uint32
		Flags           uint32
		VASize          uint64
		VAStartInternal uint64
		VALimitInternal uint64
		BigPageSize     uint64
		VABase          uint64
		Pasid           uint64
	}
	vp := vaspaceParams{
		Flags:  0x4, // IS_EXTERNALLY_OWNED (no page faulting in container)
		VABase: 0x1000,
		VASize: 0x1FFFFFB000000,
	}
	var err error
	d.vaspace, err = d.rmAlloc(d.device, FERMI_VASPACE_A, unsafe.Pointer(&vp), uint32(unsafe.Sizeof(vp)))
	if err != nil {
		return fmt.Errorf("alloc vaspace: %w", err)
	}

	// Register GPU with UVM
	regGPU := uvmRegGPUParams{
		GpuUUID: d.gpuUUID,
		RmCtrlFd: int32(d.fdCtl),
		HClient:  d.root,
	}
	if err := d.uvmIoctl(d.fdUVM, UVM_REGISTER_GPU, unsafe.Pointer(&regGPU), unsafe.Sizeof(regGPU)); err != nil {
		return fmt.Errorf("UVM register GPU: %w", err)
	}
	if regGPU.RmStatus != 0 {
		return fmt.Errorf("UVM register GPU status: 0x%X", regGPU.RmStatus)
	}

	// Register GPU VA space with UVM
	regVA := uvmRegisterGPUVASpaceParams{
		GpuUUID:  d.gpuUUID,
		RmCtrlFd: int32(d.fdCtl),
		HClient:  d.root,
		HVASpace: d.vaspace,
	}
	if err := d.uvmIoctl(d.fdUVM, UVM_REGISTER_GPU_VASPACE, unsafe.Pointer(&regVA), unsafe.Sizeof(regVA)); err != nil {
		return fmt.Errorf("UVM register VA space: %w", err)
	}
	if regVA.RmStatus != 0 {
		return fmt.Errorf("UVM register VA space status: 0x%X", regVA.RmStatus)
	}

	fmt.Printf("[nv] VA space setup OK (vaspace=0x%x)\n", d.vaspace)
	return nil
}
