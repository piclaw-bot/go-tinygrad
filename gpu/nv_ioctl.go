package gpu

// NVIDIA GPU direct ioctl interface — pure Go, no libcuda, no CGo.
//
// Talks to /dev/nvidiactl + /dev/nvidia0 + /dev/nvidia-uvm via raw ioctl syscalls.
// Based on NVIDIA open-source kernel module headers and tinygrad's ops_nv.py.
//
// Architecture:
//   1. Open device nodes
//   2. RM API: allocate root client, device, subdevice, VA space
//   3. UVM: register GPU, allocate/map memory
//   4. GPFifo: submit compute commands
//   5. QMD: dispatch CUDA kernels

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// NVIDIA ioctl escape codes (NV_IOCTL_BASE = 200)
const (
	nvIoctlMagic = 'F'

	NV_ESC_CARD_INFO         = 200
	NV_ESC_REGISTER_FD       = 201
	NV_ESC_RM_ALLOC          = 0x2B // 43 - via NV_IOCTL_MAGIC
	NV_ESC_RM_CONTROL        = 0x2A // 42
	NV_ESC_RM_FREE           = 0x29 // 41
	NV_ESC_RM_MAP_MEMORY     = 0x4E
	NV_ESC_RM_ALLOC_MEMORY   = 0x27
	NV_ESC_RM_MAP_MEMORY_DMA = 0x33
)

// NVIDIA RM class IDs
const (
	NV01_ROOT_CLIENT        = 0x0041
	NV01_DEVICE_0           = 0x0080
	NV20_SUBDEVICE_0        = 0x2080
	NV01_MEMORY_VIRTUAL     = 0x00F0
	NV01_MEMORY_SYSTEM      = 0x003E
	NV1_MEMORY_USER         = 0x003D
	FERMI_VASPACE_A         = 0x90F1
	KEPLER_CHANNEL_GROUP_A  = 0xA06C
	FERMI_CONTEXT_SHARE_A   = 0x9067
	AMPERE_CHANNEL_GPFIFO_A = 0xC46F
	AMPERE_COMPUTE_B        = 0xC7C0
	AMPERE_DMA_COPY_B       = 0xC7B5
)

// NVIDIA RM control commands
const (
	NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2          = 0x00000211
	NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2 = 0x0000013e
	NV2080_CTRL_CMD_GPU_GET_GID_INFO            = 0x2080014A
	NV2080_CTRL_CMD_GR_GET_INFO                 = 0x20801201
	NV2080_CTRL_CMD_PERF_BOOST                  = 0x2080200A
	NV0080_CTRL_CMD_GPU_GET_CLASSLIST           = 0x00800201
)

// UVM ioctl commands (these ARE the ioctl numbers, not escape codes)
const (
	UVM_INITIALIZE   = 0x30000001
	UVM_DEINITIALIZE = 0x30000002
)

// --- ioctl struct definitions (matching NVIDIA kernel module) ---

// NVOS21_PARAMETERS — RM object allocation
type nvos21Params struct {
	HRoot         uint32
	HObjectParent uint32
	HObjectNew    uint32
	HClass        uint32
	PAllocParms   uint64 // pointer, 8-byte aligned
	ParamsSize    uint32
	Status        uint32
}

// NVOS54_PARAMETERS — RM control call
type nvos54Params struct {
	HClient    uint32
	HObject    uint32
	Cmd        uint32
	Flags      uint32
	Params     uint64 // pointer, 8-byte aligned
	ParamsSize uint32
	Status     uint32
}

// NVOS00_PARAMETERS — RM object free
type nvos00Params struct {
	HRoot         uint32
	HObjectParent uint32
	HObjectOld    uint32
	Status        uint32
}

// nv_ioctl_card_info_t — GPU discovery (matches kernel struct layout)
// NvBool=4, nv_pci_info_t={domain:4, bus:1, slot:1, function:1, pad:1, vendor_id:2, device_id:2}
type nvCardInfo struct {
	Valid uint32
	// nv_pci_info_t
	Domain   uint32
	Bus      uint8
	Slot     uint8
	Function uint8
	_pad1    uint8
	Vendor   uint16
	DeviceID uint16
	// rest of card_info
	GpuID       uint32
	Interrupt   uint16
	_pad2       [6]byte // align to 8
	Reg_address uint64
	Reg_size    uint64
	Fb_address  uint64
	Fb_size     uint64
	Minor       uint32
	DevName     [10]byte
	_pad3       [2]byte // align
}

// nv_ioctl_register_fd_t
type nvRegisterFD struct {
	CtlFD int32
}

// UVM params
type uvmInitParams struct {
	Flags    uint64
	RmStatus uint32
	_pad     uint32
}

type uvmRegisterGPUParams struct {
	GpuUUID     [16]byte
	Flags       uint32
	_pad        uint32
	RmCtrlFd    int32
	HClient     uint32
	HSmcPartRef uint32
	RmStatus    uint32
}

type uvmRegisterGPUVASpaceParams struct {
	GpuUUID  [16]byte
	RmCtrlFd int32
	HClient  uint32
	HVASpace uint32
	RmStatus uint32
}

type uvmCreateExternalRangeParams struct {
	Base     uint64
	Length   uint64
	RmStatus uint32
	_pad     uint32
}

type uvmFreeParams struct {
	Base     uint64
	Length   uint64
	RmStatus uint32
	_pad     uint32
}

// --- NV Device ---

type NVDevice struct {
	fdCtl      int // /dev/nvidiactl
	fdDev      int // /dev/nvidia0
	fdDevAlloc int // /dev/nvidia0 (for memory allocs)
	fdUVM      int // /dev/nvidia-uvm
	fdUVM2     int // second UVM fd

	root      uint32 // root client handle
	device    uint32 // NV01_DEVICE_0
	subdevice uint32 // NV20_SUBDEVICE_0
	vaspace   uint32 // FERMI_VASPACE_A
	virtmem   uint32 // NV01_MEMORY_VIRTUAL

	gpuUUID   [16]byte
	gpuName   string
	smVersion uint32
	numSMs    int

	handleCounter uint32

	vaAllocator uint64 // simple bump allocator for VA space
}

var (
	nvOnce sync.Once
	nvDev  *NVDevice
	nvErr  error
)

// NVInit initializes the direct NVIDIA ioctl interface.
func NVInit() (*NVDevice, error) {
	nvOnce.Do(func() {
		nvDev, nvErr = nvInit()
	})
	return nvDev, nvErr
}

// NVAvailable returns true if direct NVIDIA GPU access works.
func NVAvailable() bool {
	dev, err := NVInit()
	return err == nil && dev != nil
}

func nvInit() (*NVDevice, error) {
	d := &NVDevice{
		handleCounter: 0x1000,
		vaAllocator:   0x1000000000, // start VA at 64GB
	}

	var err error

	// Open device nodes
	d.fdCtl, err = unix.Open("/dev/nvidiactl", unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/nvidiactl: %w", err)
	}
	d.fdDev, err = unix.Open("/dev/nvidia0", unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/nvidia0: %w", err)
	}
	// Keep a separate fd for memory allocs (driver may consume it)
	d.fdDevAlloc, err = unix.Open("/dev/nvidia0", unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/nvidia0 (alloc): %w", err)
	}
	d.fdUVM, err = unix.Open("/dev/nvidia-uvm", unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/nvidia-uvm: %w", err)
	}
	d.fdUVM2, err = unix.Open("/dev/nvidia-uvm", unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open /dev/nvidia-uvm (2): %w", err)
	}

	// Allocate root client
	d.root, err = d.rmAlloc(0, NV01_ROOT_CLIENT, nil, 0)
	if err != nil {
		return nil, fmt.Errorf("RM alloc root: %w", err)
	}

	// Discover GPU
	var cards [64]nvCardInfo
	if err := d.nvIoctl(d.fdCtl, NV_ESC_CARD_INFO, unsafe.Pointer(&cards[0]), unsafe.Sizeof(cards)); err != nil {
		return nil, fmt.Errorf("card info: %w", err)
	}

	if cards[0].Valid == 0 {
		return nil, fmt.Errorf("no GPU found")
	}

	fmt.Printf("[nv] GPU %d: vendor=0x%04x device=0x%04x minor=%d\n",
		0, cards[0].Vendor, cards[0].DeviceID, cards[0].Minor)

	// Register device FDs
	regFD := nvRegisterFD{CtlFD: int32(d.fdCtl)}
	if err := d.nvIoctl(d.fdDev, NV_ESC_REGISTER_FD, unsafe.Pointer(&regFD), unsafe.Sizeof(regFD)); err != nil {
		return nil, fmt.Errorf("register fd: %w", err)
	}
	regFD2 := nvRegisterFD{CtlFD: int32(d.fdCtl)}
	d.nvIoctl(d.fdDevAlloc, NV_ESC_REGISTER_FD, unsafe.Pointer(&regFD2), unsafe.Sizeof(regFD2))

	// Initialize UVM
	uvmInit := uvmInitParams{}
	if err := d.uvmIoctl(d.fdUVM, UVM_INITIALIZE, unsafe.Pointer(&uvmInit), unsafe.Sizeof(uvmInit)); err != nil {
		return nil, fmt.Errorf("UVM init: %w", err)
	}
	if uvmInit.RmStatus != 0 {
		return nil, fmt.Errorf("UVM init status: %d", uvmInit.RmStatus)
	}

	// Setup RM device chain
	if err := d.SetupDevice(cards[0].GpuID); err != nil {
		return nil, fmt.Errorf("setup device: %w", err)
	}

	// Setup VA space and UVM
	if err := d.SetupVASpace(); err != nil {
		fmt.Printf("[nv] Warning: VA space setup: %v\n", err)
	}

	fmt.Println("[nv] Direct ioctl interface initialized — pure Go, no libcuda")
	return d, nil
}

func (d *NVDevice) nextHandle() uint32 {
	if d == nil {
		return 0
	}
	d.handleCounter++
	return d.handleCounter
}

func (d *NVDevice) allocVA(size uint64) uint64 {
	if d == nil || size == 0 || size > ^uint64(0)-0xFFF {
		return 0
	}
	// Simple bump allocator, 4KB aligned
	size = (size + 0xFFF) &^ 0xFFF
	addr := d.vaAllocator
	if addr > ^uint64(0)-size {
		return 0
	}
	d.vaAllocator += size
	return addr
}

// --- ioctl helpers ---

func (d *NVDevice) nvIoctl(fd int, cmd uint32, arg unsafe.Pointer, size uintptr) error {
	if fd < 0 || (arg == nil && size != 0) {
		return fmt.Errorf("invalid ioctl fd=%d arg=%p size=%d", fd, arg, size)
	}
	// NVIDIA uses _IOWR('F', cmd, size) = _IOC(3, 'F', cmd, size)
	// _IOC encoding: dir(2) << 30 | size(14) << 16 | type(8) << 8 | nr(8)
	ioctlNum := uintptr(3)<<30 | (size&0x3FFF)<<16 | uintptr(nvIoctlMagic)<<8 | uintptr(cmd)&0xFF
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(fd), ioctlNum, uintptr(arg))
	if errno != 0 {
		return fmt.Errorf("ioctl 0x%x: %v", cmd, errno)
	}
	return nil
}

func (d *NVDevice) uvmIoctl(fd int, cmd uint32, arg unsafe.Pointer, size uintptr) error {
	if fd < 0 || (arg == nil && size != 0) {
		return fmt.Errorf("invalid UVM ioctl fd=%d arg=%p size=%d", fd, arg, size)
	}
	// UVM uses the command directly as the ioctl number
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))
	if errno != 0 {
		return fmt.Errorf("UVM ioctl 0x%x: %v", cmd, errno)
	}
	return nil
}

// rmAlloc allocates an RM object.
func (d *NVDevice) rmAlloc(parent uint32, class uint32, allocParams unsafe.Pointer, allocSize uint32) (uint32, error) {
	if d == nil {
		return 0, fmt.Errorf("nil NVDevice")
	}
	if allocParams == nil && allocSize != 0 {
		return 0, fmt.Errorf("nil allocation params for size %d", allocSize)
	}
	handle := d.nextHandle()
	root := d.root
	if root == 0 {
		root = handle // first alloc is root itself
	}

	params := nvos21Params{
		HRoot:         root,
		HObjectParent: parent,
		HObjectNew:    handle,
		HClass:        class,
		PAllocParms:   uint64(uintptr(allocParams)),
		ParamsSize:    allocSize,
	}

	if err := d.nvIoctl(d.fdCtl, NV_ESC_RM_ALLOC, unsafe.Pointer(&params), unsafe.Sizeof(params)); err != nil {
		return 0, err
	}
	if params.Status != 0 {
		return 0, fmt.Errorf("rmAlloc class 0x%X status: 0x%X", class, params.Status)
	}
	return handle, nil
}

// rmControl calls an RM control method.
func (d *NVDevice) rmControl(object uint32, cmd uint32, ctrlParams unsafe.Pointer, ctrlSize uint32) error {
	if d == nil {
		return fmt.Errorf("nil NVDevice")
	}
	if ctrlParams == nil && ctrlSize != 0 {
		return fmt.Errorf("nil control params for size %d", ctrlSize)
	}
	params := nvos54Params{
		HClient:    d.root,
		HObject:    object,
		Cmd:        cmd,
		Params:     uint64(uintptr(ctrlParams)),
		ParamsSize: ctrlSize,
	}

	if err := d.nvIoctl(d.fdCtl, NV_ESC_RM_CONTROL, unsafe.Pointer(&params), unsafe.Sizeof(params)); err != nil {
		return err
	}
	if params.Status != 0 {
		return fmt.Errorf("rmControl cmd 0x%X status: 0x%X", cmd, params.Status)
	}
	return nil
}

// Close releases all resources.
func (d *NVDevice) Close() {
	if d == nil {
		return
	}
	if d.fdCtl > 0 {
		unix.Close(d.fdCtl)
	}
	if d.fdDev > 0 {
		unix.Close(d.fdDev)
	}
	if d.fdUVM > 0 {
		unix.Close(d.fdUVM)
	}
	if d.fdUVM2 > 0 {
		unix.Close(d.fdUVM2)
	}
}

// Temporary file handle for mmap
var _ = os.NewFile

// SetupDevice creates the RM device/subdevice chain and queries GPU capabilities.
func (d *NVDevice) SetupDevice(gpuID uint32) error {
	if d == nil {
		return fmt.Errorf("nil NVDevice")
	}
	// Allocate NV01_DEVICE_0
	// NV0080_ALLOC_PARAMETERS
	type deviceParams struct {
		DeviceID      uint32
		HClientShare  uint32
		HTargetClient uint32
		HTargetDevice uint32
		Flags         uint32
		_pad          uint32
		VAMode        uint64
	}
	dp := deviceParams{
		DeviceID:     0, // GPU instance 0
		HClientShare: d.root,
		VAMode:       2, // OPTIONAL_MULTIPLE_VASPACES
	}
	var err error
	d.device, err = d.rmAlloc(d.root, NV01_DEVICE_0, unsafe.Pointer(&dp), uint32(unsafe.Sizeof(dp)))
	if err != nil {
		return fmt.Errorf("alloc device: %w", err)
	}

	// Allocate NV20_SUBDEVICE_0
	type subdeviceParams struct {
		SubDeviceID uint32
	}
	sdp := subdeviceParams{}
	d.subdevice, err = d.rmAlloc(d.device, NV20_SUBDEVICE_0, unsafe.Pointer(&sdp), uint32(unsafe.Sizeof(sdp)))
	if err != nil {
		return fmt.Errorf("alloc subdevice: %w", err)
	}

	// Allocate virtual memory handle
	// NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS has just an offset and limit
	type vmemParams struct {
		Offset   uint64
		Limit    uint64
		HVASpace uint32
		_pad     uint32
	}
	vp := vmemParams{
		Limit: 0x1FFFFFFFFFFFF,
	}
	d.virtmem, err = d.rmAlloc(d.device, NV01_MEMORY_VIRTUAL, unsafe.Pointer(&vp), uint32(unsafe.Sizeof(vp)))
	if err != nil {
		// Non-fatal: we can still use UVM without explicit virtmem
		fmt.Printf("[nv] Warning: virtmem alloc failed: %v\n", err)
	}

	// Get GPU UUID for UVM
	type gidInfoParams struct {
		Index  uint32
		Flags  uint32
		Length uint32
		Data   [256]byte
	}
	gid := gidInfoParams{
		Flags:  0x2, // FORMAT_BINARY
		Length: 16,
	}
	if err := d.rmControl(d.subdevice, NV2080_CTRL_CMD_GPU_GET_GID_INFO, unsafe.Pointer(&gid), uint32(unsafe.Sizeof(gid))); err != nil {
		fmt.Printf("[nv] Warning: get GPU UUID failed: %v\n", err)
	} else {
		copy(d.gpuUUID[:], gid.Data[:16])
		fmt.Printf("[nv] GPU UUID: %x\n", d.gpuUUID)
	}

	fmt.Printf("[nv] Device setup OK (device=0x%x subdevice=0x%x)\n", d.device, d.subdevice)
	return nil
}
