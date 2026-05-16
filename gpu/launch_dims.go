package gpu

func fitsUint32(v int) bool { return v >= 0 && uint64(v) <= uint64(^uint32(0)) }

func grid1DFor(n, block int) (uint32, bool) {
	if n <= 0 || block <= 0 {
		return 0, false
	}
	grid := (uint64(n) + uint64(block) - 1) / uint64(block)
	if grid == 0 || grid > uint64(^uint32(0)) {
		return 0, false
	}
	return uint32(grid), true
}
