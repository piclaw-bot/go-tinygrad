package simd

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// Capabilities describes the CPU features and SIMD entrypoints this binary can use.
// It is intentionally conservative: Go fallback paths remain valid when a feature
// is unavailable at runtime, even on an architecture with assembly files present.
type Capabilities struct {
	Arch     string
	HasAVX2  bool
	HasFMA   bool
	HasNEON  bool
	HasVec   bool
	HasDot   bool
	HasSGEMM bool
	HasBF16  bool
	HasPack  bool
}

// HasDotAsm reports whether Sdot/Saxpy may use architecture-specific assembly.
var HasDotAsm = RuntimeCapabilities().HasDot

// RuntimeCapabilities returns the active SIMD capability set.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	switch runtime.GOARCH {
	case "amd64":
		c.HasAVX2 = cpu.X86.HasAVX2
		c.HasFMA = cpu.X86.HasFMA
		c.HasVec = c.HasAVX2 && c.HasFMA
		c.HasDot = c.HasVec
		c.HasSGEMM = c.HasVec && hasSgemmAsm
		c.HasBF16 = c.HasVec
		c.HasPack = hasAvxPack
	case "arm64":
		c.HasNEON = cpu.ARM64.HasASIMD
		c.HasVec = c.HasNEON
		c.HasDot = c.HasNEON
		c.HasSGEMM = c.HasNEON && hasSgemmAsm
		c.HasBF16 = c.HasNEON
		c.HasPack = hasNeonPack
	}
	return c
}
