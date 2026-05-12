package vulkan

import (
	"fmt"
	"os"
)

func debugf(format string, args ...any) {
	if os.Getenv("GO_PHERENCE_VULKAN_DEBUG") != "" {
		fmt.Printf(format, args...)
	}
}

func debugln(args ...any) {
	if os.Getenv("GO_PHERENCE_VULKAN_DEBUG") != "" {
		fmt.Println(args...)
	}
}
