package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/quant"
	"github.com/rcarmo/go-pherence/tensor"
)

func qwen35LinearInto(out, x []float32, dense *tensor.Tensor, q *Qwen35NVFP4Weight, inDim, outDim int, name string) error {
	if len(out) != outDim || len(x) != inDim {
		return fmt.Errorf("%s vector dims out/x=%d/%d want %d/%d", name, len(out), len(x), outDim, inDim)
	}
	if q != nil {
		if q.W == nil {
			return fmt.Errorf("%s nil NVFP4 weight", name)
		}
		if q.W.InDim != inDim || q.W.OutDim != outDim {
			return fmt.Errorf("%s NVFP4 dims out/in=%d/%d want %d/%d", name, q.W.OutDim, q.W.InDim, outDim, inDim)
		}
		quant.GemvNVFP4(out, x, q.W)
		return nil
	}
	if dense == nil {
		return fmt.Errorf("missing %s", name)
	}
	gemvNT(out, x, dense.Data(), inDim, outDim)
	return nil
}
