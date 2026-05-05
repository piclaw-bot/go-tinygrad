package gpu

// Unified kernel loader: merges ALL PTX entries into one module.
// Solves the cuModuleLoadData error 201 (can't load multiple modules).

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

var (
	megaModuleOnce sync.Once
	megaModule     CUmodule
	megaModuleOK   bool

	// Function handles extracted from the mega module
	sgemmReady bool

	// Set in sgemm.go, kernels_ptx.go, attn_ptx.go, gemv_q4_ptx.go
)

// stripPTXHeader removes .version/.target/.address_size lines from PTX
func stripPTXHeader(ptx string) string {
	var lines []string
	for _, l := range strings.Split(ptx, "\n") {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, ".version") ||
			strings.HasPrefix(trimmed, ".target") ||
			strings.HasPrefix(trimmed, ".address_size") {
			continue
		}
		lines = append(lines, l)
	}
	return strings.Join(lines, "\n")
}

// loadMegaModule compiles ALL PTX kernels into one CUDA module.
func loadMegaModule() {
	megaModuleOnce.Do(func() {
		if !Init() {
			return
		}

		// Pre-warm allocator
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 64*1024*1024); r == CUDA_SUCCESS {
			cuMemFree(warmPtr)
		}

		// Combine all PTX entries into one module
		var combined strings.Builder
		combined.WriteString(".version 7.0\n.target sm_80\n.address_size 64\n\n")

		entries := []struct {
			name string
			ptx  string
		}{
			{"sgemm_nn", SgemmPTX},
			{"vec_add", VecAddPTX},
			{"vec_mul", VecMulPTX},
			{"vec_scale", VecScalePTX},
			{"vec_silu", VecSiLUPTX},
			{"rms_norm", RmsNormPTX},
			{"rope_apply", RoPEPTX},
			{"rope_partial", RoPEPartialPTX},
			{"gqa_attention", AttentionPTX},
			{"gelu_tanh_mul", GELUTanhMulPTX},
			{"gemv_q4sym", GemvQ4OptPTX},
			{"fused_silu_mul", FusedSiLUMulPTX},
			{"prefetch_l2", PrefetchPTX},
			{"gemm_q4sym", GemmQ4PTX},
			{"lm_head_gemv", LMHeadPTX},
			{"mlx_gemv", MLXGemvPTX},
			{"mlx_gemm", MLXGemmPTX},
			{"mlx_correct", MLXCorrectPTX},
			{"bf16_rms_norm", BF16RMSNormPTX},
			{"bf16_rms_norm_no_scale", BF16RMSNormNoScalePTX},
			{"bf16_vec_add", BF16VecAddPTX},
			{"bf16_silu_mul", BF16SiLUMulPTX},
			{"bf16_gelu_tanh_mul", BF16GELUTanhMulPTX},
			{"rms_norm_no_scale", RmsNormNoScalePTX},
		}

		for _, e := range entries {
			combined.WriteString(stripPTXHeader(e.ptx))
			combined.WriteString("\n")
		}

		ptxStr := combined.String()
		ptxBytes := append([]byte(ptxStr), 0)

		EnsureContext()
		if r := cuModuleLoadData(&megaModule, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
			fmt.Printf("[gpu] mega module load failed: error %d\n", r)
			return
		}

		// Extract all function handles
		allOK := true
		extractFn := func(name string) CUfunction {
			nameBytes := append([]byte(name), 0)
			var fn CUfunction
			if r := cuModuleGetFunction(&fn, megaModule, unsafe.Pointer(&nameBytes[0])); r != CUDA_SUCCESS {
				fmt.Printf("[gpu] get %s: error %d\n", name, r)
				allOK = false
				return 0
			}
			return fn
		}

		sgemmFn = extractFn("sgemm_nn")
		fnVecAdd = extractFn("vec_add")
		fnVecMul = extractFn("vec_mul")
		fnVecScale = extractFn("vec_scale")
		fnVecSilu = extractFn("vec_silu")
		fnRmsNorm = extractFn("rms_norm")
		ropeFn = extractFn("rope_apply")
		ropePartialFn = extractFn("rope_partial")
		attnFn = extractFn("gqa_attention")
		q4Fn = extractFn("gemv_q4sym")
		fnFusedSiLUMul = extractFn("fused_silu_mul")
		fnPrefetch = extractFn("prefetch_l2")
		fnGemmQ4 = extractFn("gemm_q4sym")
		fnLMHead = extractFn("lm_head_gemv")
		fnMLXGemv = extractFn("mlx_gemv")
		fnMLXGemm = extractFn("mlx_gemm")
		fnMLXCorrect = extractFn("mlx_correct")
		fnBF16RMSNorm = extractFn("bf16_rms_norm")
		fnBF16RMSNormNoScale = extractFn("bf16_rms_norm_no_scale")
		fnBF16VecAdd = extractFn("bf16_vec_add")
		fnBF16SiLUMul = extractFn("bf16_silu_mul")
		fnBF16GELUTanhMul = extractFn("bf16_gelu_tanh_mul")
		fnRmsNormNoScale = extractFn("rms_norm_no_scale")
		fnGELUTanhMul = extractFn("gelu_tanh_mul")

		if allOK {
			megaModuleOK = true
			sgemmReady = true
			sgemmOK = true
			kernelsLoaded = true
			ropeReady = true
			ropePartialReady = true
			attnReady = true
			q4Ready = true
			fusedSiLUMulOK = true
			fmt.Printf("[gpu] All %d kernels loaded in 1 module\n", len(entries))
			// Initialize streams for prefetch overlap
			if err := initStreams(); err != nil {
				fmt.Printf("[gpu] streams: %v\n", err)
			}
			// Try native BF16 kernels (Ampere+)
			InitNativeBF16()
		}
	})
}

// InitAllKernels loads all GPU kernels. Call from the CUDA-owning thread.
func InitAllKernels() {
	loadMegaModule()
}

// SgemmReady returns true if GPU SGEMM is available.
func SgemmReady() bool {
	loadMegaModule()
	return sgemmReady
}

// Q4Ready returns true if the INT4 GPU kernel is available.
func Q4Ready() bool {
	loadMegaModule()
	return q4Ready
}
