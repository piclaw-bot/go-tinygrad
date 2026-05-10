package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/model"
)

func main() {
	dir := flag.String("model", "", "model directory")
	prompt := flag.String("prompt", "The meaning of life is", "input prompt")
	tokens := flag.Int("tokens", 50, "tokens to generate")
	useGPU := flag.Bool("gpu", false, "use GPU-resident forward pass")
	gpuLayers := flag.Int("gpu-layers", 0, "number of layers on GPU (0=all)")
	turboQuant := flag.Bool("turbo-quant", false, "enable TurboQuant KV cache compression on CPU backend")
	eagerLoad := flag.Bool("eager-load", false, "pre-fault mmap'd model weights at startup")
	flag.Parse()

	if *eagerLoad {
		os.Setenv("GO_PHERENCE_EAGER_LOAD", "1")
	}
	if *useGPU {
		model.ForceOnTheFly = true
		if *turboQuant {
			fmt.Fprintln(os.Stderr, "warning: --turbo-quant currently applies to the CPU backend only")
		}
	}

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: llmgen -model <dir> [-prompt text] [-tokens N]")
		os.Exit(1)
	}

	fmt.Printf("Loading model from %s...\n", *dir)
	t0 := time.Now()
	m, err := model.LoadLlama(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	m.EnableTurboQuant = *turboQuant
	fmt.Printf("Loaded in %.2fs (%d layers, h=%d)\n", time.Since(t0).Seconds(),
		m.Config.NumLayers, m.Config.HiddenSize)

	tok, err := model.LoadTokenizer(*dir + "/tokenizer.json")
	if err == nil {
		m.Tok = tok
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}

	ids := tok.Encode(*prompt)

	var gpuMod *model.GPUModel
	if *useGPU {
		var err error
		gpuMod, err = model.LoadGPUModel(m)
		if err != nil {
			fmt.Printf("GPU model failed: %v (falling back to CPU)\n", err)
		} else if *gpuLayers > 0 {
			gpuMod.GPULayers = *gpuLayers
		}
	}

	fmt.Printf("Prompt: '%s' (%d tokens)\n", *prompt, len(ids))
	fmt.Printf("Generating %d tokens...\n\n", *tokens)

	start := time.Now()
	var output []int
	if gpuMod != nil {
		output = append(ids, gpuMod.Generate(ids, *tokens)...)
	} else {
		output = m.Generate(ids, *tokens)
	}
	elapsed := time.Since(start)

	generated := output[len(ids):]
	text := tok.Decode(output)
	genText := tok.Decode(generated)

	fmt.Printf("--- Output ---\n%s\n--- End ---\n\n", text)
	fmt.Printf("Prompt tokens:    %d\n", len(ids))
	fmt.Printf("Generated tokens: %d\n", len(generated))
	fmt.Printf("Total time:       %.2fs\n", elapsed.Seconds())

	if len(generated) > 0 {
		promptTime := elapsed.Seconds() * float64(len(ids)) / float64(len(output))
		genTime := elapsed.Seconds() - promptTime
		tokPerSec := float64(len(generated)) / genTime
		fmt.Printf("Generation time:  %.2fs\n", genTime)
		fmt.Printf("Tokens/sec:       %.1f\n", tokPerSec)
		fmt.Printf("ms/token:         %.1f\n", genTime/float64(len(generated))*1000)
	}
	_ = genText
}
