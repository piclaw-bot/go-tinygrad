package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/gpu"
	"github.com/rcarmo/go-pherence/model"
)

func main() {
	dir := flag.String("model", "", "model directory")
	maxTokens := flag.Int("n", 256, "max tokens per response")
	useGPU := flag.Bool("gpu", false, "use GPU")
	gpuLayers := flag.Int("gpu-layers", 0, "number of layers on GPU (0=all)")
	turboQuant := flag.Bool("turbo-quant", false, "enable TurboQuant KV cache compression on CPU backend")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: llmchat -model <dir> [-n 256] [-gpu]")
		os.Exit(1)
	}

	if *useGPU {
		model.ForceOnTheFly = true
		if *turboQuant {
			fmt.Fprintln(os.Stderr, "warning: --turbo-quant currently applies to the CPU backend only")
		}
	}

	fmt.Printf("Loading %s...\n", *dir)
	t0 := time.Now()
	m, err := model.LoadLlama(*dir)
	if err != nil {
		log.Fatal(err)
	}
	m.EnableTurboQuant = *turboQuant
	tok, err := model.LoadTokenizer(*dir + "/tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}
	m.Tok = tok
	fmt.Printf("Ready in %.1fs (%d layers, h=%d, %s)\n",
		time.Since(t0).Seconds(), m.Config.NumLayers, m.Config.HiddenSize, m.Config.ModelType)

	var gpuMod *model.GPUModel
	if *useGPU {
		g, err := model.LoadGPUModel(m)
		if err != nil {
			fmt.Printf("GPU failed: %v (using CPU)\n", err)
		} else {
			g.CPU.Tok = tok
			if *gpuLayers > 0 {
				g.GPULayers = *gpuLayers
			}
			gpuMod = g
			defer g.Close()
			defer gpu.Shutdown()
			fmt.Println("GPU ready")
		}
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "/quit" || input == "/exit" {
			break
		}

		ids := tok.Encode(input)

		t1 := time.Now()
		var output []int
		if gpuMod != nil {
			output = gpuMod.Generate(ids, *maxTokens)
		} else {
			output = m.Generate(ids, *maxTokens)
		}
		elapsed := time.Since(t1)

		// Extract generated tokens
		promptLen := len(ids)
		generated := output
		if len(output) > promptLen {
			generated = output[promptLen:]
		}

		// Print tokens, stop on EOS-like
		count := 0
		for _, id := range generated {
			text := tok.InvVocab[id]
			if text == "<eos>" || text == "</s>" || text == "<|endoftext|>" || id == 0 {
				break
			}
			// Skip repeated turn markers for Gemma
			if text == "<turn|>" || text == "<|turn>" {
				break
			}
			fmt.Print(text)
			count++
		}
		fmt.Println()

		tokPerSec := float64(count) / elapsed.Seconds()
		fmt.Printf("[%d tok, %.1f tok/s, %.0fms]\n\n",
			count, tokPerSec, float64(elapsed.Milliseconds()))
	}
}
