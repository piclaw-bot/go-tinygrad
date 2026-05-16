package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model"
)

type Report struct {
	ModelDir     string  `json:"model_dir"`
	HiddenSize   int     `json:"hidden_size"`
	MTPLayers    int     `json:"mtp_layers"`
	OutputLen    int     `json:"output_len"`
	OutputAbsSum float32 `json:"output_abs_sum"`
	Passed       bool    `json:"passed"`
}

func main() {
	dir := flag.String("model", "", "Qwen3.5/Qwen3.6 native-MTP model directory")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwenmtpsmoke -model <dir>")
		os.Exit(2)
	}
	data, err := os.ReadFile(filepath.Join(*dir, "config.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(2)
	}
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(data)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse config: %v\n", err)
		os.Exit(2)
	}
	head, err := model.LoadQwenNativeMTPHeadFromSafetensorsDir(*dir, meta)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load MTP head: %v\n", err)
		os.Exit(2)
	}
	embedding := make([]float32, meta.HiddenSize)
	hidden := make([]float32, meta.HiddenSize)
	if meta.HiddenSize > 0 {
		embedding[0] = 1
	}
	if meta.HiddenSize > 1 {
		hidden[1] = 1
	}
	out, err := head.ForwardOne(embedding, hidden, 0, nil, 1e-6, meta)
	if err != nil {
		fmt.Fprintf(os.Stderr, "MTP forward: %v\n", err)
		os.Exit(2)
	}
	var absSum float32
	for _, v := range out {
		if v < 0 {
			absSum -= v
		} else {
			absSum += v
		}
	}
	report := Report{ModelDir: *dir, HiddenSize: meta.HiddenSize, MTPLayers: len(head.Layers), OutputLen: len(out), OutputAbsSum: absSum, Passed: len(out) == meta.HiddenSize && len(head.Layers) == meta.MTPNumHiddenLayers}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "json: %v\n", err)
		os.Exit(2)
	}
	if !report.Passed {
		os.Exit(1)
	}
}
