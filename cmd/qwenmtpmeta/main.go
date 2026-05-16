package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type Report struct {
	Config                loaderconfig.QwenNativeMTPMetadata        `json:"config"`
	FullAttentionShapes   *loaderconfig.Qwen35FullAttentionShapes   `json:"full_attention_shapes,omitempty"`
	LinearAttentionShapes *loaderconfig.Qwen35LinearAttentionShapes `json:"linear_attention_shapes,omitempty"`
	MTPTensors            []string                                  `json:"mtp_tensors,omitempty"`
	MissingMTPTensors     []string                                  `json:"missing_mtp_tensors,omitempty"`
}

func main() {
	dir := flag.String("model", "", "model directory")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwenmtpmeta -model <dir>")
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
	report := Report{Config: meta}
	if shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim); err == nil {
		report.FullAttentionShapes = &shapes
	}
	if shapes, err := loaderconfig.Qwen35LinearAttentionShapesFor(meta.HiddenSize, meta.LinearValueHeadDim*meta.LinearNumValueHeads, meta.LinearKeyHeadDim, meta.LinearConvKernelDim, meta.LinearNumValueHeads, meta.LinearNumKeyHeads); err == nil {
		report.LinearAttentionShapes = &shapes
	}
	if names, err := safetensorNames(*dir); err == nil {
		for _, name := range names {
			if loaderconfig.IsQwenNativeMTPTensorName(name) {
				report.MTPTensors = append(report.MTPTensors, name)
			}
		}
		report.MissingMTPTensors = loaderconfig.MissingQwenNativeMTPTensors(report.MTPTensors, meta.MTPNumHiddenLayers)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "json: %v\n", err)
		os.Exit(2)
	}
}

func safetensorNames(dir string) ([]string, error) {
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}
