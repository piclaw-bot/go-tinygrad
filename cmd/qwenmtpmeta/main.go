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
	Config                    loaderconfig.QwenNativeMTPMetadata        `json:"config"`
	LayerSummary              loaderconfig.QwenNativeMTPLayerSummary    `json:"layer_summary"`
	FullAttentionShapes       *loaderconfig.Qwen35FullAttentionShapes   `json:"full_attention_shapes,omitempty"`
	LinearAttentionShapes     *loaderconfig.Qwen35LinearAttentionShapes `json:"linear_attention_shapes,omitempty"`
	MTPTensors                []string                                  `json:"mtp_tensors,omitempty"`
	OptionalSharedHeadTensors []string                                  `json:"optional_shared_head_tensors,omitempty"`
	MissingMTPTensors         []string                                  `json:"missing_mtp_tensors,omitempty"`
	CanLoadSharedHead         bool                                      `json:"can_load_shared_head"`
	MTPTensorCount            int                                       `json:"mtp_tensor_count"`
	OptionalSharedHeadCount   int                                       `json:"optional_shared_head_count"`
	MissingMTPTensorCount     int                                       `json:"missing_mtp_tensor_count"`
	MTPTensorComplete         bool                                      `json:"mtp_tensor_complete"`
}

func main() {
	dir := flag.String("model", "", "model directory")
	strict := flag.Bool("strict", false, "exit non-zero if native MTP is configured but required MTP tensors are incomplete")
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
	report := Report{Config: meta, LayerSummary: meta.LayerSummary(), CanLoadSharedHead: meta.VocabSize > 0 && meta.HiddenSize > 0}
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
			if loaderconfig.IsOptionalQwenNativeMTPSharedHeadTensorName(name) {
				report.OptionalSharedHeadTensors = append(report.OptionalSharedHeadTensors, name)
			}
		}
		report.MissingMTPTensors = loaderconfig.MissingQwenNativeMTPTensors(report.MTPTensors, meta.MTPNumHiddenLayers)
	}
	report.MTPTensorCount = len(report.MTPTensors)
	report.OptionalSharedHeadCount = len(report.OptionalSharedHeadTensors)
	report.MissingMTPTensorCount = len(report.MissingMTPTensors)
	report.MTPTensorComplete = meta.MTPNumHiddenLayers > 0 && len(report.MissingMTPTensors) == 0 && len(report.MTPTensors) > 0
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "json: %v\n", err)
		os.Exit(2)
	}
	if shouldFailStrict(*strict, meta, report) {
		fmt.Fprintf(os.Stderr, "qwenmtpmeta: native MTP tensor set incomplete: missing=%d available=%d\n", report.MissingMTPTensorCount, report.MTPTensorCount)
		os.Exit(1)
	}
}

func shouldFailStrict(strict bool, meta loaderconfig.QwenNativeMTPMetadata, report Report) bool {
	return strict && meta.HasNativeMTP && !report.MTPTensorComplete
}

func safetensorNames(dir string) ([]string, error) {
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}
