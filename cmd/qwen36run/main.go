package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model"
)

type Report struct {
	ModelDir     string  `json:"model_dir"`
	TokenID      int     `json:"token_id"`
	NextID       int     `json:"next_id"`
	Logit        float32 `json:"logit"`
	HiddenAbsSum float32 `json:"hidden_abs_sum"`
	Passed       bool    `json:"passed"`
}

type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
}

func main() {
	dir := flag.String("model", "", "Qwen3.6 model directory")
	token := flag.Int("token", 0, "single token id to run")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen36run -model <dir> [-token id]")
		os.Exit(2)
	}
	data, err := os.ReadFile(filepath.Join(*dir, "config.json"))
	check("config", err)
	meta, err := loaderconfig.ParseQwenNativeMTPMetadata(data)
	check("parse config", err)
	bundle, err := model.LoadQwen35NativeMTPBundleFromDir(*dir)
	check("load bundle", err)
	state, err := bundle.NewForwardState()
	check("state", err)
	src, err := model.OpenQwenNativeMTPSafetensorsSource(*dir)
	check("open tensors", err)
	defer src.Close()
	emb := mustRaw(src, "model.language_model.embed_tokens.weight")
	norm := mustRaw(src, "model.language_model.norm.weight")
	lm := mustRaw(src, "lm_head.weight")
	hidden := bf16Row(emb, *token)
	outs, nextState, err := bundle.ForwardBaseSequence([][]float32{hidden}, state, nil, 1e-6)
	check("base forward", err)
	_ = nextState
	h := outs[len(outs)-1]
	rmsNorm(h, bf16All(norm), 1e-6)
	id, val := argmaxBF16MatVec(lm, h)
	var sum float32
	for _, v := range h {
		if v < 0 {
			sum -= v
		} else {
			sum += v
		}
	}
	rep := Report{ModelDir: *dir, TokenID: *token, NextID: id, Logit: val, HiddenAbsSum: sum, Passed: id >= 0 && len(h) == meta.HiddenSize}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
	if !rep.Passed {
		os.Exit(1)
	}
}

func check(what string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(2)
	}
}
func mustRaw(src interface {
	GetRaw(string) ([]byte, string, []int, error)
}, name string) rawTensor { r, d, s, e := src.GetRaw(name); check(name, e); return rawTensor{r, d, s} }
func bf16(bits []byte, i int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(bits[i*2:])) << 16)
}
func bf16Row(t rawTensor, row int) []float32 {
	if t.dtype != "BF16" || len(t.shape) != 2 {
		panic("bad BF16 matrix")
	}
	cols := t.shape[1]
	out := make([]float32, cols)
	off := row * cols * 2
	for i := 0; i < cols; i++ {
		out[i] = bf16(t.raw[off:], i)
	}
	return out
}
func bf16All(t rawTensor) []float32 {
	if t.dtype != "BF16" {
		panic("bad BF16")
	}
	n := len(t.raw) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = bf16(t.raw, i)
	}
	return out
}
func rmsNorm(x, w []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		x[i] *= scale * w[i]
	}
}
func argmaxBF16MatVec(t rawTensor, x []float32) (int, float32) {
	rows, cols := t.shape[0], t.shape[1]
	best := -1
	bestv := float32(math.Inf(-1))
	for r := 0; r < rows; r++ {
		off := r * cols * 2
		var s float32
		for c := 0; c < cols; c++ {
			s += bf16(t.raw[off:], c) * x[c]
		}
		if s > bestv {
			bestv = s
			best = r
		}
	}
	return best, bestv
}
