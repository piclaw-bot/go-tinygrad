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
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

type Report struct {
	ModelDir     string  `json:"model_dir"`
	Prompt       string  `json:"prompt,omitempty"`
	InputIDs     []int   `json:"input_ids"`
	GeneratedIDs []int   `json:"generated_ids,omitempty"`
	Decoded      string  `json:"decoded,omitempty"`
	TokenID      int     `json:"token_id,omitempty"`
	NextID       int     `json:"next_id"`
	Logit        float32 `json:"logit"`
	HiddenAbsSum float32 `json:"hidden_abs_sum"`
	MTPOutputLen int     `json:"mtp_output_len,omitempty"`
	MTPAbsSum    float32 `json:"mtp_abs_sum,omitempty"`
	Passed       bool    `json:"passed"`
}

type rawTensor struct {
	raw   []byte
	dtype string
	shape []int
}

type runner struct {
	bundle *model.Qwen35NativeMTPBundle
	state  model.Qwen35BaseForwardState
	emb    rawTensor
	normW  []float32
	lm     rawTensor
}

func main() {
	dir := flag.String("model", "", "Qwen3.6 model directory")
	token := flag.Int("token", 0, "single token id to run when -prompt is empty")
	prompt := flag.String("prompt", "", "text prompt to encode and run")
	steps := flag.Int("steps", 1, "greedy decode steps after prompt/token")
	mtp := flag.Bool("mtp", false, "also run native MTP head from last base hidden state and generated token")
	flag.Parse()
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen36run -model <dir> [-token id | -prompt text] [-steps n]")
		os.Exit(2)
	}
	if *steps < 1 {
		fmt.Fprintln(os.Stderr, "steps must be >= 1")
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
	r := runner{bundle: bundle, state: state, emb: mustRaw(src, "model.language_model.embed_tokens.weight"), normW: bf16All(mustRaw(src, "model.language_model.norm.weight")), lm: mustRaw(src, "lm_head.weight")}
	inputIDs := []int{*token}
	var tok *tokenizer.Tokenizer
	if *prompt != "" {
		tok, err = tokenizer.Load(filepath.Join(*dir, "tokenizer.json"))
		check("tokenizer", err)
		inputIDs = tok.Encode(*prompt)
		if len(inputIDs) == 0 {
			fmt.Fprintln(os.Stderr, "prompt encoded to zero tokens")
			os.Exit(2)
		}
	}
	var next int
	var logit float32
	var h []float32
	var preNormHidden []float32
	for _, id := range inputIDs {
		next, logit, h, preNormHidden, err = r.step(id)
		check("prefill", err)
	}
	generated := make([]int, 0, *steps)
	cur := next
	for i := 0; i < *steps; i++ {
		generated = append(generated, cur)
		next, logit, h, preNormHidden, err = r.step(cur)
		check("decode", err)
		cur = next
	}
	var sum float32
	for _, v := range h {
		if v < 0 {
			sum -= v
		} else {
			sum += v
		}
	}
	decoded := ""
	if tok != nil {
		decoded = tok.Decode(generated)
	}
	rep := Report{ModelDir: *dir, Prompt: *prompt, InputIDs: inputIDs, GeneratedIDs: generated, Decoded: decoded, TokenID: inputIDs[len(inputIDs)-1], NextID: next, Logit: logit, HiddenAbsSum: sum, Passed: next >= 0 && len(h) == meta.HiddenSize}
	if *mtp {
		mtpHead, err := model.LoadQwenNativeMTPHeadFromSafetensorsDir(*dir, meta)
		check("load MTP head", err)
		mtpEmbedding := bf16Row(r.emb, generated[len(generated)-1])
		mtpOut, err := mtpHead.ForwardOne(mtpEmbedding, preNormHidden, r.state.Pos-1, nil, 1e-6, meta)
		check("MTP forward", err)
		rep.MTPOutputLen = len(mtpOut)
		for _, v := range mtpOut {
			if v < 0 {
				rep.MTPAbsSum -= v
			} else {
				rep.MTPAbsSum += v
			}
		}
		rep.Passed = rep.Passed && rep.MTPOutputLen == meta.HiddenSize
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
	if !rep.Passed {
		os.Exit(1)
	}
}

func (r *runner) step(tokenID int) (int, float32, []float32, []float32, error) {
	hidden := bf16Row(r.emb, tokenID)
	outs, nextState, err := r.bundle.ForwardBaseSequence([][]float32{hidden}, r.state, nil, 1e-6)
	if err != nil {
		return 0, 0, nil, nil, err
	}
	r.state = nextState
	preNorm := append([]float32(nil), outs[len(outs)-1]...)
	h := append([]float32(nil), preNorm...)
	rmsNorm(h, r.normW, 1e-6)
	id, val := argmaxBF16MatVec(r.lm, h)
	return id, val, h, preNorm, nil
}

func check(what string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", what, err)
		os.Exit(2)
	}
}
func mustRaw(src interface {
	GetRaw(string) ([]byte, string, []int, error)
}, name string) rawTensor {
	r, d, s, e := src.GetRaw(name)
	check(name, e)
	return rawTensor{r, d, s}
}
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
