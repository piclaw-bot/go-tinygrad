package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
	"github.com/rcarmo/go-pherence/model"
)

// OpenAI API types

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature float32       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelListResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

// Server

type Server struct {
	cpuModel *model.LlamaModel
	gpuModel *model.GPUModel
	tok      *tokenizer.Tokenizer
	mu       sync.Mutex
	modelID  string
	maxCtx   int
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelListResponse{
		Object: "list",
		Data: []ModelObject{{
			ID:      s.modelID,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "local",
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	// Build prompt from messages
	var parts []string
	for _, msg := range req.Messages {
		parts = append(parts, msg.Content)
	}
	prompt := strings.Join(parts, "\n")

	s.mu.Lock()
	defer s.mu.Unlock()

	ids := s.tok.Encode(prompt)

	if req.Stream {
		s.streamResponse(w, r, ids, maxTokens)
	} else {
		s.nonStreamResponse(w, ids, maxTokens)
	}
}

func (s *Server) generate(ids []int, maxTokens int, emit func(token int, text string) bool) (int, string) {
	var output []int
	if s.gpuModel != nil {
		output = s.gpuModel.Generate(ids, maxTokens)
	} else {
		output = s.cpuModel.Generate(ids, maxTokens)
	}

	// Find generated tokens (after prompt)
	promptLen := len(ids)
	// The Generate function includes prompt in output for CPU, but not for GPU
	// Normalize: extract only the new tokens
	generated := output
	if len(output) > promptLen {
		// CPU path includes prompt
		generated = output[promptLen:]
	}

	var out strings.Builder
	count := 0
	for _, tok := range generated {
		if tok < 0 || tok >= len(s.tok.InvVocab) {
			break
		}
		text := s.tok.InvVocab[tok]
		// Stop on EOS-like tokens
		if text == "<eos>" || text == "</s>" || tok == 0 {
			break
		}
		out.WriteString(text)
		count++
		if emit != nil && !emit(tok, text) {
			break
		}
	}
	return count, out.String()
}

func (s *Server) nonStreamResponse(w http.ResponseWriter, ids []int, maxTokens int) {
	generated, text := s.generate(ids, maxTokens, nil)
	finishReason := "stop"
	if generated >= maxTokens {
		finishReason = "length"
	}

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: text},
			FinishReason: finishReason,
		}},
		Usage: Usage{
			PromptTokens:     len(ids),
			CompletionTokens: generated,
			TotalTokens:      len(ids) + generated,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, ids []int, maxTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	// Initial chunk with role
	writeSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Role: "assistant"}}},
	})

	generated := 0
	finishReason := "stop"

	s.generate(ids, maxTokens, func(tok int, text string) bool {
		select {
		case <-r.Context().Done():
			return false
		default:
		}
		writeSSE(w, flusher, StreamChunk{
			ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
			Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Content: text}}},
		})
		generated++
		return true
	})

	if generated >= maxTokens {
		finishReason = "length"
	}

	writeSSE(w, flusher, StreamChunk{
		ID: id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.modelID,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &finishReason}},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeSSE(w io.Writer, flusher http.Flusher, chunk StreamChunk) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func main() {
	dir := flag.String("model", "", "model directory")
	listen := flag.String("listen", ":8080", "address to listen on")
	useGPU := flag.Bool("gpu", false, "use GPU")
	gpuLayers := flag.Int("gpu-layers", 0, "number of layers on GPU (0=all)")
	turboQuant := flag.Bool("turbo-quant", false, "enable TurboQuant KV cache compression on CPU backend")
	eagerLoad := flag.Bool("eager-load", false, "pre-fault mmap'd model weights at startup")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: llmserver -model <dir> [-listen :8080] [-gpu]")
		os.Exit(1)
	}

	if *eagerLoad {
		os.Setenv("GO_PHERENCE_EAGER_LOAD", "1")
	}
	if *useGPU {
		model.ForceOnTheFly = true
		if *turboQuant {
			log.Printf("warning: --turbo-quant currently applies to the CPU backend only")
		}
	}

	log.Printf("Loading model from %s...", *dir)
	t0 := time.Now()
	m, err := model.LoadLlama(*dir)
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	m.EnableTurboQuant = *turboQuant
	tok, err := tokenizer.Load(*dir + "/tokenizer.json")
	if err != nil {
		log.Fatalf("Tokenizer failed: %v", err)
	}
	m.Tok = tok
	log.Printf("Model loaded in %.1fs (%d layers, h=%d)", time.Since(t0).Seconds(),
		m.Config.NumLayers, m.Config.HiddenSize)

	modelID := filepath.Base(*dir)
	srv := &Server{cpuModel: m, tok: tok, modelID: modelID, maxCtx: 4096}

	if *useGPU {
		g, err := model.LoadGPUModel(m)
		if err != nil {
			log.Printf("GPU failed: %v (using CPU)", err)
		} else {
			g.CPU.Tok = tok
			if *gpuLayers > 0 {
				g.GPULayers = *gpuLayers
			}
			srv.gpuModel = g
			defer g.Close()
			defer gpu.Shutdown()
			log.Printf("GPU model ready")
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)

	log.Printf("Listening on %s", *listen)
	log.Printf("  POST /v1/chat/completions")
	log.Printf("  GET  /v1/models")
	if err := http.ListenAndServe(*listen, mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
