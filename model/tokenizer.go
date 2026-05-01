package model

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
)

// Tokenizer handles BPE tokenization for LLaMA-style models.
type Tokenizer struct {
	Vocab    map[string]int // token string → ID
	InvVocab map[int]string // ID → token string
	Merges   [][2]string    // BPE merge pairs in priority order
}

// LoadTokenizer loads a HuggingFace tokenizer.json.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var raw struct {
		Model struct {
			Vocab  map[string]int `json:"vocab"`
			Merges []string       `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	t := &Tokenizer{
		Vocab:    raw.Model.Vocab,
		InvVocab: make(map[int]string, len(raw.Model.Vocab)),
	}
	for k, v := range raw.Model.Vocab {
		t.InvVocab[v] = k
	}

	t.Merges = make([][2]string, len(raw.Model.Merges))
	for i, m := range raw.Model.Merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) == 2 {
			t.Merges[i] = [2]string{parts[0], parts[1]}
		}
	}

	return t, nil
}

// Encode tokenizes a string into token IDs.
func (t *Tokenizer) Encode(text string) []int {
	// Pre-tokenize: split on whitespace, prefix space tokens with Ġ
	words := strings.Fields(text)
	var tokens []string
	for i, w := range words {
		if i > 0 {
			w = "Ġ" + w // GPT-2 style space prefix
		}
		// Split into individual bytes as initial tokens
		for _, b := range []byte(w) {
			ch := byteToToken(b)
			tokens = append(tokens, ch)
		}
	}

	// Apply BPE merges
	mergeRank := make(map[[2]string]int, len(t.Merges))
	for i, m := range t.Merges {
		mergeRank[m] = i
	}

	for {
		if len(tokens) < 2 {
			break
		}
		// Find best merge
		bestRank := len(t.Merges)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			pair := [2]string{tokens[i], tokens[i+1]}
			if rank, ok := mergeRank[pair]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		// Apply merge
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}

	// Look up IDs
	ids := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		if id, ok := t.Vocab[tok]; ok {
			ids = append(ids, id)
		}
		// Skip unknown tokens silently
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if tok, ok := t.InvVocab[id]; ok {
			parts = append(parts, tok)
		}
	}
	text := strings.Join(parts, "")
	// Reverse byte-level BPE encoding
	byteDecoder := getByteDecoder()
	var decoded []byte
	for _, r := range text {
		if b, ok := byteDecoder[r]; ok {
			decoded = append(decoded, b)
		} else {
			decoded = append(decoded, byte(r))
		}
	}
	text = string(decoded)
	return text
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.Vocab) }

// byteToToken converts a byte to its BPE token representation.
func byteToToken(b byte) string {
	// GPT-2 byte-level BPE: bytes 0-255 map to Unicode characters
	// See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
	byteEncoder := getByteEncoder()
	if r, ok := byteEncoder[b]; ok {
		return string(r)
	}
	return string(rune(b))
}

var _byteEncoder map[byte]rune

func getByteEncoder() map[byte]rune {
	if _byteEncoder != nil {
		return _byteEncoder
	}
	_byteEncoder = make(map[byte]rune)
	// Standard visible ASCII + Latin-1 supplement
	n := 0
	bs := make([]int, 0, 256)
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
	}
	sort.Ints(bs)
	bsSet := map[int]bool{}
	for _, b := range bs {
		bsSet[b] = true
		_byteEncoder[byte(b)] = rune(b)
	}
	n = 256
	for i := 0; i < 256; i++ {
		if !bsSet[i] {
			_byteEncoder[byte(i)] = rune(n)
			n++
		}
	}
	return _byteEncoder
}

var _byteDecoder map[rune]byte

func getByteDecoder() map[rune]byte {
	if _byteDecoder != nil {
		return _byteDecoder
	}
	enc := getByteEncoder()
	_byteDecoder = make(map[rune]byte, len(enc))
	for b, r := range enc {
		_byteDecoder[r] = b
	}
	return _byteDecoder
}
