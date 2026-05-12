package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// Tokenizer handles BPE tokenization for LLaMA-style models.
type Tokenizer struct {
	Vocab    map[string]int // token string → ID
	InvVocab map[int]string // ID → token string
	Merges   [][2]string    // BPE merge pairs in priority order
}

// Load loads a HuggingFace tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var raw struct {
		Model struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	if raw.Model.Vocab == nil {
		raw.Model.Vocab = map[string]int{}
	}
	t := &Tokenizer{
		Vocab:    raw.Model.Vocab,
		InvVocab: make(map[int]string, len(raw.Model.Vocab)),
	}
	for k, v := range raw.Model.Vocab {
		t.InvVocab[v] = k
	}
	// Add special/added tokens
	for _, at := range raw.AddedTokens {
		if _, exists := t.Vocab[at.Content]; !exists {
			t.Vocab[at.Content] = at.ID
		}
		t.InvVocab[at.ID] = at.Content
	}

	if len(raw.Model.Merges) == 0 || string(raw.Model.Merges) == "null" {
		return t, nil
	}

	// Merges can be ["a b", ...] (strings) or [["a","b"], ...] (arrays)
	var mergeStrings []string
	if err := json.Unmarshal(raw.Model.Merges, &mergeStrings); err == nil {
		t.Merges = make([][2]string, len(mergeStrings))
		for i, m := range mergeStrings {
			parts := strings.SplitN(m, " ", 2)
			if len(parts) == 2 {
				t.Merges[i] = [2]string{parts[0], parts[1]}
			}
		}
	} else {
		var mergeArrays [][2]string
		if err := json.Unmarshal(raw.Model.Merges, &mergeArrays); err == nil {
			t.Merges = mergeArrays
		} else {
			return nil, fmt.Errorf("unsupported merges format")
		}
	}

	return t, nil
}

// Encode tokenizes a string into token IDs.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || t.Vocab == nil {
		return nil
	}
	// Pre-tokenize: split on word boundaries, add space prefix
	// Auto-detect: Ġ (U+0120, GPT-2/Qwen) or ▁ (U+2581, SentencePiece/Gemma)
	spacePrefix := "\u0120" // GPT-2 default
	if _, ok := t.Vocab["\u2581the"]; ok {
		spacePrefix = "\u2581" // SentencePiece
	}
	words := strings.Fields(text)
	var pieces []string
	for i, w := range words {
		if i > 0 {
			w = spacePrefix + w
		}
		pieces = append(pieces, w)
	}

	// For each piece, try direct vocab lookup first, then BPE
	mergeRank := make(map[[2]string]int, len(t.Merges))
	for i, m := range t.Merges {
		mergeRank[m] = i
	}

	var ids []int
	for _, piece := range pieces {
		// Direct lookup
		if id, ok := t.Vocab[piece]; ok {
			ids = append(ids, id)
			continue
		}

		// BPE: split into characters
		chars := make([]string, 0, len(piece))
		for _, r := range piece {
			chars = append(chars, string(r))
		}

		// Apply BPE merges
		for len(chars) >= 2 {
			bestRank := len(t.Merges)
			bestIdx := -1
			for i := 0; i < len(chars)-1; i++ {
				pair := [2]string{chars[i], chars[i+1]}
				if rank, ok := mergeRank[pair]; ok && rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
			if bestIdx < 0 {
				break
			}
			merged := chars[bestIdx] + chars[bestIdx+1]
			newChars := make([]string, 0, len(chars)-1)
			newChars = append(newChars, chars[:bestIdx]...)
			newChars = append(newChars, merged)
			newChars = append(newChars, chars[bestIdx+2:]...)
			chars = newChars
		}

		for _, ch := range chars {
			if id, ok := t.Vocab[ch]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	if t == nil || t.InvVocab == nil {
		return ""
	}
	var parts []string
	for _, id := range ids {
		if tok, ok := t.InvVocab[id]; ok {
			parts = append(parts, tok)
		}
	}
	text := strings.Join(parts, "")
	// Replace SentencePiece space marker with actual space
	text = strings.ReplaceAll(text, "\u2581", " ")
	// Reverse byte-level BPE encoding
	byteDecoder := getByteDecoder()
	var decoded []byte
	for _, r := range text {
		if b, ok := byteDecoder[r]; ok {
			decoded = append(decoded, b)
		} else {
			decoded = append(decoded, string(r)...)
		}
	}
	text = string(decoded)
	return text
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	if t == nil || t.Vocab == nil {
		return 0
	}
	return len(t.Vocab)
}

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

var (
	_byteEncoder     map[byte]rune
	_byteEncoderOnce sync.Once
)

func getByteEncoder() map[byte]rune {
	_byteEncoderOnce.Do(func() {
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
	})
	return _byteEncoder
}

var (
	_byteDecoder     map[rune]byte
	_byteDecoderOnce sync.Once
)

func getByteDecoder() map[rune]byte {
	_byteDecoderOnce.Do(func() {
		enc := getByteEncoder()
		_byteDecoder = make(map[rune]byte, len(enc))
		for b, r := range enc {
			_byteDecoder[r] = b
		}
	})
	return _byteDecoder
}
