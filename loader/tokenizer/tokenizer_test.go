package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadTokenizerMissingVocabDoesNotPanic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	json := `{"model":{"merges":null},"added_tokens":[{"id":7,"content":"<x>"}]}`
	if err := os.WriteFile(path, []byte(json), 0644); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok.Vocab["<x>"] != 7 || tok.InvVocab[7] != "<x>" {
		t.Fatalf("added token not loaded: vocab=%v inv=%v", tok.Vocab, tok.InvVocab)
	}
}

func TestLoadTokenizerRejectsMalformedMerges(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"string": `{"model":{"vocab":{"a":1,"b":2},"merges":["a b","broken"]}}`,
		"array":  `{"model":{"vocab":{"a":1,"b":2},"merges":[["a","b"],["", "c"]]}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, name+".json")
			if err := os.WriteFile(path, []byte(body), 0644); err != nil {
				t.Fatalf("write tokenizer: %v", err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("Load accepted malformed merges")
			}
		})
	}
}

func TestTokenizerNilSafe(t *testing.T) {
	var tok *Tokenizer
	if got := tok.Encode("hello"); got != nil {
		t.Fatalf("nil Encode=%v, want nil", got)
	}
	if got := tok.Decode([]int{1}); got != "" {
		t.Fatalf("nil Decode=%q, want empty", got)
	}
}

func TestDecodePreservesUnknownUnicodeRunes(t *testing.T) {
	tok := &Tokenizer{InvVocab: map[int]string{1: "☃"}}
	if got := tok.Decode([]int{1}); got != "☃" {
		t.Fatalf("Decode unicode=%q, want snowman", got)
	}
}

func TestTokenizerNilVocabSizeAndByteMaps(t *testing.T) {
	var tok *Tokenizer
	if got := tok.VocabSize(); got != 0 {
		t.Fatalf("nil tokenizer VocabSize=%d, want 0", got)
	}
	enc := getByteEncoder()
	dec := getByteDecoder()
	if len(enc) != 256 || len(dec) != 256 {
		t.Fatalf("byte maps sizes enc=%d dec=%d, want 256", len(enc), len(dec))
	}
	for b := 0; b < 256; b++ {
		r := enc[byte(b)]
		if got := dec[r]; got != byte(b) {
			t.Fatalf("byte map roundtrip %d -> %U -> %d", b, r, got)
		}
	}
}
