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
