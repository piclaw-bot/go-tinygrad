package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGeneratedTokenCount(t *testing.T) {
	cases := []struct {
		name      string
		output    []int
		promptLen int
		want      int
	}{
		{"normal", []int{1, 2, 3, 4}, 2, 2},
		{"short output", []int{1, 2}, 2, 0},
		{"negative prompt", []int{1, 2}, -1, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := generatedTokenCount(tc.output, tc.promptLen); got != tc.want {
				t.Fatalf("generatedTokenCount=%d want %d", got, tc.want)
			}
		})
	}
}

func TestTokensPerSecond(t *testing.T) {
	if got := tokensPerSecond(4, 2*time.Second); got != 2 {
		t.Fatalf("tokensPerSecond=%v want 2", got)
	}
	if got := tokensPerSecond(4, 0); got != 0 {
		t.Fatalf("zero elapsed tokensPerSecond=%v want 0", got)
	}
}

func TestSpecbenchLoadPrompts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")
	if err := os.WriteFile(path, []byte("# c\n\none\n two \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := loadPrompts("fallback", path)
	if err != nil {
		t.Fatalf("loadPrompts: %v", err)
	}
	want := []string{"one", "two"}
	if len(got) != len(want) {
		t.Fatalf("prompts=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prompt[%d]=%q want %q", i, got[i], want[i])
		}
	}
	got, err = loadPrompts("fallback", "")
	if err != nil || len(got) != 1 || got[0] != "fallback" {
		t.Fatalf("fallback prompts=%v err=%v", got, err)
	}
}

func TestSpecbenchSameIntsAndBaseName(t *testing.T) {
	if !sameInts([]int{1, 2}, []int{1, 2}) {
		t.Fatal("sameInts returned false for equal slices")
	}
	if sameInts([]int{1, 2}, []int{1, 3}) {
		t.Fatal("sameInts returned true for different slices")
	}
	if got := baseName("/tmp/models/foo/"); got != "foo" {
		t.Fatalf("baseName=%q want foo", got)
	}
}
