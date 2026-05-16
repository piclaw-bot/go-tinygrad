package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFirstMismatch(t *testing.T) {
	cases := []struct {
		name string
		a, b []int
		want int
	}{
		{"same", []int{1, 2}, []int{1, 2}, -1},
		{"value", []int{1, 2}, []int{1, 3}, 1},
		{"short", []int{1, 2}, []int{1}, 1},
		{"long", []int{1}, []int{1, 2}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := firstMismatch(tc.a, tc.b); got != tc.want {
				t.Fatalf("firstMismatch=%d want %d", got, tc.want)
			}
		})
	}
}

func TestParseList(t *testing.T) {
	got := parseList(" prompt, repeat-last ,, none ")
	want := []string{"prompt", "repeat-last", "none"}
	if len(got) != len(want) {
		t.Fatalf("parseList len=%d want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("parseList[%d]=%q want %q", i, got[i], want[i])
		}
	}
}

func TestCheckReportSummaryFields(t *testing.T) {
	r := CheckReport{Model: "m", Passed: true, TotalChecks: 3, FailedChecks: 0}
	if r.TotalChecks != 3 || r.FailedChecks != 0 || !r.Passed {
		t.Fatalf("report=%+v", r)
	}
}

func TestGoldenRoundTripAndCompare(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "golden.json")
	golden := GoldenReport{Model: "m", Prompts: []GoldenPrompt{{PromptIndex: 0, PromptTokens: 2, MaxTokens: 3, Output: []int{1, 2, 3}}}}
	if err := writeGoldenFile(path, golden); err != nil {
		t.Fatalf("writeGoldenFile: %v", err)
	}
	loaded, err := loadGolden(path)
	if err != nil {
		t.Fatalf("loadGolden: %v", err)
	}
	if err := compareGoldenPrompt(loaded, 0, 2, 3, []int{1, 2, 3}); err != nil {
		t.Fatalf("compareGoldenPrompt match: %v", err)
	}
	if err := compareGoldenPrompt(loaded, 0, 2, 3, []int{1, 9, 3}); err == nil {
		t.Fatal("compareGoldenPrompt mismatch returned nil")
	}
	if err := compareGoldenPrompt(loaded, 0, 2, 4, []int{1, 2, 3}); err == nil {
		t.Fatal("compareGoldenPrompt metadata mismatch returned nil")
	}
}

func TestLoadPrompts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")
	if err := os.WriteFile(path, []byte("# comment\n\nhello\n world \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := loadPrompts("fallback", path)
	if err != nil {
		t.Fatalf("loadPrompts: %v", err)
	}
	want := []string{"hello", "world"}
	if len(got) != len(want) {
		t.Fatalf("prompts=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prompt[%d]=%q want %q", i, got[i], want[i])
		}
	}
}
