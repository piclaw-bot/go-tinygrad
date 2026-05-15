package model

import "testing"

func TestShouldUseCompactMLXLMHead(t *testing.T) {
	const mb = uint64(1024 * 1024)
	tests := []struct {
		name    string
		hasMLX  bool
		lmBytes uint64
		free    uint64
		want    bool
	}{
		{name: "no mlx uses f32 path", hasMLX: false, lmBytes: 2 * compactMLXLMHeadThresholdBytes, free: 8 * 1024 * mb, want: false},
		{name: "moderate head fits f32", hasMLX: true, lmBytes: 1208 * mb, free: 4 * 1024 * mb, want: false},
		{name: "large head prefers compact", hasMLX: true, lmBytes: 2180 * mb, free: 8 * 1024 * mb, want: true},
		{name: "moderate head falls back to compact when f32 lacks headroom", hasMLX: true, lmBytes: 1208 * mb, free: 1250 * mb, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldUseCompactMLXLMHead(tt.hasMLX, tt.lmBytes, tt.free)
			if got != tt.want {
				t.Fatalf("shouldUseCompactMLXLMHead(%v, %d, %d)=%v, want %v", tt.hasMLX, tt.lmBytes, tt.free, got, tt.want)
			}
		})
	}
}
