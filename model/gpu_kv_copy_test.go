package model

import "testing"

func TestKVCopyByteRange(t *testing.T) {
	bytes, off, ok := kvCopyByteRange(3, 8, 4*8*4)
	if !ok || bytes != 32 || uint64(off) != 96 {
		t.Fatalf("kvCopyByteRange = bytes %d off %d ok %v, want 32 96 true", bytes, off, ok)
	}
	if _, _, ok := kvCopyByteRange(4, 8, 4*8*4); ok {
		t.Fatal("kvCopyByteRange allowed copy past capacity")
	}
	if _, _, ok := kvCopyByteRange(0, 8, 31); ok {
		t.Fatal("kvCopyByteRange allowed destination smaller than one KV row")
	}
	maxInt := int(^uint(0) >> 1)
	if _, _, ok := kvCopyByteRange(maxInt, 8, maxInt); ok {
		t.Fatal("kvCopyByteRange allowed overflowing position arithmetic")
	}
}
