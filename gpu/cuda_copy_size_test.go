package gpu

import "testing"

func TestCheckedByteSize(t *testing.T) {
	if got, err := checkedByteSize(3, 12); err != nil || got != 12 {
		t.Fatalf("checkedByteSize(3,12) = %d, %v; want 12, nil", got, err)
	}
	if _, err := checkedByteSize(4, 12); err == nil {
		t.Fatal("checkedByteSize allowed copy beyond buffer capacity")
	}
	if _, err := checkedByteSize(1, 0); err == nil {
		t.Fatal("checkedByteSize allowed non-empty copy into zero-byte buffer")
	}
	if got, err := checkedByteSize(1, -1); err != nil || got != 4 {
		t.Fatalf("checkedByteSize no-cap = %d, %v; want 4, nil", got, err)
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := checkedByteSize(maxInt/4+1, 0); err == nil {
		t.Fatal("checkedByteSize allowed int byte-size overflow")
	}
}
