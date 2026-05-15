package gpu

import "testing"

func TestCheckedByteSize(t *testing.T) {
	if got, err := checkedByteSize(3, 12); err != nil || got != 12 {
		t.Fatalf("checkedByteSize(3,12) = %d, %v; want 12, nil", got, err)
	}
	if _, err := checkedByteSize(4, 12); err == nil {
		t.Fatal("checkedByteSize allowed copy beyond buffer capacity")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := checkedByteSize(maxInt/4+1, 0); err == nil {
		t.Fatal("checkedByteSize allowed int byte-size overflow")
	}
}
