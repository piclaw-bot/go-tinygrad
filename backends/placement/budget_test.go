package placement

import "testing"

func TestBudgetManagerBasic(t *testing.T) {
	b := NewBudgetManager(100, 200, 50, 0)

	// Check initial state
	if b.Available(BudgetResident) != 100*1024*1024 {
		t.Fatalf("expected 100MB resident, got %d", b.Available(BudgetResident))
	}

	// Alloc within budget
	if !b.Alloc(BudgetResident, 50*1024*1024) {
		t.Fatal("should fit in budget")
	}
	if b.Available(BudgetResident) != 50*1024*1024 {
		t.Fatalf("expected 50MB remaining, got %d", b.Available(BudgetResident))
	}

	// Alloc that exceeds budget
	if b.Alloc(BudgetResident, 60*1024*1024) {
		t.Fatal("should not fit in budget")
	}

	// Free and re-alloc
	b.Free(BudgetResident, 50*1024*1024)
	if !b.Alloc(BudgetResident, 60*1024*1024) {
		t.Fatal("should fit after free")
	}

	// Hit/evict counters
	b.Hit(BudgetLayer)
	b.Hit(BudgetLayer)
	b.Evict(BudgetLayer)
	if b.LayerHits.Load() != 2 || b.LayerEvicts.Load() != 1 {
		t.Fatalf("expected 2 hits / 1 evict, got %d / %d", b.LayerHits.Load(), b.LayerEvicts.Load())
	}

	// Report
	report := b.Report()
	if len(report) == 0 {
		t.Fatal("empty report")
	}
	t.Log(report)
}

func TestBudgetManagerUnlimited(t *testing.T) {
	b := NewBudgetManager(0, 0, 0, 0) // all unlimited

	// Should always succeed
	if !b.Alloc(BudgetResident, 10*1024*1024*1024) {
		t.Fatal("unlimited budget should always succeed")
	}
	if !b.Alloc(BudgetLayer, 10*1024*1024*1024) {
		t.Fatal("unlimited budget should always succeed")
	}
}

func TestBudgetManagerRejectsNegativeAccounting(t *testing.T) {
	var nilBudget *BudgetManager
	if nilBudget.Alloc(BudgetResident, 1) || nilBudget.Available(BudgetResident) != 0 || nilBudget.Report() == "" {
		t.Fatal("nil budget manager should reject alloc and report safely")
	}
	nilBudget.Free(BudgetResident, 1)
	nilBudget.Hit(BudgetLayer)
	nilBudget.Evict(BudgetLayer)

	b := NewBudgetManager(100, -1, 0, 0)
	if b.LayerBudget != 0 {
		t.Fatalf("negative budget should clamp to unlimited/zero, got %d", b.LayerBudget)
	}
	if b.Alloc(BudgetResident, -1) || b.Alloc(BudgetCategory(99), 1) || b.Available(BudgetCategory(99)) != 0 {
		t.Fatal("invalid allocation/accounting inputs should be rejected")
	}
	if !b.Alloc(BudgetResident, 10) {
		t.Fatal("positive allocation should work")
	}
	b.Free(BudgetResident, -100)
	b.Free(BudgetCategory(99), 100)
	if b.ResidentUsed != 10 {
		t.Fatalf("invalid free changed usage to %d", b.ResidentUsed)
	}
	b.ResidentUsed = int64(^uint64(0) >> 1)
	if b.Alloc(BudgetResident, 1) {
		t.Fatal("allocation should reject usage overflow")
	}
}

func TestAutoBudgetManagerClampsInputs(t *testing.T) {
	b := NewAutoBudgetManager(^uint64(0), ^uint64(0), 512, 128, 256)
	if b.ResidentBudget <= 0 || b.LayerBudget <= 0 || b.StreamBudget <= 0 || b.ExpertBudget <= 0 {
		t.Fatalf("expected positive budgets from huge input, got %+v", b)
	}

	noGPU := NewAutoBudgetManager(0, 0, 512, 128, 256)
	if noGPU.StreamBudget != 128*1024*1024 || noGPU.ResidentBudget != 0 || noGPU.LayerBudget != 0 || noGPU.ExpertBudget != 0 {
		t.Fatalf("unexpected no-GPU budgets: %+v", noGPU)
	}
}

func TestBudgetManagerReport(t *testing.T) {
	b := NewBudgetManager(500, 3000, 256, 1024)
	b.Alloc(BudgetResident, 487*1024*1024)
	b.Alloc(BudgetLayer, 2800*1024*1024)
	b.Hit(BudgetExpert)
	b.Hit(BudgetExpert)
	b.Hit(BudgetExpert)
	b.Evict(BudgetExpert)
	report := b.Report()
	t.Log(report)
}
