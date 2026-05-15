package gpu

import "testing"

func TestStatsSnapshotDoesNotEnableCounters(t *testing.T) {
	prev := SetStatsEnabled(false)
	defer SetStatsEnabled(prev)
	_ = StatsSnapshot()
	if got := SetStatsEnabled(false); got {
		t.Fatal("StatsSnapshot enabled counters")
	}
}

func TestSetStatsEnabledReturnsPreviousState(t *testing.T) {
	prev := SetStatsEnabled(false)
	defer SetStatsEnabled(prev)
	if old := SetStatsEnabled(true); old {
		t.Fatal("SetStatsEnabled(true) old=true, want false")
	}
	if old := SetStatsEnabled(false); !old {
		t.Fatal("SetStatsEnabled(false) old=false, want true")
	}
}
