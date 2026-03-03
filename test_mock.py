import os
import time
import tempfile
from predictor import Predictor


def test_scale_event_logging():
    """log_scale_event() writes to scale_events; get_performance_summary() aggregates correctly."""
    import storage as st

    # Use a temporary DB so this test is isolated from production data
    with tempfile.TemporaryDirectory() as tmp:
        original_db = st.DATABASE_PATH if hasattr(st, "DATABASE_PATH") else None
        import config
        original_config_db = config.DATABASE_PATH
        config.DATABASE_PATH = os.path.join(tmp, "test.db")

        # Patch storage to use the temp DB
        st.DATABASE_PATH = config.DATABASE_PATH
        def _tmp_conn():
            import sqlite3 as _sq
            c = _sq.connect(config.DATABASE_PATH)
            c.row_factory = _sq.Row
            return c
        _orig_conn = st.get_db_connection
        st.get_db_connection = _tmp_conn

        st.init_db()

        now = time.time()

        # Log a scale_down event (RAM freed: 2048 → 1024, -1024 MB)
        st.log_scale_event(
            entity_id="200",
            entity_type="LXC",
            cpus_before=4, cpus_after=4,
            ram_before_mb=2048.0, ram_after_mb=1024.0,
            trigger="prediction",
        )

        # Log a scale_up event (+512 MB)
        st.log_scale_event(
            entity_id="201",
            entity_type="LXC",
            cpus_before=2, cpus_after=2,
            ram_before_mb=512.0, ram_after_mb=1024.0,
            trigger="prediction",
        )

        # Log a no-change event — should NOT be recorded
        st.log_scale_event(
            entity_id="202",
            entity_type="LXC",
            cpus_before=2, cpus_after=2,
            ram_before_mb=1024.0, ram_after_mb=1024.0,
            trigger="prediction",
        )

        summary = st.get_performance_summary(days=1)
        ev = summary["scale_events"]

        print("\nTesting Scale Event Logging:")
        print(f"  Total events   : {ev['total']}")
        print(f"  Scale-ups      : {ev['scale_up_count']}")
        print(f"  Scale-downs    : {ev['scale_down_count']}")
        print(f"  Net RAM freed  : {ev['net_ram_freed_mb']} MB")

        assert ev["total"] == 2, f"Expected 2 events (no-change skipped), got {ev['total']}"
        assert ev["scale_down_count"] == 1, "Expected 1 scale_down"
        assert ev["scale_up_count"] == 1, "Expected 1 scale_up"
        # net_ram_freed = -((-1024) + 512) = 512 MB freed overall
        assert ev["net_ram_freed_mb"] == 512.0, (
            f"Expected net 512 MB freed, got {ev['net_ram_freed_mb']}"
        )

        # Restore
        st.get_db_connection = _orig_conn
        st.DATABASE_PATH = original_db
        config.DATABASE_PATH = original_config_db



def test_predictor():
    predictor = Predictor(prediction_horizon=2)

    # Simulate a rising trend
    # format: [{'time': timestamp, 'cpu': ratio, 'mem': bytes, 'diskread': bps, ...}]
    base_time = time.time() - 300
    metrics = [
        {"time": base_time,       "cpu": 0.10, "mem": 512  * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 60,  "cpu": 0.20, "mem": 600  * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 120, "cpu": 0.30, "mem": 750  * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 180, "cpu": 0.50, "mem": 900  * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 240, "cpu": 0.75, "mem": 1024 * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
    ]

    # We pass entity_id="100", rrd_data=metrics, entity_type="LXC"
    predictions = predictor.predict_next_usage("100", metrics, "LXC")
    print("Testing Rising Trend Prediction (Fallback Logic):")
    print("Metrics: CPU rising 10 -> 75, RAM rising 512 -> 1024")
    print(f"Predicted Output: {predictions}")

    # Assertions - XGBoost fallback logic currently returns the latest valid metric when <15 elements
    assert (
        predictions["cpu_percent"] == 75.0
    ), "CPU Prediction fallback should match the latest metric"
    assert (
        predictions["ram_usage_mb"] == 1024.0
    ), "RAM Prediction fallback should match the latest metric"

    # Simulate a falling trend
    metrics_falling = [
        {"time": base_time,       "cpu": 0.90, "mem": 2048 * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 60,  "cpu": 0.80, "mem": 1900 * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 120, "cpu": 0.50, "mem": 1500 * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 180, "cpu": 0.30, "mem": 1024 * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
        {"time": base_time + 240, "cpu": 0.15, "mem": 512  * 1024 * 1024, "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0},
    ]

    predictions_fallback = predictor.predict_next_usage("100", metrics_falling, "LXC")
    print("\nTesting Falling Trend Prediction (Fallback Logic):")
    print("Metrics: CPU falling 90 -> 15, RAM falling 2048 -> 512")
    print(f"Predicted Output: {predictions_fallback}")

    assert (
        predictions_fallback["cpu_percent"] == 15.0
    ), "CPU Prediction fallback should match the latest metric"
    assert (
        predictions_fallback["ram_usage_mb"] == 512.0
    ), "RAM Prediction fallback should match the latest metric"


def test_predictor_new_metric_keys():
    """Predictor output must expose all four I/O peak keys so callers can log/display them."""
    predictor = Predictor(prediction_horizon=2)

    base_time = time.time() - 300
    # 5 points -> hits the <15 fallback path; verifies the keys are included even there
    metrics = [
        {
            "time": base_time + i * 60,
            "cpu": 0.10,
            "mem": 512 * 1024 * 1024,
            "diskread": 1024 * 1024 * (i + 1),   # rising disk read
            "diskwrite": 512 * 1024 * (i + 1),
            "netin": 100_000 * (i + 1),
            "netout": 50_000 * (i + 1),
        }
        for i in range(5)
    ]

    result = predictor.predict_next_usage("999", metrics, "LXC",
                                          host_context={"cpu_percent": 45.0, "ram_percent": 72.0, "swap_percent": 5.0})

    print("\nTesting New I/O Peak Keys Present in Predictor Output:")
    print(f"Predicted Output: {result}")

    for key in (
        "recent_peak_disk_read",
        "recent_peak_disk_write",
        "recent_peak_net_in",
        "recent_peak_net_out",
    ):
        assert key in result, f"Expected key '{key}' in predictor output"
        assert result[key] > 0, f"Expected non-zero peak for '{key}' given rising I/O fixture"


def test_predictor_delta_features():
    """Rate-of-change deltas must be non-zero for a rising-trend fixture
    when using >= 15 intervals (exercises the XGBoost code path, but verifies
    the context vector is assembled without error even with no model file)."""
    predictor = Predictor(prediction_horizon=2)
    base_time = time.time() - 900
    # 15 points: CPU rises 10 -> 75, RAM rises 512 -> 1024
    metrics = [
        {
            "time": base_time + i * 60,
            "cpu": 0.10 + i * 0.04,   # rises from 10% to 70%
            "mem": (512 + i * 35) * 1024 * 1024,  # rises from 512 to 1022 MB
            "diskread": 1_000_000 * (i + 1),
            "diskwrite": 500_000 * (i + 1),
            "netin": 200_000 * (i + 1),
            "netout": 100_000 * (i + 1),
        }
        for i in range(15)
    ]
    rich_context = {
        "cpu_percent": 45.0, "ram_percent": 72.0, "swap_percent": 2.0,
        "load_avg_1m": 3.2, "load_avg_5m": 2.8, "ksm_sharing_mb": 1024.0,
        "cpu_overcommit_ratio": 1.8, "ram_overcommit_ratio": 0.85, "container_count": 12.0,
    }
    result = predictor.predict_next_usage("998", metrics, "LXC", host_context=rich_context)

    print("\nTesting Delta Feature Computation (rising 15-point fixture):")
    print(f"Predicted Output: {result}")

    # Peaks should reflect the max point in the window
    assert result["recent_peak_cpu"] > 50.0, "CPU peak should be near the max of the rising trend"
    assert result["recent_peak_disk_read"] > 0.0, "disk_read peak should be positive"


def test_predictor_host_context_full():
    """All new host_context fields (load, ksm, overcommit) should pass through
    without raising any errors and appear in the fallback result."""
    predictor = Predictor(prediction_horizon=2)
    base_time = time.time() - 300
    metrics = [
        {"time": base_time + i * 60, "cpu": 0.20, "mem": 512 * 1024 * 1024,
         "diskread": 0, "diskwrite": 0, "netin": 0, "netout": 0}
        for i in range(5)  # < 15 → uses fallback path
    ]
    full_context = {
        "cpu_percent": 60.0, "ram_percent": 80.0, "swap_percent": 10.0,
        "load_avg_1m": 4.5, "load_avg_5m": 3.9, "ksm_sharing_mb": 2048.0,
        "cpu_overcommit_ratio": 2.0, "ram_overcommit_ratio": 1.1, "container_count": 20.0,
    }
    result = predictor.predict_next_usage("997", metrics, "LXC", host_context=full_context)

    print("\nTesting Full Host Context Pass-Through (no model, fallback path):")
    print(f"Predicted Output: {result}")

    assert result is not None, "Should return a valid fallback result"
    assert result["cpu_percent"] == 20.0, "Fallback CPU should match latest metric"


def test_scaler():
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            # Simulate host node under extreme load, triggering the 95% safety cap
            return {"cpu_percent": 96.0, "ram_percent": 98.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):  # pylint: disable=unused-argument
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    baseline = {"min_cpus": 1, "max_cpus": 8, "min_ram_mb": 512, "max_ram_mb": 8192}
    predicted = {"cpu_percent": 95.0, "ram_usage_mb": 4096.0}
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 1800.0,  # 88% of allocation — above 50% idle threshold so active scale-down doesn't fire
        "cpu_percent": 80.0,
        "allocated_swap_mb": 256.0,
    }

    scaler.evaluate_and_scale("100", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Scaler Safety Cap (Host 96% CPU / 98% RAM):")
    print(f"Update Requested: {px.last_update}")

    # It should refuse to scale up CPU beyond current (2) and refuse to scale RAM beyond current (2048)
    # even though predictions want 3 cores and 4096 * 1.2 RAM.
    # Since both are capped to their current states, the Scaler will deliberately NOT make an API call to Proxmox.
    assert (
        px.last_update is None
    ), "Scaler should abort the API call since all requested scale-ups were denied by the safety cap!"


def test_scaler_uses_peak_ram():
    """Scaler must use recent_peak_ram (not the lower predicted average) as
    the allocation basis to prevent swap under bursty workloads."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            return {"cpu_percent": 10.0, "ram_percent": 30.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):  # pylint: disable=unused-argument
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # ML forecast says 512 MB, but the observed peak was 900 MB
    baseline = {"min_cpus": 1, "max_cpus": 8, "min_ram_mb": 512, "max_ram_mb": 8192}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 512.0,
        "recent_peak_cpu": 15.0,
        "recent_peak_ram": 900.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 600.0,
        "cpu_percent": 10.0,
    }

    scaler.evaluate_and_scale("200", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Scaler Uses Peak RAM (Peak 900 MB > Predicted 512 MB):")
    print(f"Update Requested: {px.last_update}")

    # desired = 900 * 1.30 = 1170 MB -> capped at max_ram_mb=8192, so expect 1170
    assert px.last_update is not None, "Scaler should have issued an update"
    assert px.last_update["ram_mb"] >= 1100, (
        f"Expected allocation based on peak (>=1100 MB), got {px.last_update['ram_mb']}"
    )


def test_scaler_min_ram_floor():
    """Scaler must never scale below the configured min_ram_mb baseline.
    With min_ram anchored to the current allocation, an idle LXC with low
    predicted usage must stay at (or above) its current allocation."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            return {"cpu_percent": 5.0, "ram_percent": 20.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):  # pylint: disable=unused-argument
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # min_ram_mb matches the current allocation (as main.py now sets it).
    # Pin min/max_cpus to 2 to prevent a CPU scale-down from confounding the test.
    baseline = {
        "min_cpus": 2,
        "max_cpus": 2,
        "min_ram_mb": 2048.0,
        "max_ram_mb": 4096.0,
    }
    predicted = {
        "cpu_percent": 5.0,
        "ram_usage_mb": 200.0,
        "recent_peak_cpu": 6.0,
        "recent_peak_ram": 250.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "cpu_percent": 5.0,
        "allocated_swap_mb": 256.0,
    }

    scaler.evaluate_and_scale("201", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Min RAM Floor (Predicted 200 MB on 2048 MB LXC):")
    print(f"Update Requested: {px.last_update}")

    # desired = 250 * 1.30 = 325 MB, but min_ram_mb=2048 clamps it to 2048.
    # RAM diff vs current (2048) = 0, CPU unchanged -> no API call at all.
    assert px.last_update is None, (
        "Scaler should not make any API call when both RAM "
        "is floored at current allocation and CPU is unchanged"
    )


def test_scaler_small_deficit_triggers_update():
    """A 40 MB deficit (above the 32 MB threshold) must trigger an update.
    Under the old 64 MB threshold this would have been silently ignored."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            return {"cpu_percent": 5.0, "ram_percent": 20.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):  # pylint: disable=unused-argument
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # Scenario: LXC needs ~540 MB but was allocated 500 MB -> 40 MB gap
    # peak = 415 MB -> desired = 415 * 1.30 = 539.5 -> int = 539, rounded to 539
    # diff = |539 - 500| = 39 MB < old 64 MB threshold but >= new 32 MB threshold
    # Use a slightly higher peak so the diff is clearly >=32 MB
    # peak = 420 MB -> desired = 420 * 1.30 = 546 -> diff = 46 >= 32 -> should update
    baseline = {
        "min_cpus": 1,
        "max_cpus": 8,
        "min_ram_mb": 500.0,
        "max_ram_mb": 4096.0,
    }
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 400.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 420.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 500.0,
        "cpu_percent": 10.0,
    }

    scaler.evaluate_and_scale("202", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Small Deficit Triggers Update (46 MB gap, new 32 MB threshold):")
    print(f"Update Requested: {px.last_update}")

    assert px.last_update is not None, (
        "Scaler should issue update for a 46 MB deficit (above 32 MB threshold)"
    )


def test_scaler_swap_flush_triggered():
    """When an LXC has swap usage above the threshold the scaler must:
    1. Call update_lxc_resources with swap_mb=0 (the target cap).
    2. Call flush_lxc_swap to drain active swap back to RAM.
    """
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None
            self.flush_called = False

        def get_host_usage(self):
            return {"cpu_percent": 10.0, "ram_percent": 30.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

        def flush_lxc_swap(self, _lxc_id):
            self.flush_called = True
            return True

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # LXC with 400/512 MB swap used -> 78% -> above the 50% threshold
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 1024.0, "max_ram_mb": 4096.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 900.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 950.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 1024.0,
        "cpu_percent": 10.0,
        "swap_mb": 400.0,
        "allocated_swap_mb": 512.0,
    }

    scaler.evaluate_and_scale("203", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Swap Saturation Flush (400/512 MB swap used, >50% threshold):")
    print(f"Update Requested: {px.last_update}")
    print(f"Flush Called: {px.flush_called}")

    assert px.last_update is not None, "Scaler should have issued a resource update"
    assert px.last_update["swap_mb"] >= 256, (
        f"Scaler should set swap_mb to at least LXC_MIN_SWAP_MB (256), "
        f"got {px.last_update['swap_mb']}"
    )
    assert px.flush_called, "Scaler should have called flush_lxc_swap on swap saturation"


def test_scaler_dynamic_swap_sizing():
    """In auto mode (LXC_TARGET_SWAP_MB=-1) the scaler must size swap from the
    observed peak with a 30% buffer, floored at LXC_MIN_SWAP_MB."""
    from scaler import Scaler
    import config as cfg

    original_target = cfg.LXC_TARGET_SWAP_MB
    original_min = cfg.LXC_MIN_SWAP_MB
    cfg.LXC_TARGET_SWAP_MB = -1    # auto mode
    cfg.LXC_MIN_SWAP_MB = 256

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None
            self.flush_called = False

        def get_host_usage(self):
            return {"cpu_percent": 10.0, "ram_percent": 20.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

        def flush_lxc_swap(self, _lxc_id):
            self.flush_called = True
            return True

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # peak_swap = max(200, 300) = 300; desired = 300 * 1.30 = 390; floor=256 -> 390
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 2048.0, "max_ram_mb": 4096.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 1500.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 1600.0,
        "predicted_swap_mb": 200.0,
        "recent_peak_swap": 300.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 1400.0,
        "cpu_percent": 10.0,
        "swap_mb": 0.0,
        "allocated_swap_mb": 256.0,
    }

    scaler.evaluate_and_scale("204", "LXC", baseline, predicted, current_metrics)
    cfg.LXC_TARGET_SWAP_MB = original_target
    cfg.LXC_MIN_SWAP_MB = original_min

    print("\nTesting Dynamic Swap Sizing (peak=300 MB, expect ~390 MB):")
    print(f"Update Requested: {px.last_update}")

    assert px.last_update is not None, "Scaler should have issued an update"
    assert px.last_update["swap_mb"] >= 380, (
        f"Expected swap ~390 MB (300*1.30), got {px.last_update['swap_mb']}"
    )


def test_scaler_safe_flush_guard():
    """The scaler must NOT flush swap when RAM headroom < swap_used * 1.1,
    even when swap saturation is detected."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None
            self.flush_called = False

        def get_host_usage(self):
            return {"cpu_percent": 10.0, "ram_percent": 50.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

        def flush_lxc_swap(self, _lxc_id):
            self.flush_called = True
            return True

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # target_ram is clamped at max_ram_mb=2048; ram_usage_mb=1910 -> headroom=138 MB
    # swap_used=400 MB -> needs 400*1.1=440 MB headroom -> 138 < 440 -> flush blocked
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 2048.0, "max_ram_mb": 2048.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 1800.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 1900.0,
        "predicted_swap_mb": 400.0,
        "recent_peak_swap": 400.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 1910.0,
        "cpu_percent": 10.0,
        "swap_mb": 400.0,
        "allocated_swap_mb": 512.0,  # 400/512 = 78% > 50% threshold -> flush desired
    }

    scaler.evaluate_and_scale("205", "LXC", baseline, predicted, current_metrics)

    print("\nTesting Safe Flush Guard (headroom 148 MB < swap 400*1.1=440 MB):")
    print(f"Update Requested: {px.last_update}")
    print(f"Flush Called: {px.flush_called}")

    assert px.last_update is not None, "Scaler should still issue a resource update"
    assert not px.flush_called, (
        "Scaler must NOT flush swap when RAM headroom is insufficient"
    )


def test_scaler_swap_cap_corrected_without_ram_change():
    """Regression test: when RAM and CPU are adequate but the swap cap differs
    significantly from the target, the scaler must still issue an update to
    apply the correct swap cap — even with no RAM or CPU change.
    (Bug: swap pressure was never relieved when only swap cap needed adjusting.)
    """
    from scaler import Scaler
    import config as cfg

    original_target = cfg.LXC_TARGET_SWAP_MB
    original_min = cfg.LXC_MIN_SWAP_MB
    cfg.LXC_TARGET_SWAP_MB = -1  # auto mode
    cfg.LXC_MIN_SWAP_MB = 256

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None
            self.flush_called = False

        def get_host_usage(self):
            return {"cpu_percent": 10.0, "ram_percent": 20.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

        def flush_lxc_swap(self, _lxc_id):
            self.flush_called = True
            return True

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # RAM is fine (usage 900 MB on 2048 MB allocation -> no RAM scaling needed).
    # CPU is fine. BUT: Proxmox currently has swap_cap=2048 MB (old default),
    # while ML says target_swap = max(200, 150) * 1.30 = 260 MB -> swap_diff=1788 MB.
    # The scaler MUST update to apply the new 260 MB cap even with no RAM/CPU change.
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 2048.0, "max_ram_mb": 2048.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 900.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 950.0,
        "predicted_swap_mb": 200.0,
        "recent_peak_swap": 150.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 900.0,
        "cpu_percent": 10.0,
        "swap_mb": 30.0,              # Low swap usage — below flush threshold
        "allocated_swap_mb": 2048.0,  # Old Proxmox default (huge)
    }

    scaler.evaluate_and_scale("206", "LXC", baseline, predicted, current_metrics)
    cfg.LXC_TARGET_SWAP_MB = original_target
    cfg.LXC_MIN_SWAP_MB = original_min

    print("\nTesting Swap Cap Corrected Without RAM Change:")
    print(f"Update Requested: {px.last_update}")

    assert px.last_update is not None, (
        "Scaler must issue an update when swap_diff >= 32, even with no RAM/CPU change"
    )
    # target_swap = max(int(200 * 1.30), 256) = max(260, 256) = 260 MB
    assert px.last_update["swap_mb"] <= 300, (
        f"Swap cap should have been reduced to ~260 MB, got {px.last_update['swap_mb']}"
    )
    assert px.last_update["swap_mb"] >= 256, (
        f"Swap cap should be at least LXC_MIN_SWAP_MB=256, got {px.last_update['swap_mb']}"
    )


def test_scaler_host_pressure_active_scaledown():
    """When host RAM > 90% and container is using < 50% of its allocation,
    the scaler must actively push the container's RAM down to reclaim host headroom."""
    from scaler import Scaler
    import scaler as scaler_mod

    original_threshold = scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
    scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = 90.0

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            # Host RAM at 92% — above the 90% active scale-down threshold
            return {"cpu_percent": 40.0, "ram_percent": 92.0, "swap_percent": 5.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # Container: 2048 MB allocated, only 400 MB used (19.5% — well below 50% idle threshold)
    # Expected: scaled down to max(int(400 * 1.5), min_ram_mb=512) = max(600, 512) = 600 MB
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 512.0, "max_ram_mb": 4096.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 400.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 420.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 400.0,
        "cpu_percent": 10.0,
    }

    scaler.evaluate_and_scale("400", "LXC", baseline, predicted, current_metrics)
    scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = original_threshold

    print("\nTesting Host Pressure Active Scale-Down (Host 92% RAM, container 400/2048 MB used):")
    print(f"Update Requested: {px.last_update}")

    assert px.last_update is not None, "Scaler should have issued an update to reclaim RAM"
    assert px.last_update["ram_mb"] <= 650, (
        f"Expected active scale-down to ~600 MB (400*1.5), got {px.last_update['ram_mb']}"
    )
    assert px.last_update["ram_mb"] >= 512, (
        f"RAM must not go below min_ram_mb=512, got {px.last_update['ram_mb']}"
    )


def test_scaler_host_pressure_busy_container_protected():
    """When host RAM > 90% but the container is actively using > 50% of its allocation,
    the active scale-down must NOT fire — a busy container is protected."""
    from scaler import Scaler
    import scaler as scaler_mod

    original_threshold = scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
    scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = 90.0

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            return {"cpu_percent": 40.0, "ram_percent": 92.0, "swap_percent": 5.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # Container: 2048 MB allocated, 1800 MB used (87.9% — above 50% threshold)
    # No scale-up needed either (predicted low). Result: no meaningful change -> no API call.
    baseline = {"min_cpus": 2, "max_cpus": 2, "min_ram_mb": 2048.0, "max_ram_mb": 4096.0}
    predicted = {
        "cpu_percent": 10.0,
        "ram_usage_mb": 1800.0,
        "recent_peak_cpu": 12.0,
        "recent_peak_ram": 1850.0,
    }
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "ram_usage_mb": 1800.0,
        "cpu_percent": 10.0,
        # Set swap to LXC_MIN_SWAP_MB so the swap diff is zero and
        # the test exclusively validates the active scale-down guard.
        "allocated_swap_mb": 256.0,
        "swap_mb": 0.0,
    }

    scaler.evaluate_and_scale("401", "LXC", baseline, predicted, current_metrics)
    scaler_mod.HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = original_threshold

    print("\nTesting Busy Container Protected from Host Pressure Scale-Down (1800/2048 MB used):")
    print(f"Update Requested: {px.last_update}")

    # The container is busy; active scale-down must not fire.
    # desired RAM = 1850 * 1.30 = 2405 -> blocked by host RAM 92% cap -> stays at 2048.
    # No cpu/ram/swap diff -> no API call.
    assert px.last_update is None, (
        "Busy container must NOT be scaled down even when host RAM is critically high"
    )


def test_scaler_host_swap_safety_cap():
    """Scaler must refuse to scale UP CPU or RAM when host swap usage exceeds MAX_HOST_SWAP_USAGE_PERCENT."""
    from scaler import Scaler
    import config as cfg

    original_cap = cfg.MAX_HOST_SWAP_USAGE_PERCENT
    cfg.MAX_HOST_SWAP_USAGE_PERCENT = 20.0

    class MockProxmoxClient:
        def __init__(self):
            self.last_update = None

        def get_host_usage(self):
            # Host CPU and RAM are fine, but swap is at 25% (over 20% limit)
            return {"cpu_percent": 50.0, "ram_percent": 50.0, "swap_percent": 25.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb, swap_mb=0):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb, "swap_mb": swap_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    baseline = {"min_cpus": 1, "max_cpus": 4, "min_ram_mb": 1024.0, "max_ram_mb": 8192.0}
    predicted = {"cpu_percent": 90.0, "ram_usage_mb": 3000.0}
    current_metrics = {"allocated_cpus": 1, "allocated_ram_mb": 1024.0, "cpu_percent": 60.0}

    scaler.evaluate_and_scale("300", "LXC", baseline, predicted, current_metrics)

    cfg.MAX_HOST_SWAP_USAGE_PERCENT = original_cap

    print("\nTesting Host Swap Safety Cap (Host 25% Swap > 20% Limit):")
    print(f"Update Requested: {px.last_update}")

    # Scale UP should be blocked, keeping variables at current baseline (1 cpu, 1024 RAM)
    # The scaler still calls the API because swap auto-provisioning asks for LXC_MIN_SWAP_MB
    assert px.last_update is not None
    assert px.last_update["cpus"] == 1, "CPU scale-up must be blocked by host swap cap"
    assert px.last_update["ram_mb"] == 1024.0, "RAM scale-up must be blocked by host swap cap"
def test_vm_pending_config_from_rolling_peaks():
    """14-day rolling peak data → pending config includes 30% headroom and is written."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_vm_update = None

        def get_host_usage(self):
            return {"cpu_percent": 30.0, "ram_percent": 50.0, "swap_percent": 2.0,
                    "total_ram_mb": 64000, "physical_cpus": 16,
                    "load_avg_1m": 1.0, "load_avg_5m": 0.9, "ksm_sharing_mb": 0.0}

        def update_vm_resources(self, _vm_id, cpus, ram_mb):
            self.last_vm_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    baseline  = {"min_cpus": 2, "max_cpus": 8, "min_ram_mb": 1024.0, "max_ram_mb": 16384.0}
    predicted = {"cpu_percent": 50.0, "ram_usage_mb": 2000.0,
                 "recent_peak_cpu": 60.0, "recent_peak_ram": 2100.0}
    current   = {"allocated_cpus": 2, "allocated_ram_mb": 1024.0, "ram_usage_mb": 900.0,
                 "cpu_percent": 50.0}
    # 14-day observed peak: 3000 MB RAM / 80% CPU on 2 cores
    rolling_peaks = {"peak_cpu_pct": 80.0, "peak_ram_mb": 3000.0, "sample_count": 1440}

    scaler.apply_vm_pending_config("500", baseline, predicted, current, rolling_peaks)

    print("\nTesting VM Pending Config from Rolling Peaks (3000 MB peak, 80% CPU):")
    print(f"VM Update: {px.last_vm_update}")

    # target_ram = int(3000 * 1.30) = 3900, clamped to [1024, 16384]
    assert px.last_vm_update is not None, "Should have written a pending config"
    assert px.last_vm_update["ram_mb"] == 3900, (
        f"Expected 3900 MB (3000 * 1.30), got {px.last_vm_update['ram_mb']}"
    )
    # needed_cores = int(0.80 * 2 * 1.20) + 1 = int(1.92) + 1 = 2 -> max(2, min(2,8)) = 2
    assert px.last_vm_update["cpus"] >= 2, "Should have at least baseline min_cpus"


def test_vm_pending_config_bootstrap():
    """No log data (day one) → bootstrap from ML prediction peaks, still writes config."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_vm_update = None

        def get_host_usage(self):
            return {"cpu_percent": 20.0, "ram_percent": 40.0, "swap_percent": 0.0,
                    "total_ram_mb": 64000, "physical_cpus": 16,
                    "load_avg_1m": 0.5, "load_avg_5m": 0.4, "ksm_sharing_mb": 0.0}

        def update_vm_resources(self, _vm_id, cpus, ram_mb):
            self.last_vm_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    baseline  = {"min_cpus": 1, "max_cpus": 8, "min_ram_mb": 1024.0, "max_ram_mb": 16384.0}
    predicted = {"cpu_percent": 40.0, "ram_usage_mb": 2500.0,
                 "recent_peak_cpu": 55.0, "recent_peak_ram": 2800.0}
    current   = {"allocated_cpus": 4, "allocated_ram_mb": 1024.0, "ram_usage_mb": 800.0,
                 "cpu_percent": 40.0}
    rolling_peaks = {"peak_cpu_pct": 0.0, "peak_ram_mb": 0.0, "sample_count": 0}

    scaler.apply_vm_pending_config("501", baseline, predicted, current, rolling_peaks)

    print("\nTesting VM Pending Config Bootstrap (no log data, using prediction peaks):")
    print(f"VM Update: {px.last_vm_update}")

    # Bootstrap: peak_ram = max(2500, 2800) = 2800; target_ram = int(2800*1.30) = 3640
    assert px.last_vm_update is not None, "Should bootstrap from prediction peaks"
    assert px.last_vm_update["ram_mb"] == 3640, (
        f"Expected 3640 MB (2800 * 1.30), got {px.last_vm_update['ram_mb']}"
    )


def test_vm_pending_config_no_change():
    """Peaks don't exceed 5% delta from current allocation → no API call made."""
    from scaler import Scaler

    class MockProxmoxClient:
        def __init__(self):
            self.last_vm_update = None

        def get_host_usage(self):
            return {"cpu_percent": 20.0, "ram_percent": 40.0, "swap_percent": 0.0,
                    "total_ram_mb": 64000, "physical_cpus": 16,
                    "load_avg_1m": 0.5, "load_avg_5m": 0.4, "ksm_sharing_mb": 0.0}

        def update_vm_resources(self, _vm_id, cpus, ram_mb):
            self.last_vm_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    # VM allocated 4096 MB.  Rolling peak 3000 MB -> target = int(3000*1.30) = 3900.
    # Delta from 4096: abs(3900 - 4096) / 4096 * 100 = 4.8% < 5% -> no write.
    baseline  = {"min_cpus": 4, "max_cpus": 8, "min_ram_mb": 1024.0, "max_ram_mb": 16384.0}
    predicted = {"cpu_percent": 50.0, "ram_usage_mb": 2900.0,
                 "recent_peak_cpu": 55.0, "recent_peak_ram": 3000.0}
    current   = {"allocated_cpus": 4, "allocated_ram_mb": 4096.0, "ram_usage_mb": 2900.0,
                 "cpu_percent": 50.0}
    rolling_peaks = {"peak_cpu_pct": 60.0, "peak_ram_mb": 3000.0, "sample_count": 500}

    scaler.apply_vm_pending_config("502", baseline, predicted, current, rolling_peaks)

    print("\nTesting VM Pending Config No-Change (4.8% delta < 5% threshold):")
    print(f"VM Update: {px.last_vm_update}")

    assert px.last_vm_update is None, (
        "Should NOT write config when change is < 5% (avoids micro-updates every cycle)"
    )


if __name__ == "__main__":
    print("Running Mock AI Predictor Tests...")
    test_scale_event_logging()
    test_predictor()
    test_predictor_new_metric_keys()
    test_predictor_delta_features()
    test_predictor_host_context_full()
    test_scaler()
    test_scaler_uses_peak_ram()
    test_scaler_min_ram_floor()
    test_scaler_small_deficit_triggers_update()
    test_scaler_swap_flush_triggered()
    test_scaler_dynamic_swap_sizing()
    test_scaler_safe_flush_guard()
    test_scaler_swap_cap_corrected_without_ram_change()
    test_scaler_host_pressure_active_scaledown()
    test_scaler_host_pressure_busy_container_protected()
    test_scaler_host_swap_safety_cap()
    test_vm_pending_config_from_rolling_peaks()
    test_vm_pending_config_bootstrap()
    test_vm_pending_config_no_change()
    print("All mock tests passed!")
