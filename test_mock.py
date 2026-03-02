import time
from predictor import Predictor


def test_predictor():
    predictor = Predictor(prediction_horizon=2)

    # Simulate a rising trend
    # format: [{'time': timestamp, 'cpu': ratio, 'mem': bytes}]
    base_time = time.time() - 300
    metrics = [
        {"time": base_time, "cpu": 0.10, "mem": 512 * 1024 * 1024},
        {"time": base_time + 60, "cpu": 0.20, "mem": 600 * 1024 * 1024},
        {"time": base_time + 120, "cpu": 0.30, "mem": 750 * 1024 * 1024},
        {"time": base_time + 180, "cpu": 0.50, "mem": 900 * 1024 * 1024},
        {"time": base_time + 240, "cpu": 0.75, "mem": 1024 * 1024 * 1024},
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
        {"time": base_time, "cpu": 0.90, "mem": 2048 * 1024 * 1024},
        {"time": base_time + 60, "cpu": 0.80, "mem": 1900 * 1024 * 1024},
        {"time": base_time + 120, "cpu": 0.50, "mem": 1500 * 1024 * 1024},
        {"time": base_time + 180, "cpu": 0.30, "mem": 1024 * 1024 * 1024},
        {"time": base_time + 240, "cpu": 0.15, "mem": 512 * 1024 * 1024},
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
        "swap_mb": 30.0,            # Low swap usage — below flush threshold
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


if __name__ == "__main__":
    print("Running Mock AI Predictor Tests...")
    test_predictor()
    test_scaler()
    test_scaler_uses_peak_ram()
    test_scaler_min_ram_floor()
    test_scaler_small_deficit_triggers_update()
    test_scaler_swap_flush_triggered()
    test_scaler_dynamic_swap_sizing()
    test_scaler_safe_flush_guard()
    test_scaler_swap_cap_corrected_without_ram_change()
    print("All mock tests passed!")
