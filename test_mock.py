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

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    scaler = Scaler(px)

    baseline = {"min_cpus": 1, "max_cpus": 8, "min_ram_mb": 512, "max_ram_mb": 8192}
    predicted = {"cpu_percent": 95.0, "ram_usage_mb": 4096.0}
    current_metrics = {
        "allocated_cpus": 2,
        "allocated_ram_mb": 2048.0,
        "cpu_percent": 80.0,
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

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb):
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

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb):
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

        def update_lxc_resources(self, _lxc_id, cpus, ram_mb):
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


if __name__ == "__main__":
    print("Running Mock AI Predictor Tests...")
    test_predictor()
    test_scaler()
    test_scaler_uses_peak_ram()
    test_scaler_min_ram_floor()
    test_scaler_small_deficit_triggers_update()
    print("All mock tests passed!")
