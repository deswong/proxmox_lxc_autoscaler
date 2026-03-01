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
        def get_host_usage(self):
            # Simulate host node under extreme load, triggering the 95% safety cap
            return {"cpu_percent": 96.0, "ram_percent": 98.0, "total_ram_mb": 64000}

        def update_lxc_resources(self, lxc_id, cpus, ram_mb):
            self.last_update = {"cpus": cpus, "ram_mb": ram_mb}

    px = MockProxmoxClient()
    px.last_update = None
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


if __name__ == "__main__":
    print("Running Mock AI Predictor Tests...")
    test_predictor()
    test_scaler()
    print("All mock tests passed!")
