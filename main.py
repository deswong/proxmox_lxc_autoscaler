import time
import logging
from config import POLL_INTERVAL_SECONDS
import storage
from proxmox_api import ProxmoxClient
from predictor import Predictor
from scaler import Scaler
import gc

# Configure root logger for the service
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def run():
    logger.info("Starting Proxmox Universal AI Autoscaler Service...")

    # Initialize SQLite database and baseline configurations
    storage.init_db()

    # Initialize Proxmox API Client
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to initialize Proxmox Client. Fatal Error.")
        return

    predictor = Predictor(prediction_horizon=2)  # Predict 2 cycles (minutes) ahead
    scaler = Scaler(px_client)

    # Main run loop
    while True:
        cycle_start_time = time.time()

        # 1. (Legacy SQLite Telemetry Cleanup Removed)

        # 2. Re-fetch explicit baselines (from .env/DB) and excluded lists
        explicit_baselines = storage.get_baselines()
        from config import EXCLUDED_LXCS

        # 3. Bulk fetch telemetry for all running LXCs and VMs
        all_lxc_metrics = px_client.get_all_lxc_metrics()
        all_vm_metrics = px_client.get_all_vm_metrics()

        from config import EXCLUDED_VMS

        if not all_lxc_metrics and not all_vm_metrics:
            logger.warning(
                "No running LXCs or VMs found on this node. Nothing to monitor."
            )

        # 4. Evaluate each discovered LXC
        for lxc_id, current_metrics in all_lxc_metrics.items():
            if lxc_id in EXCLUDED_LXCS:
                logger.debug(
                    f"[LXC {lxc_id}] Skipping (Listed in EXCLUDED_LXCS blacklist)."
                )
                continue

            # Skip recently booted entities (under 15 minutes) to prevent learning from boot storms
            if current_metrics.get("uptime", 0) < 900:
                logger.info(
                    f"[LXC {lxc_id}] Skipping scaling. Entity just booted (Uptime: {current_metrics.get('uptime', 0)}s < 900s)."
                )
                continue

            try:
                # current_metrics is already bulk-fetched from RAM

                # Determine baseline: Use explicit if it exists, otherwise build a dynamic one
                if lxc_id in explicit_baselines:
                    baseline = explicit_baselines[lxc_id]
                else:
                    # Dynamic Zero-Config Baseline Setup
                    baseline = {
                        "min_cpus": 1,
                        "min_ram_mb": 512,
                        "max_cpus": current_metrics["allocated_cpus"] + 4,
                        "max_ram_mb": current_metrics["allocated_ram_mb"] * 2,
                    }
                    logger.debug(
                        f"[LXC {lxc_id}] Using dynamic fallback baseline: {baseline}"
                    )
                # Retrieve recent RRD time-series graph directly from Proxmox
                historical_metrics = px_client.get_lxc_rrd_history(
                    lxc_id, timeframe="hour"
                )

                # Predict impending usage
                predicted_usage = predictor.predict_next_usage(
                    lxc_id, historical_metrics, entity_type="LXC"
                )

                if predicted_usage is None:
                    continue

                # Record this prediction for the nightly XGBoost trainer to review and learn from
                try:
                    storage.log_prediction(
                        lxc_id,
                        predicted_usage["cpu_percent"],
                        predicted_usage["ram_usage_mb"],
                    )
                except Exception as db_err:
                    logger.warning(
                        f"[LXC {lxc_id}] Failed to log prediction for reinforcement learning: {db_err}"
                    )

                # Evaluate and emit scaling decisions
                scaler.evaluate_and_scale(
                    lxc_id, "LXC", baseline, predicted_usage, current_metrics
                )

            except Exception as e:
                logger.error(f"[LXC {lxc_id}] Exception during autoscaling cycle: {e}")

        # 5. Evaluate each discovered VM
        for vm_id, current_metrics in all_vm_metrics.items():
            if vm_id in EXCLUDED_VMS:
                logger.debug(
                    f"[VM {vm_id}] Skipping (Listed in EXCLUDED_VMS blacklist)."
                )
                continue

            # Skip recently booted entities (under 15 minutes) to prevent learning from boot storms
            if current_metrics.get("uptime", 0) < 900:
                logger.info(
                    f"[VM {vm_id}] Skipping scaling. Entity just booted (Uptime: {current_metrics.get('uptime', 0)}s < 900s)."
                )
                continue

            try:
                # current_metrics is already bulk-fetched from RAM

                # Determine baseline: Use explicit if it exists, otherwise build a dynamic one
                if vm_id in explicit_baselines:
                    baseline = explicit_baselines[vm_id]
                else:
                    # Dynamic Zero-Config Baseline Setup
                    baseline = {
                        "min_cpus": 1,
                        "min_ram_mb": 512,
                        "max_cpus": current_metrics["allocated_cpus"] + 4,
                        "max_ram_mb": current_metrics["allocated_ram_mb"] * 2,
                    }
                    logger.debug(
                        f"[VM {vm_id}] Using dynamic fallback baseline: {baseline}"
                    )

                # Retrieve recent RRD time-series graph directly from Proxmox
                historical_metrics = px_client.get_vm_rrd_history(
                    vm_id, timeframe="hour"
                )

                # Predict impending usage
                predicted_usage = predictor.predict_next_usage(
                    vm_id, historical_metrics, entity_type="VM"
                )

                if predicted_usage is None:
                    continue

                # Record this prediction for the nightly XGBoost trainer to review and learn from
                try:
                    storage.log_prediction(
                        vm_id,
                        predicted_usage["cpu_percent"],
                        predicted_usage["ram_usage_mb"],
                    )
                except Exception as db_err:
                    logger.warning(
                        f"[VM {vm_id}] Failed to log prediction for reinforcement learning: {db_err}"
                    )

                # Evaluate and emit scaling decisions
                scaler.evaluate_and_scale(
                    vm_id, "VM", baseline, predicted_usage, current_metrics
                )

            except Exception as e:
                logger.error(f"[VM {vm_id}] Exception during autoscaling cycle: {e}")

        # Force garbage collection to keep daemon memory usage very low over time
        gc.collect()

        # Sleep until next poll interval
        elapsed = time.time() - cycle_start_time
        sleep_time = max(0.5, POLL_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}")
