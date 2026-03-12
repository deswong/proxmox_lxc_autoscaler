import datetime
import time
import logging
from config import POLL_INTERVAL_SECONDS, EXCLUDED_LXCS, EXCLUDED_VMS
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

    # Warn if poll interval is below Proxmox's RRD resolution (1-minute granularity).
    # Polling faster than 60s wastes API calls since no new RRD data will be available yet.
    if POLL_INTERVAL_SECONDS < 60:
        logger.warning(
            f"POLL_INTERVAL_SECONDS is set to {POLL_INTERVAL_SECONDS}s, which is below Proxmox's "
            f"RRD resolution of 60s. Predictions will repeat the same data until a new data point "
            f"arrives. Consider setting POLL_INTERVAL_SECONDS=60 or higher."
        )

    # Main run loop
    while True:
        cycle_start_time = time.time()

        # 1. (Legacy SQLite Telemetry Cleanup Removed)

        # 2. Re-fetch explicit baselines (from .env/DB) and excluded lists
        explicit_baselines = storage.get_baselines()

        # 3. Bulk fetch telemetry for all running LXCs and VMs
        all_lxc_metrics = px_client.get_all_lxc_metrics()
        all_vm_metrics = px_client.get_all_vm_metrics()

        if not all_lxc_metrics and not all_vm_metrics:
            logger.warning(
                "No running LXCs or VMs found on this node. Nothing to monitor."
            )

        # 3b. Fetch host node metrics once per cycle — used as ML context AND safety caps
        host_metrics = px_client.get_host_usage()

        # 3c. Compute cluster overcommit ratios from data already in RAM (zero extra API calls).
        #     These are merged into host_context so both the predictor and the log entry
        #     see the current tenant density on the hypervisor.
        all_containers = {**all_lxc_metrics, **all_vm_metrics}
        physical_cpus = max(host_metrics.get("physical_cpus", 1), 1)
        total_ram_mb = max(host_metrics.get("total_ram_mb", 1.0), 1.0)
        total_alloc_cpus = sum(m.get("allocated_cpus", 0) for m in all_containers.values())
        total_alloc_ram = sum(m.get("allocated_ram_mb", 0.0) for m in all_containers.values())
        host_metrics["cpu_overcommit_ratio"] = total_alloc_cpus / physical_cpus
        host_metrics["ram_overcommit_ratio"] = total_alloc_ram / total_ram_mb
        host_metrics["container_count"] = float(len(all_containers))
        _now = datetime.datetime.now()

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

                # Check and record initial resource allocation to notes
                notes = px_client.get_entity_notes(lxc_id, "LXC")
                if "[Proxmox AI Autoscaler - Initial Allocation]" not in notes:
                    logger.info(
                        f"[LXC {lxc_id}] First time detection. Recording initial resources to notes."
                    )
                    initial_allocation_stamp = (
                        "\n\n[Proxmox AI Autoscaler - Initial Allocation]\n"
                        f"CPU Cores: {current_metrics['allocated_cpus']}\n"
                        f"RAM: {int(current_metrics['allocated_ram_mb'])} MB\n"
                    )
                    px_client.set_entity_notes(
                        lxc_id, "LXC", notes + initial_allocation_stamp
                    )

                # Determine baseline: Use explicit if it exists, otherwise build a dynamic one
                if lxc_id in explicit_baselines:
                    baseline = explicit_baselines[lxc_id]
                else:
                    # Dynamic Zero-Config Baseline Setup.
                    # min_ram_mb is anchored to the current allocation so the
                    # scaler never shrinks a container below its provisioned size.
                    baseline = {
                        "min_cpus": 1,
                        "min_ram_mb": max(128, int(current_metrics["allocated_ram_mb"])),
                        "max_cpus": current_metrics["allocated_cpus"] + 4,
                        "max_ram_mb": max(256, int(current_metrics["allocated_ram_mb"] * 2)),
                    }
                    logger.debug(
                        f"[LXC {lxc_id}] Using dynamic fallback baseline: {baseline}"
                    )
                # Retrieve recent RRD time-series (last hour is sufficient; we only use 15 points)
                historical_metrics = px_client.get_lxc_rrd_history(
                    lxc_id, timeframe="hour"
                )

                # Predict impending usage (pass live host metrics as ML context)
                predicted_usage = predictor.predict_next_usage(
                    lxc_id, historical_metrics, entity_type="LXC",
                    host_context=host_metrics,
                )

                if predicted_usage is None:
                    continue

                # Record this prediction for the nightly XGBoost trainer to review and learn from
                try:
                    storage.log_prediction(
                        lxc_id,
                        predicted_usage["cpu_percent"],
                        predicted_usage["ram_usage_mb"],
                        predicted_usage.get("predicted_swap_mb", 0.0),
                        predicted_usage.get("recent_peak_disk_read", 0.0),
                        predicted_usage.get("recent_peak_disk_write", 0.0),
                        predicted_usage.get("recent_peak_net_in", 0.0),
                        predicted_usage.get("recent_peak_net_out", 0.0),
                        ctx_hour=_now.hour,
                        ctx_dow=_now.weekday(),
                        ctx_host_load_1m=host_metrics.get("load_avg_1m", 0.0),
                        ctx_host_load_5m=host_metrics.get("load_avg_5m", 0.0),
                        ctx_cpu_overcommit=host_metrics.get("cpu_overcommit_ratio", 0.0),
                        ctx_ram_overcommit=host_metrics.get("ram_overcommit_ratio", 0.0),
                        ctx_container_count=int(host_metrics.get("container_count", 0)),
                        ctx_ksm_sharing_mb=host_metrics.get("ksm_sharing_mb", 0.0),
                        ctx_actual_cpu=current_metrics.get("cpu_percent", 0.0),
                        ctx_actual_ram=current_metrics.get("ram_usage_mb", 0.0),
                        ctx_actual_swap=current_metrics.get("swap_mb", 0.0),
                        ctx_min_cpus=baseline.get("min_cpus", 0),
                        ctx_max_cpus=baseline.get("max_cpus", 0),
                        ctx_min_ram=baseline.get("min_ram_mb", 0),
                        ctx_max_ram=baseline.get("max_ram_mb", 0),
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

                # Check and record initial resource allocation to notes
                notes = px_client.get_entity_notes(vm_id, "VM")
                if "[Proxmox AI Autoscaler - Initial Allocation]" not in notes:
                    logger.info(
                        f"[VM {vm_id}] First time detection. Recording initial resources to notes."
                    )
                    initial_allocation_stamp = (
                        "\n\n[Proxmox AI Autoscaler - Initial Allocation]\n"
                        f"CPU Cores: {current_metrics['allocated_cpus']}\n"
                        f"RAM: {int(current_metrics['allocated_ram_mb'])} MB\n"
                    )
                    px_client.set_entity_notes(
                        vm_id, "VM", notes + initial_allocation_stamp
                    )

                # Determine baseline: Use explicit if it exists, otherwise build a dynamic one
                if vm_id in explicit_baselines:
                    baseline = explicit_baselines[vm_id]
                else:
                    # Dynamic Zero-Config Baseline Setup.
                    # min_ram_mb is anchored to the current allocation so the
                    # scaler never shrinks a VM below its provisioned size.
                    baseline = {
                        "min_cpus": 1,
                        "min_ram_mb": max(1024, int(current_metrics["allocated_ram_mb"])),
                        "max_cpus": current_metrics["allocated_cpus"] + 4,
                        "max_ram_mb": max(1024, int(current_metrics["allocated_ram_mb"] * 2)),
                    }
                    logger.debug(
                        f"[VM {vm_id}] Using dynamic fallback baseline: {baseline}"
                    )

                # Retrieve recent RRD time-series (last hour is sufficient; we only use 15 points)
                historical_metrics = px_client.get_vm_rrd_history(
                    vm_id, timeframe="hour"
                )

                # Predict impending usage (pass live host metrics as ML context)
                predicted_usage = predictor.predict_next_usage(
                    vm_id, historical_metrics, entity_type="VM",
                    host_context=host_metrics,
                )

                if predicted_usage is None:
                    continue

                # Record this prediction for the nightly XGBoost trainer to review and learn from
                try:
                    storage.log_prediction(
                        vm_id,
                        predicted_usage["cpu_percent"],
                        predicted_usage["ram_usage_mb"],
                        predicted_usage.get("predicted_swap_mb", 0.0),
                        predicted_usage.get("recent_peak_disk_read", 0.0),
                        predicted_usage.get("recent_peak_disk_write", 0.0),
                        predicted_usage.get("recent_peak_net_in", 0.0),
                        predicted_usage.get("recent_peak_net_out", 0.0),
                        ctx_hour=_now.hour,
                        ctx_dow=_now.weekday(),
                        ctx_host_load_1m=host_metrics.get("load_avg_1m", 0.0),
                        ctx_host_load_5m=host_metrics.get("load_avg_5m", 0.0),
                        ctx_cpu_overcommit=host_metrics.get("cpu_overcommit_ratio", 0.0),
                        ctx_ram_overcommit=host_metrics.get("ram_overcommit_ratio", 0.0),
                        ctx_container_count=int(host_metrics.get("container_count", 0)),
                        ctx_actual_cpu=current_metrics.get("cpu_percent", 0.0),
                        ctx_actual_ram=current_metrics.get("ram_usage_mb", 0.0),
                    )
                except Exception as db_err:
                    logger.warning(
                        f"[VM {vm_id}] Failed to log prediction for reinforcement learning: {db_err}"
                    )

                # Compute optimal next-reboot sizing from 14-day rolling peak.
                # No live hotplug is attempted for VMs — the config is written as
                # a pending Proxmox entry that takes effect on the next reboot,
                # ensuring the VM always has headroom without risking guest OS crashes.
                rolling_peaks = storage.get_vm_rolling_peaks(vm_id)
                scaler.apply_vm_pending_config(
                    vm_id, baseline, predicted_usage, current_metrics, rolling_peaks
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
