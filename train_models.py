import datetime
import os
import logging
import numpy as np
import lightgbm as lgb
from config import EXCLUDED_LXCS, EXCLUDED_VMS, TRAINING_DAYS_LOOKBACK
from proxmox_api import ProxmoxClient
import storage

# Setup Dedicated Logging for the Trainer
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - TRAINER - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_models")



def train_for_entity(px_client, entity_id, entity_type, models_dir="./models"):
    """
    Pulls historical RRD data based on the configured lookback window,
    constructs the feature matrix, and trains three LightGBM regressors:
    one for CPU%, one for RAM MB, and one for swap MB (LXC only).
    """
    # Proxmox API only accepts specific timeframe strings.
    if TRAINING_DAYS_LOOKBACK <= 7:
        timeframe = "week"
    elif TRAINING_DAYS_LOOKBACK <= 30:
        timeframe = "month"
    else:
        timeframe = "year"

    logger.info(
        f"Fetching {TRAINING_DAYS_LOOKBACK} days of RRD data ({timeframe}) for {entity_type} {entity_id}..."
    )

    if entity_type == "LXC":
        rrd_data = px_client.get_lxc_rrd_history(entity_id, timeframe=timeframe)
    else:
        rrd_data = px_client.get_vm_rrd_history(entity_id, timeframe=timeframe)

    # Load error penalties from prediction_logs BEFORE building the matrix.
    # Maps {minute_ts: penalty_multiplier (1.0-3.0)} — empty on day one.
    error_penalties = storage.get_prediction_errors(entity_id)
    if error_penalties:
        logger.info(
            f"[{entity_type} {entity_id}] Loaded {len(error_penalties)} error-penalty "
            f"buckets (avg penalty: {sum(error_penalties.values())/len(error_penalties):.2f}×)."
        )
    else:
        logger.info(
            f"[{entity_type} {entity_id}] No logged prediction errors yet — "
            "using pure time-based weights for this training run."
        )

    # Fetch node-level RRD for the same timeframe to provide host context features.
    # Build a {unix_time: {cpu%, ram%, swap%}} lookup keyed to the nearest minute.
    node_rrd = px_client.get_node_rrd_history(timeframe=timeframe)
    node_context_by_time = {}
    for n in node_rrd:
        ts = n.get("time", 0)
        if ts == 0:
            continue
        mem_total = n.get("memtotal", 0) or 1  # avoid division by zero
        swap_total = n.get("swaptotal", 0) or 1
        node_context_by_time[ts] = {
            "cpu_pct": float(n.get("cpu", 0.0) * 100),
            "ram_pct": float(n.get("memused", 0.0) / mem_total * 100),
            "swap_pct": float(n.get("swapused", 0.0) / swap_total * 100),
        }

    # Compute cluster overcommit once at training start (one extra API call).
    # Live-only fields (load_avg, ksm) remain 0.0 in training — the model will start
    # weighting them once they accumulate in prediction_logs from live cycles.
    all_lxc = px_client.get_all_lxc_metrics()
    all_vm = px_client.get_all_vm_metrics()
    all_containers = {**all_lxc, **all_vm}
    host_info = px_client.get_host_usage()
    physical_cpus = max(host_info.get("physical_cpus", 1), 1)
    total_ram_mb = max(host_info.get("total_ram_mb", 1.0), 1.0)
    total_alloc_cpus = sum(m.get("allocated_cpus", 0) for m in all_containers.values())
    total_alloc_ram = sum(m.get("allocated_ram_mb", 0.0) for m in all_containers.values())
    cpu_overcommit = total_alloc_cpus / physical_cpus
    ram_overcommit = total_alloc_ram / total_ram_mb
    container_count = float(len(all_containers))

    valid_metrics = [
        m
        for m in rrd_data
        if m.get("cpu") is not None and m.get("mem") is not None
    ]
    if len(valid_metrics) < 30:
        logger.warning(
            f"Not enough historical data to train {entity_type} {entity_id}. Needs at least 30 intervals."
        )
        return

    logger.info(f"Building supervised dataset from {len(valid_metrics)} data points...")

    X_matrix = []
    y_cpu = []
    y_ram = []
    y_swap = []          # LXC only; stays empty for VMs
    window_timestamps = []  # end-timestamp of each training window (for error-penalty lookup)

    # We need 15 past data points to predict 2 points into the future.
    # Full feature vector layout — MUST stay in sync with Predictor._build_context_features():
    #   [0-89]   Per-interval history: cpu%, mem_mb, diskread, diskwrite, netin, netout (6 × 15)
    #   [90-99]  Node health: host_cpu%, host_ram%, host_swap%, load_1m, load_5m, ksm,
    #            cpu_overcommit, ram_overcommit, container_count, reserved
    #   [100-101] Temporal: hour_of_day, day_of_week
    #   [101-106] Deltas: Δcpu%, Δmem_mb, Δdiskread, Δdiskwrite, Δnetin, Δnetout
    #   Total: 107 features
    PREDICTION_HORIZON = 2
    LOOKBACK = 15

    for i in range(LOOKBACK, len(valid_metrics) - PREDICTION_HORIZON):
        past_window = valid_metrics[i - LOOKBACK : i]
        target = valid_metrics[i + PREDICTION_HORIZON]

        # Flatten the 15 past intervals into 90 features (6 per interval).
        # Disk and network fields default to 0.0 when absent from historical RRD data.
        features = []
        for m in past_window:
            features.append(m.get("cpu", 0.0) * 100)
            features.append(m.get("mem", 0.0) / (1024 * 1024))
            features.append(m.get("diskread", 0.0))
            features.append(m.get("diskwrite", 0.0))
            features.append(m.get("netin", 0.0))
            features.append(m.get("netout", 0.0))

        # Append 17 global context features to match the predictor layout exactly.
        # Feature order MUST stay in sync with Predictor._build_context_features().
        window_ts = past_window[-1].get("time", 0)
        host_snap = node_context_by_time.get(
            window_ts,
            # try nearest minute bucket (node RRD may be rounded differently)
            node_context_by_time.get(window_ts - (window_ts % 60), {}),
        )
        # [90-99] Node health (10 values)
        features.append(host_snap.get("cpu_pct", 0.0))
        features.append(host_snap.get("ram_pct", 0.0))
        features.append(host_snap.get("swap_pct", 0.0))
        features.append(0.0)  # load_avg_1m — not in RRD; populated at inference time
        features.append(0.0)  # load_avg_5m — not in RRD; populated at inference time
        features.append(0.0)  # ksm_sharing_mb — not in RRD; populated at inference time
        features.append(cpu_overcommit)  # cluster CPU overcommit ratio
        features.append(ram_overcommit)  # cluster RAM overcommit ratio
        features.append(container_count)  # number of running containers
        features.append(0.0)  # reserved
        # [100-101] Temporal context (2 values)
        window_dt = datetime.datetime.fromtimestamp(window_ts) if window_ts else datetime.datetime.now()
        features.append(float(window_dt.hour))     # hour_of_day
        features.append(float(window_dt.weekday())) # day_of_week
        # [101-106] Rate-of-change deltas (6 values: last - first of window)
        first, last = past_window[0], past_window[-1]
        features.append((last.get("cpu", 0.0) - first.get("cpu", 0.0)) * 100)
        features.append((last.get("mem", 0.0) - first.get("mem", 0.0)) / (1024 * 1024))
        features.append(last.get("diskread", 0.0) - first.get("diskread", 0.0))
        features.append(last.get("diskwrite", 0.0) - first.get("diskwrite", 0.0))
        features.append(last.get("netin", 0.0) - first.get("netin", 0.0))
        features.append(last.get("netout", 0.0) - first.get("netout", 0.0))

        X_matrix.append(features)
        window_timestamps.append(window_ts)
        y_cpu.append(target.get("cpu", 0.0) * 100)
        y_ram.append(target.get("mem", 0.0) / (1024 * 1024))
        # Proxmox RRD returns swap in bytes for LXCs; default to 0 if absent
        y_swap.append(target.get("swap", 0.0) / (1024 * 1024))

    X_matrix = np.array(X_matrix)
    y_cpu = np.array(y_cpu)
    y_ram = np.array(y_ram)
    y_swap = np.array(y_swap)

    # Sample weights: compound time-recency with prediction-error penalties.
    #
    # 1. Time-recency: exponential ramp from e^0=1 to e^3≈20 so recent
    #    intervals are favoured over old history.
    # 2. Error penalty: for intervals where a previous model run made a
    #    large prediction error (from prediction_logs telemetry), multiply
    #    the sample weight by 1.0–3.0 so XGBoost trains harder on mistakes.
    #
    # Both components are then re-normalised so weights sum to 1.
    n_samples = len(X_matrix)
    time_weights = np.exp(np.linspace(0, 3, n_samples))  # e^0 → e^3 ≈ 20

    error_multipliers = np.ones(n_samples)
    for i, window_ts in enumerate(window_timestamps):
        minute_ts = int(window_ts) - (int(window_ts) % 60)
        penalty = error_penalties.get(
            minute_ts,
            error_penalties.get(minute_ts - 60, 1.0),  # ±1 minute tolerance
        )
        error_multipliers[i] = penalty

    sample_weights = time_weights * error_multipliers
    sample_weights /= sample_weights.sum()  # normalise so weights sum to 1

    if error_penalties:
        boosted = int((error_multipliers > 1.05).sum())
        logger.info(
            f"[{entity_type} {entity_id}] Error-penalty weighting applied: "
            f"{boosted}/{n_samples} intervals boosted above 1× "
            f"(max penalty: {error_multipliers.max():.2f}×)."
        )

    logger.info(f"Training LightGBM regressors for {entity_type} {entity_id}...")

    # LightGBM params — DART booster reduces over-fitting on the recency-weighted distribution.
    # hour_of_day (feature index 100) and day_of_week (101) are declared as ordered categoricals
    # so LightGBM learns temporal thresholds rather than treating them as continuous floats.
    _lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "dart",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    # Feature indices of temporals — these are declared categorical so LightGBM
    # groups hours and weekdays into ordered buckets rather than treating them as floats.
    _categorical_features = [100, 101]
    _num_boost_round = 300

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    prefix = entity_type.lower()

    ds_cpu = lgb.Dataset(
        X_matrix, label=y_cpu,
        weight=sample_weights,
        categorical_feature=_categorical_features,
        free_raw_data=True,
    )
    model_cpu = lgb.train(_lgb_params, ds_cpu, num_boost_round=_num_boost_round)
    model_cpu.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_cpu.lgb"))

    ds_ram = lgb.Dataset(
        X_matrix, label=y_ram,
        weight=sample_weights,
        categorical_feature=_categorical_features,
        free_raw_data=True,
    )
    model_ram = lgb.train(_lgb_params, ds_ram, num_boost_round=_num_boost_round)
    model_ram.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_ram.lgb"))

    # Swap predictor is LXC-only (VMs manage swap inside the guest OS)
    if entity_type == "LXC" and y_swap.sum() > 0:
        ds_swap = lgb.Dataset(
            X_matrix, label=y_swap,
            weight=sample_weights,
            categorical_feature=_categorical_features,
            free_raw_data=True,
        )
        model_swap = lgb.train(_lgb_params, ds_swap, num_boost_round=_num_boost_round)
        model_swap.save_model(
            os.path.join(models_dir, f"{prefix}_{entity_id}_swap.lgb")
        )
        logger.info(
            f"[LXC {entity_id}] Swap predictor trained "
            f"(peak observed: {y_swap.max():.0f} MB)."
        )
    elif entity_type == "LXC":
        logger.info(
            f"[LXC {entity_id}] No swap usage in training window — "
            "skipping swap model (will use LXC_MIN_SWAP_MB floor)."
        )

    logger.info(
        f"Successfully saved LightGBM models for {entity_type} {entity_id}."
    )


def run():
    logger.info("Starting Nightly LightGBM Batch Training Daemon...")
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to connect to Proxmox API.")
        return

    storage.init_db()

    # Discover and train all LXCs then all VMs on this node.
    all_lxc_ids = px_client.get_all_lxc_ids()

    if not all_lxc_ids:
        logger.warning("No LXCs found on this node. Nothing to train.")
        return

    for lxc_id in all_lxc_ids:
        if lxc_id in EXCLUDED_LXCS:
            logger.debug(
                f"Skipping training for LXC {lxc_id} (Listed in EXCLUDED_LXCS)."
            )
            continue

        try:
            train_for_entity(px_client, lxc_id, "LXC")
        except Exception as e:
            logger.error(f"Fatal error training LXC {lxc_id}: {e}")

    # 3. Discover all VMs on the node
    all_vm_ids = px_client.get_all_vm_ids()

    if not all_vm_ids:
        logger.warning("No VMs found on this node. Nothing to train.")
    else:
        for vm_id in all_vm_ids:
            if vm_id in EXCLUDED_VMS:
                logger.debug(
                    f"Skipping training for VM {vm_id} (Listed in EXCLUDED_VMS)."
                )
                continue

            try:
                train_for_entity(px_client, vm_id, "VM")
            except Exception as e:
                logger.error(f"Fatal error training VM {vm_id}: {e}")

    # Clean up old offline logs so the DB doesn't grow infinitely
    storage.cleanup_prediction_logs(retention_days=14)

    # Emit a 24-hour performance snapshot to the training log so the operator
    # can see savings and accuracy trends without running report.py manually.
    try:
        summary = storage.get_performance_summary(days=1)
        ev = summary["scale_events"]
        logger.info(
            "=== 24h Performance Snapshot === "
            f"Scale events: {ev['total']} total "
            f"({ev['scale_up_count']} up / {ev['scale_down_count']} down / "
            f"{ev['vm_pending_count']} VM pending). "
            f"Net RAM freed: {ev['net_ram_freed_mb']:.0f} MB. "
            f"Host-pressure events: {ev['host_pressure_count']}."
        )
        for a in summary["prediction_accuracy"]:
            logger.info(
                f"  Accuracy [{a['entity_id']}]: "
                f"CPU MAE ±{a['mae_cpu_pct']:.1f}%, "
                f"RAM MAE ±{a['mae_ram_mb']:.0f} MB  "
                f"({a['samples']} samples)"
            )
    except Exception as report_err:
        logger.warning(f"Performance summary failed: {report_err}")

    logger.info("Batch training complete. Service will now use the new weights.")


if __name__ == "__main__":
    run()
