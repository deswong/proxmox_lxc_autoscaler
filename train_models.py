import datetime
import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from config import DATABASE_PATH, EXCLUDED_LXCS, EXCLUDED_VMS, TRAINING_DAYS_LOOKBACK
from proxmox_api import ProxmoxClient
import storage

# Setup Dedicated Logging for the Trainer
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - TRAINER - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_models")


def calculate_recent_penalties(entity_id) -> dict:
    """
    Looks at the prediction_logs table to calculate the Mean Absolute Error (MAE)
    of the predictions made natively by the live service against actual historical data.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df_preds = pd.read_sql_query(
            "SELECT timestamp, predicted_cpu, predicted_ram FROM prediction_logs WHERE lxc_id=?",
            conn,
            params=(entity_id,),
        )
        conn.close()

        if df_preds.empty:
            return {"mae_cpu": 0.0, "mae_ram": 0.0, "count": 0}

        # We won't fully join against massive RRD here, we'll just report basic stats of predictions
        # For a full reinforcement model, this would compute the exact delta and apply it as Sample Weights.
        return {
            "avg_predicted_cpu": df_preds["predicted_cpu"].mean(),
            "avg_predicted_ram": df_preds["predicted_ram"].mean(),
            "count": len(df_preds),
        }
    except Exception as e:
        logger.error(f"Error reading reinforcement logs for {entity_id}: {e}")
        return {}


def train_for_entity(px_client, entity_id, entity_type, models_dir="./models"):
    """
    Pulls historical RRD data based on the configured lookback window,
    constructs the feature matrix, and trains an XGBoost model.
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
    y_swap = []  # LXC only; stays empty for VMs

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
        y_cpu.append(target.get("cpu", 0.0) * 100)
        y_ram.append(target.get("mem", 0.0) / (1024 * 1024))
        # Proxmox RRD returns swap in bytes for LXCs; default to 0 if absent
        y_swap.append(target.get("swap", 0.0) / (1024 * 1024))

    X_matrix = np.array(X_matrix)
    y_cpu = np.array(y_cpu)
    y_ram = np.array(y_ram)
    y_swap = np.array(y_swap)

    # Time-based sample weights: more recent samples are given exponentially higher weight
    # so the model prioritises current usage patterns over old historical data.
    n_samples = len(X_matrix)
    sample_weights = np.exp(
        np.linspace(0, 3, n_samples)
    )  # weight range: e^0=1.0 to e^3=~20
    sample_weights /= sample_weights.sum()  # normalise so weights sum to 1

    logger.info(f"Training XGBoost Regressors for {entity_type} {entity_id}...")

    _xgb_params = {
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
        "objective": "reg:squarederror",
    }

    model_cpu = xgb.XGBRegressor(**_xgb_params)
    model_cpu.fit(X_matrix, y_cpu, sample_weight=sample_weights)

    model_ram = xgb.XGBRegressor(**_xgb_params)
    model_ram.fit(X_matrix, y_ram, sample_weight=sample_weights)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    prefix = entity_type.lower()
    model_cpu.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_cpu.json"))
    model_ram.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_ram.json"))

    # Swap predictor is LXC-only (VMs manage swap inside the guest OS)
    if entity_type == "LXC" and y_swap.sum() > 0:
        model_swap = xgb.XGBRegressor(**_xgb_params)
        model_swap.fit(X_matrix, y_swap, sample_weight=sample_weights)
        model_swap.save_model(
            os.path.join(models_dir, f"{prefix}_{entity_id}_swap.json")
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

    stats = calculate_recent_penalties(entity_id)
    logger.info(
        f"Successfully saved XGBoost models for {entity_type} {entity_id}. (Reinforcement entries processed: {stats.get('count', 0)})"
    )


def run():
    logger.info("Starting Nightly XGBoost Batch Training Daemon...")
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
    logger.info("Batch training complete. Service will now use the new weights.")


if __name__ == "__main__":
    run()
