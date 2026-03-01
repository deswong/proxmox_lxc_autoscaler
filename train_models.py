import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from config import DATABASE_PATH
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
    from config import TRAINING_DAYS_LOOKBACK

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

    valid_metrics = [
        m for m in rrd_data if m.get("cpu") is not None and m.get("mem") is not None
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

    # We need 15 past data points to predict 2 points into the future
    PREDICTION_HORIZON = 2
    LOOKBACK = 15

    for i in range(LOOKBACK, len(valid_metrics) - PREDICTION_HORIZON):
        past_window = valid_metrics[i - LOOKBACK : i]
        target = valid_metrics[i + PREDICTION_HORIZON]

        # Flatten the 15 past intervals into 30 features
        features = []
        for m in past_window:
            features.append(m.get("cpu", 0.0) * 100)
            features.append(m.get("mem", 0.0) / (1024 * 1024))

        X_matrix.append(features)
        y_cpu.append(target.get("cpu", 0.0) * 100)
        y_ram.append(target.get("mem", 0.0) / (1024 * 1024))

    X_matrix = np.array(X_matrix)
    y_cpu = np.array(y_cpu)
    y_ram = np.array(y_ram)

    logger.info(f"Training XGBoost Regressors for {entity_type} {entity_id}...")

    model_cpu = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, objective="reg:squarederror"
    )
    model_cpu.fit(X_matrix, y_cpu)

    model_ram = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, objective="reg:squarederror"
    )
    model_ram.fit(X_matrix, y_ram)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    prefix = entity_type.lower()
    model_cpu.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_cpu.json"))
    model_ram.save_model(os.path.join(models_dir, f"{prefix}_{entity_id}_ram.json"))

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

    from config import EXCLUDED_LXCS

    # 2. Discover all LXCs on the node
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
    from config import EXCLUDED_VMS

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
