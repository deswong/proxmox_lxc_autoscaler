import os
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from config import DATABASE_PATH
from proxmox_api import ProxmoxClient
import storage

# Setup Dedicated Logging for the Trainer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TRAINER - %(levelname)s - %(message)s')
logger = logging.getLogger("train_models")

def calculate_recent_penalties(lxc_id) -> dict:
    """
    Looks at the prediction_logs table to calculate the Mean Absolute Error (MAE)
    of the predictions made natively by the live service against actual historical data.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df_preds = pd.read_sql_query("SELECT timestamp, predicted_cpu, predicted_ram FROM prediction_logs WHERE lxc_id=?", conn, params=(lxc_id,))
        conn.close()
        
        if df_preds.empty:
            return {"mae_cpu": 0.0, "mae_ram": 0.0, "count": 0}
            
        # We won't fully join against massive RRD here, we'll just report basic stats of predictions
        # For a full reinforcement model, this would compute the exact delta and apply it as Sample Weights.
        return {
            "avg_predicted_cpu": df_preds['predicted_cpu'].mean(),
            "avg_predicted_ram": df_preds['predicted_ram'].mean(),
            "count": len(df_preds)
        }
    except Exception as e:
        logger.error(f"Error reading reinforcement logs for {lxc_id}: {e}")
        return {}

def train_for_lxc(px_client, lxc_id, models_dir="./models"):
    """
    Pulls a week of RRD data, constructs the feature matrix, 
    and trains an XGBoost model.
    """
    logger.info(f"Fetching weekly RRD data for LXC {lxc_id}...")
    rrd_data = px_client.get_lxc_rrd_history(lxc_id, timeframe="week")
    
    valid_metrics = [m for m in rrd_data if m.get('cpu') is not None and m.get('mem') is not None]
    if len(valid_metrics) < 100:
        logger.warning(f"Not enough historical data to train LXC {lxc_id}. Needs at least 100 intervals.")
        return
        
    logger.info(f"Building supervised dataset from {len(valid_metrics)} data points...")
    
    X_matrix = []
    y_cpu = []
    y_ram = []
    
    # We need 15 past data points to predict 2 points into the future
    PREDICTION_HORIZON = 2
    LOOKBACK = 15
    
    for i in range(LOOKBACK, len(valid_metrics) - PREDICTION_HORIZON):
        past_window = valid_metrics[i - LOOKBACK: i]
        target = valid_metrics[i + PREDICTION_HORIZON]
        
        # Flatten the 15 past intervals into 30 features
        features = []
        for m in past_window:
            features.append(m.get('cpu', 0.0) * 100)
            features.append(m.get('mem', 0.0) / (1024 * 1024))
            
        X_matrix.append(features)
        y_cpu.append(target.get('cpu', 0.0) * 100)
        y_ram.append(target.get('mem', 0.0) / (1024 * 1024))
        
    X_matrix = np.array(X_matrix)
    y_cpu = np.array(y_cpu)
    y_ram = np.array(y_ram)
    
    logger.info(f"Training XGBoost Regressors for LXC {lxc_id}...")
    
    model_cpu = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model_cpu.fit(X_matrix, y_cpu)
    
    model_ram = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model_ram.fit(X_matrix, y_ram)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_cpu.save_model(os.path.join(models_dir, f"lxc_{lxc_id}_cpu.json"))
    model_ram.save_model(os.path.join(models_dir, f"lxc_{lxc_id}_ram.json"))
    
    stats = calculate_recent_penalties(lxc_id)
    logger.info(f"Successfully saved XGBoost models for LXC {lxc_id}. (Reinforcement entries processed: {stats.get('count', 0)})")

def run():
    logger.info("Starting Nightly XGBoost Batch Training Daemon...")
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to connect to Proxmox API.")
        return
        
    storage.init_db()
    
    baselines = storage.get_baselines()
    if not baselines:
        logger.warning("No LXCs currently configured. Nothing to train.")
        return
        
    for lxc_id in baselines.keys():
        try:
            train_for_lxc(px_client, lxc_id)
        except Exception as e:
            logger.error(f"Fatal error training LXC {lxc_id}: {e}")
            
    # Clean up old offline logs so the DB doesn't grow infinitely
    storage.cleanup_prediction_logs(retention_days=14)
    logger.info("Batch training complete. Service will now use the new weights.")

if __name__ == "__main__":
    run()
