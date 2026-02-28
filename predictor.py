import numpy as np
import xgboost as xgb
import logging
import os
from typing import List, Tuple

logger = logging.getLogger("predictor")

class Predictor:
    def __init__(self, prediction_horizon=2, models_dir="./models"):
        """
        prediction_horizon: Number of future intervals (minutes) to predict.
        models_dir: Location where the nightly training cron task will save the XGBoost .json weights.
        """
        self.prediction_horizon = prediction_horizon
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
    def predict_next_usage(self, lxc_id: str, rrd_data: List[dict]) -> dict:
        """
        Takes chronological data from Proxmox RRD API.
        Only performs fast inference using pre-trained XGBoost weights. 
        If no weights exist yet (first day), falls back to the latest telemetry reading safely.
        """
        # Filter out invalid or purely null/0 data points that might appear in RRD
        valid_metrics = [m for m in rrd_data if m.get('cpu') is not None and m.get('mem') is not None]
        
        if not valid_metrics:
            return {"cpu_percent": 0.0, "ram_usage_mb": 0.0, "recent_peak_cpu": 0.0, "recent_peak_ram": 0.0}
            
        # We want the most recent 15 valid data points for a smooth, fast trend
        metrics = valid_metrics[-15:]
        
        # Capture peaks before destroying the array
        highest_recent_cpu = float(max([m.get('cpu', 0.0) * 100 for m in metrics]))
        highest_recent_ram = float(max([m.get('mem', 0.0) / (1024 * 1024) for m in metrics]))
        
        # Explicit memory optimization: We no longer need the heavy original RRD json array
        # or the large filtered array. Free them before we spin up Scikit-Learn matrices.
        del rrd_data
        del valid_metrics
            
        latest = metrics[-1]
        fallback_cpu = (latest.get('cpu', 0.0) * 100)
        fallback_ram = (latest.get('mem', 0.0) / (1024 * 1024))
        
        if len(metrics) < 15:
            # Not enough data for the rigid XGBoost feature array, fallback
            del metrics
            return {
                "cpu_percent": fallback_cpu,
                "ram_usage_mb": fallback_ram,
                "recent_peak_cpu": highest_recent_cpu,
                "recent_peak_ram": highest_recent_ram
            }
            
        # Try to load models
        cpu_model_path = os.path.join(self.models_dir, f"lxc_{lxc_id}_cpu.json")
        ram_model_path = os.path.join(self.models_dir, f"lxc_{lxc_id}_ram.json")
        
        pred_cpu = fallback_cpu
        pred_ram = fallback_ram
        
        if os.path.exists(cpu_model_path) and os.path.exists(ram_model_path):
            try:
                # Prepare data identically to training phase: flatten the 15 intervals into 30 features
                X_features = []
                for m in metrics:
                    X_features.append((m.get('cpu', 0.0) * 100))
                    X_features.append((m.get('mem', 0.0) / (1024 * 1024)))
                
                # Single row prediction matrix
                X_pred = np.array([X_features])
                
                model_cpu = xgb.XGBRegressor()
                model_cpu.load_model(cpu_model_path)
                
                model_ram = xgb.XGBRegressor()
                model_ram.load_model(ram_model_path)
                
                pred_cpu = max(0.0, float(model_cpu.predict(X_pred)[0]))
                pred_ram = max(0.0, float(model_ram.predict(X_pred)[0]))

                # Explicitly drop XGBoost C++ handles
                del model_cpu
                del model_ram

            except Exception as e:
                logger.error(f"Failed to run XGBoost inference for LXC {lxc_id}: {e}")
        else:
            logger.debug(f"No XGBoost models found yet for LXC {lxc_id}. Falling back to live metrics.")
                
        # Explicit memory optimization
        del metrics
        
        return {
            "cpu_percent": pred_cpu,
            "ram_usage_mb": pred_ram,
            "recent_peak_cpu": highest_recent_cpu,
            "recent_peak_ram": highest_recent_ram
        }
