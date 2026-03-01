import numpy as np
import xgboost as xgb
import logging
import os
from typing import List

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

        self._model_cache = {}  # Store loaded models in RAM
        self._model_mtimes = (
            {}
        )  # Store file modification times to detect fresh nightly trains

    def _get_model(self, model_path: str):
        """
        Retrieves a cached XGBoost Booster from RAM, or loads it from disk if it's new/updated.
        """
        if not os.path.exists(model_path):
            # Explicit RAM optimization: If the user deleted the container or added it to EXCLUDE_List,
            # the nightly trainer will delete the .json file. We must evict it from RAM.
            if model_path in self._model_cache:
                del self._model_cache[model_path]
                del self._model_mtimes[model_path]
            return None

        mtime = os.path.getmtime(model_path)

        # Cache hit
        if (
            model_path in self._model_cache
            and self._model_mtimes.get(model_path) == mtime
        ):
            return self._model_cache[model_path]

        # Cache miss or file was updated
        try:
            # Using raw Booster is fundamentally faster and lighter than the Scikit-Learn XGBRegressor wrapper
            booster = xgb.Booster()
            booster.load_model(model_path)
            self._model_cache[model_path] = booster
            self._model_mtimes[model_path] = mtime
            return booster
        except Exception as e:
            logger.error(
                f"Failed to load native C++ XGBoost model from {model_path}: {e}"
            )
            return None

    def predict_next_usage(
        self, entity_id: str, rrd_data: List[dict], entity_type: str = "LXC"
    ) -> dict:
        """
        Takes chronological data from Proxmox RRD API.
        Only performs fast inference using pre-trained XGBoost weights.
        If no weights exist yet (first day), falls back to the latest telemetry reading safely.
        """
        # Filter out invalid or purely null/0 data points that might appear in RRD
        valid_metrics = [
            m for m in rrd_data if m.get("cpu") is not None and m.get("mem") is not None
        ]

        if not valid_metrics:
            logger.warning(
                f"No valid telemetry data received for {entity_type} {entity_id}. Aborting prediction to prevent dangerous scale-down."
            )
            return None

        # We want the most recent 15 valid data points for a smooth, fast trend
        metrics = valid_metrics[-15:]

        # Capture peaks before destroying the array
        highest_recent_cpu = float(max(m.get("cpu", 0.0) * 100 for m in metrics))
        highest_recent_ram = float(
            max(m.get("mem", 0.0) / (1024 * 1024) for m in metrics)
        )

        # Explicit memory optimization: We no longer need the heavy original RRD json array
        # or the large filtered array. Free them before we spin up Scikit-Learn matrices.
        del rrd_data
        del valid_metrics

        latest = metrics[-1]
        fallback_cpu = latest.get("cpu", 0.0) * 100
        fallback_ram = latest.get("mem", 0.0) / (1024 * 1024)

        if len(metrics) < 15:
            # Not enough data for the rigid XGBoost feature array, fallback
            del metrics
            return {
                "cpu_percent": fallback_cpu,
                "ram_usage_mb": fallback_ram,
                "recent_peak_cpu": highest_recent_cpu,
                "recent_peak_ram": highest_recent_ram,
            }

        # Try to load models
        prefix = entity_type.lower()
        cpu_model_path = os.path.join(self.models_dir, f"{prefix}_{entity_id}_cpu.json")
        ram_model_path = os.path.join(self.models_dir, f"{prefix}_{entity_id}_ram.json")

        pred_cpu = fallback_cpu
        pred_ram = fallback_ram

        if os.path.exists(cpu_model_path) and os.path.exists(ram_model_path):
            try:
                # Prepare data identically to training phase: flatten the 15 intervals into 30 features
                X_features = []
                for m in metrics:
                    X_features.append((m.get("cpu", 0.0) * 100))
                    X_features.append((m.get("mem", 0.0) / (1024 * 1024)))

                # Retrieve from lightning RAM cache instead of Disk Load
                model_cpu = self._get_model(cpu_model_path)
                model_ram = self._get_model(ram_model_path)

                if model_cpu and model_ram:
                    # Native XGBoost Booster uses DMatrix instead of raw numpy lists directly
                    dmatrix = xgb.DMatrix(np.array([X_features]))

                    pred_cpu = max(0.0, float(model_cpu.predict(dmatrix)[0]))
                    pred_ram = max(0.0, float(model_ram.predict(dmatrix)[0]))

            except Exception as e:
                logger.error(
                    f"Failed to run XGBoost inference for {entity_type} {entity_id}: {e}"
                )
        else:
            logger.debug(
                f"No XGBoost models found yet for {entity_type} {entity_id}. Falling back to live metrics."
            )

        # Explicit memory optimization
        del metrics

        return {
            "cpu_percent": pred_cpu,
            "ram_usage_mb": pred_ram,
            "recent_peak_cpu": highest_recent_cpu,
            "recent_peak_ram": highest_recent_ram,
        }
