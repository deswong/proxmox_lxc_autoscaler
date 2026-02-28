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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

def run():
    logger.info("Starting Proxmox LXC Autoscaler Service...")
    
    # Initialize SQLite database and baseline configurations
    storage.init_db()
    
    # Initialize Proxmox API Client
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to initialize Proxmox Client. Fatal Error.")
        return
        
    predictor = Predictor(prediction_horizon=2) # Predict 2 cycles (minutes) ahead
    scaler = Scaler(px_client)
    
    # Main run loop
    while True:
        cycle_start_time = time.time()
        
        # 1. (Legacy SQLite Telemetry Cleanup Removed)
        
        # 2. Re-fetch baselines in case they were updated externally
        baselines = storage.get_baselines()
        
        if not baselines:
            logger.warning("No LXC baselines configured in database. Nothing to monitor.")
        
        # 3. Evaluate each configured LXC
        for lxc_id, baseline in baselines.items():
            try:
                # Fetch current telemetry from hypervisor
                current_metrics = px_client.get_lxc_metrics(lxc_id)
                if not current_metrics:
                    logger.debug(f"[LXC {lxc_id}] Is not running or could not fetch metrics. Skipping.")
                    continue
                
                # Retrieve recent RRD time-series graph directly from Proxmox
                historical_metrics = px_client.get_lxc_rrd_history(lxc_id, timeframe="hour")
                
                # Predict impending usage
                predicted_usage = predictor.predict_next_usage(lxc_id, historical_metrics)
                
                # Record this prediction for the nightly XGBoost trainer to review and learn from
                try:
                    storage.log_prediction(lxc_id, predicted_usage['cpu_percent'], predicted_usage['ram_usage_mb'])
                except Exception as db_err:
                    logger.warning(f"[LXC {lxc_id}] Failed to log prediction for reinforcement learning: {db_err}")
                
                # Evaluate and emit scaling decisions
                scaler.evaluate_and_scale(lxc_id, baseline, predicted_usage, current_metrics)
                
            except Exception as e:
                logger.error(f"[LXC {lxc_id}] Exception during autoscaling cycle: {e}")
                
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
