import sqlite3
import json
import time
import logging
from typing import Dict, List, Tuple
from config import DATABASE_PATH, INITIAL_LXC_CONFIGS

logger = logging.getLogger("storage")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table to store configured baselines for each LXC
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lxc_baselines (
            lxc_id TEXT PRIMARY KEY,
            min_cpus INTEGER,
            min_ram_mb INTEGER,
            max_cpus INTEGER,
            max_ram_mb INTEGER,
            updated_at REAL
        )
    ''')
    
    # Table to store reinforcement learning metrics (predicted vs actual)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lxc_id TEXT,
            timestamp REAL,
            predicted_cpu REAL,
            predicted_ram REAL
        )
    ''')
    
    # Create an index for faster time-series querying during batch training
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_pred_lxc_time ON prediction_logs(lxc_id, timestamp)
    ''')
    
    conn.commit()
    conn.close()
    
    _seed_initial_baselines()

def _seed_initial_baselines():
    """
    Syncs the database with baselines from the .env file.
    This ensures that if an LXC is deleted and recreated with a new config in .env,
    the new configuration strictly overrides any old data upon service restart.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    current_time = time.time()
    
    # 1. Update or Insert configs from .env tracking
    for lxc_id, config in INITIAL_LXC_CONFIGS.items():
        cursor.execute('''
            INSERT INTO lxc_baselines (lxc_id, min_cpus, min_ram_mb, max_cpus, max_ram_mb, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(lxc_id) DO UPDATE SET
                min_cpus=excluded.min_cpus,
                min_ram_mb=excluded.min_ram_mb,
                max_cpus=excluded.max_cpus,
                max_ram_mb=excluded.max_ram_mb,
                updated_at=excluded.updated_at
        ''', (str(lxc_id), config.get('min_cpus', 1), config.get('min_ram_mb', 512), 
              config.get('max_cpus', 4), config.get('max_ram_mb', 4096), current_time))
              
    # 2. Prune any forgotten instances out of the local cache
    if INITIAL_LXC_CONFIGS:
        placeholders = ','.join(['?'] * len(INITIAL_LXC_CONFIGS))
        cursor.execute(f"DELETE FROM lxc_baselines WHERE lxc_id NOT IN ({placeholders})", list(INITIAL_LXC_CONFIGS.keys()))
    else:
        cursor.execute("DELETE FROM lxc_baselines")
    
    conn.commit()
    conn.close()

def get_baselines() -> Dict[str, Dict]:
    """Retrieves all LXC baselines from the DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lxc_baselines")
    rows = cursor.fetchall()
    
    baselines = {}
    for row in rows:
        baselines[row['lxc_id']] = {
            'min_cpus': row['min_cpus'],
            'min_ram_mb': row['min_ram_mb'],
            'max_cpus': row['max_cpus'],
            'max_ram_mb': row['max_ram_mb']
        }
    
    conn.close()
    return baselines

def log_prediction(lxc_id: str, predicted_cpu: float, predicted_ram: float):
    """
    Saves a prediction to the database so the batch training script can 
    compare it against actual historical usage later to calculate penalties.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    current_time = time.time()
    cursor.execute('''
        INSERT INTO prediction_logs (lxc_id, timestamp, predicted_cpu, predicted_ram)
        VALUES (?, ?, ?, ?)
    ''', (str(lxc_id), current_time, predicted_cpu, predicted_ram))
    
    conn.commit()
    conn.close()

def cleanup_prediction_logs(retention_days=14):
    """
    Prunes prediction logs older than the retention period to stop the SQLite file 
    from growing forever. Defaults to 14 days, offering plenty of time for weekly training.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cutoff_time = time.time() - (retention_days * 86400)
    cursor.execute("DELETE FROM prediction_logs WHERE timestamp < ?", (cutoff_time,))
    deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    if deleted > 0:
        logger.debug(f"Pruned {deleted} old reinforcement learning logs.")
