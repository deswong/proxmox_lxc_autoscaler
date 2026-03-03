import sqlite3
import time
import logging
from typing import Dict
from config import DATABASE_PATH, INITIAL_LXC_CONFIGS, INITIAL_VM_CONFIGS

INITIAL_CONFIGS = {**INITIAL_LXC_CONFIGS, **INITIAL_VM_CONFIGS}

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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lxc_baselines (
            lxc_id TEXT PRIMARY KEY,
            min_cpus INTEGER,
            min_ram_mb INTEGER,
            max_cpus INTEGER,
            max_ram_mb INTEGER,
            updated_at REAL
        )
    """)

    # Table to store reinforcement learning metrics (predicted vs actual)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lxc_id TEXT,
            timestamp REAL,
            predicted_cpu REAL,
            predicted_ram REAL,
            predicted_swap REAL DEFAULT 0.0,
            pred_disk_read REAL DEFAULT 0.0,
            pred_disk_write REAL DEFAULT 0.0,
            pred_net_in REAL DEFAULT 0.0,
            pred_net_out REAL DEFAULT 0.0
        )
    """)

    # Migrations: add columns to pre-existing databases that lack them
    _migrate_add_column(cursor, "prediction_logs", "predicted_swap", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "pred_disk_read", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "pred_disk_write", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "pred_net_in", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "pred_net_out", "REAL DEFAULT 0.0")
    # Context telemetry columns (added in v3 — rich per-cycle environment snapshot)
    _migrate_add_column(cursor, "prediction_logs", "ctx_hour", "INTEGER DEFAULT 0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_dow", "INTEGER DEFAULT 0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_host_load_1m", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_host_load_5m", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_cpu_overcommit", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_ram_overcommit", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_container_count", "INTEGER DEFAULT 0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_actual_cpu", "REAL DEFAULT 0.0")
    _migrate_add_column(cursor, "prediction_logs", "ctx_actual_ram", "REAL DEFAULT 0.0")

    # Create an index for faster time-series querying during batch training
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_lxc_time ON prediction_logs(lxc_id, timestamp)
    """)

    conn.commit()
    conn.close()

    _seed_initial_baselines()


def _migrate_add_column(cursor, table: str, column: str, col_type: str):
    """Adds a column to an existing table if it does not yet exist."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:  # pylint: disable=broad-except
        pass  # Column already exists — normal on subsequent startups


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
    for entity_id, config in INITIAL_CONFIGS.items():
        cursor.execute(
            """
            INSERT INTO lxc_baselines (lxc_id, min_cpus, min_ram_mb, max_cpus, max_ram_mb, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(lxc_id) DO UPDATE SET
                min_cpus=excluded.min_cpus,
                min_ram_mb=excluded.min_ram_mb,
                max_cpus=excluded.max_cpus,
                max_ram_mb=excluded.max_ram_mb,
                updated_at=excluded.updated_at
        """,
            (
                str(entity_id),
                config.get("min_cpus", 1),
                config.get("min_ram_mb", 512),
                config.get("max_cpus", 4),
                config.get("max_ram_mb", 4096),
                current_time,
            ),
        )

    # 2. Prune any forgotten instances out of the local cache
    if INITIAL_CONFIGS:
        placeholders = ",".join(["?"] * len(INITIAL_CONFIGS))
        cursor.execute(
            f"DELETE FROM lxc_baselines WHERE lxc_id NOT IN ({placeholders})",
            list(INITIAL_CONFIGS.keys()),
        )
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
        baselines[row["lxc_id"]] = {
            "min_cpus": row["min_cpus"],
            "min_ram_mb": row["min_ram_mb"],
            "max_cpus": row["max_cpus"],
            "max_ram_mb": row["max_ram_mb"],
        }

    conn.close()
    return baselines


def log_prediction(
    lxc_id: str,
    predicted_cpu: float,
    predicted_ram: float,
    predicted_swap: float = 0.0,
    pred_disk_read: float = 0.0,
    pred_disk_write: float = 0.0,
    pred_net_in: float = 0.0,
    pred_net_out: float = 0.0,
    ctx_hour: int = 0,
    ctx_dow: int = 0,
    ctx_host_load_1m: float = 0.0,
    ctx_host_load_5m: float = 0.0,
    ctx_cpu_overcommit: float = 0.0,
    ctx_ram_overcommit: float = 0.0,
    ctx_container_count: int = 0,
    ctx_actual_cpu: float = 0.0,
    ctx_actual_ram: float = 0.0,
):
    """
    Saves a prediction to the database along with the full environment context
    observed at the moment of prediction. This rich per-cycle snapshot lets future
    training runs use actual load_avg / overcommit data instead of placeholder zeros.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    current_time = time.time()
    cursor.execute(
        """
        INSERT INTO prediction_logs (
            lxc_id, timestamp, predicted_cpu, predicted_ram, predicted_swap,
            pred_disk_read, pred_disk_write, pred_net_in, pred_net_out,
            ctx_hour, ctx_dow, ctx_host_load_1m, ctx_host_load_5m,
            ctx_cpu_overcommit, ctx_ram_overcommit, ctx_container_count,
            ctx_actual_cpu, ctx_actual_ram
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            str(lxc_id), current_time, predicted_cpu, predicted_ram, predicted_swap,
            pred_disk_read, pred_disk_write, pred_net_in, pred_net_out,
            ctx_hour, ctx_dow, ctx_host_load_1m, ctx_host_load_5m,
            ctx_cpu_overcommit, ctx_ram_overcommit, ctx_container_count,
            ctx_actual_cpu, ctx_actual_ram,
        ),
    )

    conn.commit()
    conn.close()


def get_vm_rolling_peaks(entity_id: str, days: int = 14) -> dict:
    """
    Returns the rolling maximum CPU% and RAM usage observed for a VM/LXC over
    the last ``days`` days from the prediction_logs telemetry table.

    Uses the ``ctx_actual_cpu`` and ``ctx_actual_ram`` columns written by the live
    daemon every cycle, so the peaks reflect what the entity *actually* used —
    not just what the ML model predicted.

    Returns:
        {
            "peak_cpu_pct": float,   # highest observed CPU%  (0 if no data)
            "peak_ram_mb":  float,   # highest observed RAM MB (0 if no data)
            "sample_count": int,     # rows found; 0 means bootstrap from prediction peaks
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = time.time() - (days * 86400)
    cursor.execute(
        """
        SELECT MAX(ctx_actual_cpu) AS peak_cpu,
               MAX(ctx_actual_ram) AS peak_ram,
               COUNT(*)            AS samples
        FROM prediction_logs
        WHERE lxc_id = ? AND timestamp >= ?
        """,
        (str(entity_id), cutoff),
    )
    row = cursor.fetchone()
    conn.close()

    if row and row["samples"]:
        return {
            "peak_cpu_pct": float(row["peak_cpu"] or 0.0),
            "peak_ram_mb":  float(row["peak_ram"] or 0.0),
            "sample_count": int(row["samples"]),
        }
    return {"peak_cpu_pct": 0.0, "peak_ram_mb": 0.0, "sample_count": 0}


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
