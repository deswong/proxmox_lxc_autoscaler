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

    # scale_events: records every actual resource change the autoscaler makes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scale_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      REAL    NOT NULL,
            entity_id      TEXT    NOT NULL,
            entity_type    TEXT    NOT NULL,
            action         TEXT    NOT NULL,
            trigger        TEXT    NOT NULL,
            cpus_before    REAL,
            cpus_after     REAL,
            ram_before_mb  REAL,
            ram_after_mb   REAL,
            swap_before_mb REAL,
            swap_after_mb  REAL,
            cpu_delta      REAL,
            ram_delta_mb   REAL
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scale_entity_time
        ON scale_events(entity_id, timestamp)
    """)

    conn.commit()
    conn.close()

    _seed_initial_baselines()


def log_scale_event(
    entity_id: str,
    entity_type: str,
    cpus_before: float,
    cpus_after: float,
    ram_before_mb: float,
    ram_after_mb: float,
    trigger: str = "prediction",
    swap_before_mb: float = 0.0,
    swap_after_mb: float = 0.0,
):
    """
    Records every actual resource change the autoscaler makes to an entity.

    ``trigger`` should be one of:
      - ``"prediction"``          — LXC live scale driven by LightGBM forecast
      - ``"host_pressure"``       — LXC reclaim driven by host RAM/CPU stress
      - ``"vm_pending_config"``   — VM config written for next reboot

    The ``action`` column is derived automatically:
      ``scale_up``, ``scale_down``, ``vm_pending_config``, or ``no_change``.
    """
    cpu_delta = cpus_after - cpus_before
    ram_delta = ram_after_mb - ram_before_mb

    if entity_type == "VM":
        action = "vm_pending_config"
    elif cpu_delta > 0 or ram_delta > 0:
        action = "scale_up"
    elif cpu_delta < 0 or ram_delta < 0:
        action = "scale_down"
    else:
        return  # Nothing changed — don't clutter the log

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO scale_events
            (timestamp, entity_id, entity_type, action, trigger,
             cpus_before, cpus_after, ram_before_mb, ram_after_mb,
             swap_before_mb, swap_after_mb, cpu_delta, ram_delta_mb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(), str(entity_id), entity_type, action, trigger,
            cpus_before, cpus_after, ram_before_mb, ram_after_mb,
            swap_before_mb, swap_after_mb, cpu_delta, ram_delta,
        ),
    )
    conn.commit()
    conn.close()


def get_performance_summary(days: int = 1) -> dict:
    """
    Returns a structured performance report covering the last ``days`` days.

    Structure::

        {
            "period_days": int,
            "scale_events": {
                "total": int,
                "scale_up_count": int,
                "scale_down_count": int,
                "vm_pending_count": int,
                "host_pressure_count": int,
                "net_ram_freed_mb": float,    # positive = freed, negative = allocated
                "net_cpu_cores_delta": float, # negative = freed cores
            },
            "prediction_accuracy": [
                {"entity_id": str, "mae_cpu_pct": float, "mae_ram_mb": float, "samples": int}
            ],
        }
    """
    cutoff = time.time() - (days * 86400)
    conn = get_db_connection()
    cursor = conn.cursor()

    # --- Scale event summary ---
    cursor.execute(
        """
        SELECT action, trigger,
               SUM(ram_delta_mb)  AS total_ram_delta,
               SUM(cpu_delta)     AS total_cpu_delta,
               COUNT(*)           AS cnt
        FROM scale_events
        WHERE timestamp >= ?
        GROUP BY action, trigger
        """,
        (cutoff,),
    )
    rows = cursor.fetchall()

    scale_up = scale_down = vm_pending = host_pressure = 0
    net_ram = net_cpu = 0.0
    for r in rows:
        if r["action"] == "scale_up":
            scale_up += r["cnt"]
        elif r["action"] == "scale_down":
            scale_down += r["cnt"]
        elif r["action"] == "vm_pending_config":
            vm_pending += r["cnt"]
        if r["trigger"] == "host_pressure":
            host_pressure += r["cnt"]
        net_ram += (r["total_ram_delta"] or 0.0)
        net_cpu += (r["total_cpu_delta"] or 0.0)

    # --- Prediction accuracy per entity ---
    cursor.execute(
        """
        SELECT lxc_id,
               AVG(ABS(predicted_cpu - ctx_actual_cpu))  AS mae_cpu,
               AVG(ABS(predicted_ram - ctx_actual_ram))  AS mae_ram,
               COUNT(*) AS samples
        FROM prediction_logs
        WHERE timestamp >= ?
          AND ctx_actual_cpu IS NOT NULL
          AND ctx_actual_ram IS NOT NULL
        GROUP BY lxc_id
        ORDER BY mae_ram DESC
        """,
        (cutoff,),
    )
    accuracy = [
        {
            "entity_id": r["lxc_id"],
            "mae_cpu_pct": round(r["mae_cpu"] or 0.0, 2),
            "mae_ram_mb":  round(r["mae_ram"]  or 0.0, 1),
            "samples":     r["samples"],
        }
        for r in cursor.fetchall()
    ]

    conn.close()

    return {
        "period_days": days,
        "scale_events": {
            "total":              scale_up + scale_down + vm_pending,
            "scale_up_count":     scale_up,
            "scale_down_count":   scale_down,
            "vm_pending_count":   vm_pending,
            "host_pressure_count": host_pressure,
            "net_ram_freed_mb":   round(-net_ram, 1),   # invert: negative delta = freed
            "net_cpu_cores_delta": round(net_cpu, 2),
        },
        "prediction_accuracy": accuracy,
    }


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


def get_prediction_errors(entity_id: str, days: int = 14) -> dict:
    """
    Returns a per-minute-bucket error scale factor derived from the gap between
    predictions and actual observed usage in ``prediction_logs``.

    For each logged cycle where both a predicted value AND an actual reading
    (``ctx_actual_cpu``, ``ctx_actual_ram``) are stored, this computes:

        cpu_err  = |predicted_cpu  - ctx_actual_cpu|   (percentage points)
        ram_err  = |predicted_ram  - ctx_actual_ram|   (MB)

    Both errors are normalised to [0, 1] over the window, summed, then
    mapped to a **penalty multiplier** in the range [1.0, 3.0]:

        penalty = 1.0 + 2.0 × norm_error    (so 0 error → 1×, max error → 3×)

    The trainer multiplies the time-based sample weight for each training row
    by the penalty for the nearest minute bucket, causing XGBoost to train
    harder on time windows where it previously made large errors.

    Returns a ``{minute_timestamp: penalty_multiplier}`` dict. Returns an
    empty dict when no telemetry has been logged yet (day-one bootstrap).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cutoff = time.time() - (days * 86400)
    cursor.execute(
        """
        SELECT timestamp,
               ABS(predicted_cpu - ctx_actual_cpu) AS cpu_err,
               ABS(predicted_ram - ctx_actual_ram) AS ram_err
        FROM prediction_logs
        WHERE lxc_id = ?
          AND timestamp >= ?
          AND ctx_actual_cpu IS NOT NULL
          AND ctx_actual_ram IS NOT NULL
        ORDER BY timestamp
        """,
        (str(entity_id), cutoff),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {}

    cpu_errors = [r["cpu_err"] or 0.0 for r in rows]
    ram_errors = [r["ram_err"] or 0.0 for r in rows]

    max_cpu = max(cpu_errors) or 1.0
    max_ram = max(ram_errors) or 1.0

    result = {}
    for r, cpu_e, ram_e in zip(rows, cpu_errors, ram_errors):
        norm = (cpu_e / max_cpu + ram_e / max_ram) / 2.0      # [0, 1]
        penalty = 1.0 + 2.0 * norm                             # [1.0, 3.0]
        # Key by nearest minute bucket so it aligns with RRD timestamps
        minute_ts = int(r["timestamp"]) - (int(r["timestamp"]) % 60)
        result[minute_ts] = max(result.get(minute_ts, 1.0), penalty)

    return result


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
