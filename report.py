#!/usr/bin/env python3
"""
report.py — Proxmox AI Autoscaler Performance Reporter

Queries the local autoscaler.db telemetry database and prints a structured
performance summary showing scaling activity, resource savings, and
prediction accuracy trends.

Usage:
    python report.py               # last 24 hours (default)
    python report.py --days 7      # last 7 days
    python report.py --days 30     # last 30 days
    python report.py --json        # machine-readable JSON output
"""

import argparse
import json
import sys
import storage


def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Returns a simple ASCII progress bar."""
    if max_value <= 0:
        return " " * width
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(days: int = 1):
    """Prints a human-readable performance report to stdout."""
    storage.init_db()
    data = storage.get_performance_summary(days=days)
    ev   = data["scale_events"]
    acc  = data["prediction_accuracy"]

    period_label = f"Last {days} day{'s' if days != 1 else ''}"

    print()
    print("═" * 60)
    print("  📊 Proxmox AI Autoscaler — Performance Report")
    print(f"  Period: {period_label}")
    print("═" * 60)

    # --- Scaling activity ---
    print()
    print("  ┌─ Scaling Activity ──────────────────────────────────")
    total = ev["total"]
    if total == 0:
        print("  │  No scaling events recorded in this period.")
    else:
        up   = ev["scale_up_count"]
        down = ev["scale_down_count"]
        vm   = ev["vm_pending_count"]
        hp   = ev["host_pressure_count"]
        print(f"  │  Total events   : {total}")
        print(f"  │  Scale-ups      : {up} events")
        print(f"  │  Scale-downs    : {down} events")
        if vm:
            print(f"  │  VM pending cfg : {vm} updates (apply on next reboot)")
        if hp:
            print(f"  │  Host pressure  : {hp} events drove reclaim actions")

    # --- Resource savings ---
    print()
    print("  ┌─ Resource Savings (Realized - LXC) ─────────────────")
    freed_ram = ev["net_ram_freed_mb"]
    cpu_delta = ev["net_cpu_cores_delta"]

    if freed_ram > 0:
        print(f"  │  Net RAM freed  : {freed_ram:.0f} MB  ({freed_ram/1024:.2f} GB)")
    elif freed_ram < 0:
        print(f"  │  Net RAM added  : {-freed_ram:.0f} MB  ({-freed_ram/1024:.2f} GB)")
    else:
        print("  │  Net RAM change : 0 MB")

    if cpu_delta < 0:
        print(f"  │  Net CPU freed  : {-cpu_delta:.1f} vCPU cores")
    elif cpu_delta > 0:
        print(f"  │  Net CPU added  : +{cpu_delta:.1f} vCPU cores")
    else:
        print("  │  Net CPU change : 0 cores")

    net_swap = ev["net_swap_freed_mb"]
    if net_swap > 0:
        print(f"  │  Net Swap freed : {net_swap:.0f} MB")
    elif net_swap < 0:
        print(f"  │  Net Swap added : {-net_swap:.0f} MB")
    else:
        print("  │  Net Swap change: 0 MB")

    if ev["vm_pending_count"] > 0:
        print("  │")
        print("  ├─ Potential Savings (Pending Reboot - VM) ───────────")
        p_ram = ev["potential_ram_freed_mb"]
        p_cpu = ev["potential_cpu_cores_delta"]
        
        if p_ram > 0:
            print(f"  │  Est. RAM freed : {p_ram:.0f} MB  ({p_ram/1024:.2f} GB)")
        elif p_ram < 0:
            print(f"  │  Est. RAM added : {-p_ram:.0f} MB  ({-p_ram/1024:.2f} GB)")
            
        if p_cpu < 0:
            print(f"  │  Est. CPU freed : {-p_cpu:.1f} vCPU cores")
        elif p_cpu > 0:
            print(f"  │  Est. CPU added : +{p_cpu:.1f} vCPU cores")

    # --- Prediction accuracy ---
    print()
    print("  ┌─ Prediction Accuracy (MAE per entity) ──────────────")
    if not acc:
        print("  │  No telemetry logged yet.")
        print("  │  Accuracy tracking begins after the first live inference cycle.")
    else:
        max_mae_ram = max(a["mae_ram_mb"] for a in acc) or 1.0
        print(f"  │  {'Entity':<12} {'CPU MAE':>10}  {'RAM MAE':>10}  {'Samples':>8}")
        print("  │  " + "─" * 46)
        for a in acc:
            mae_bar = _bar(a["mae_ram_mb"], max_mae_ram, width=12)
            print(
                f"  │  {a['entity_id']:<12}  "
                f"{a['mae_cpu_pct']:>7.1f}%   "
                f"{a['mae_ram_mb']:>7.0f} MB  "
                f"{a['samples']:>8d}   {mae_bar}"
            )

    print()
    print("═" * 60)
    print()


def print_json(days: int = 1):
    """Prints a machine-readable JSON performance report to stdout."""
    storage.init_db()
    data = storage.get_performance_summary(days=days)
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Proxmox AI Autoscaler performance report"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to include in the report (default: 1)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of formatted text"
    )
    args = parser.parse_args()

    if args.days < 1:
        print("Error: --days must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print_json(days=args.days)
    else:
        print_report(days=args.days)


if __name__ == "__main__":
    main()
