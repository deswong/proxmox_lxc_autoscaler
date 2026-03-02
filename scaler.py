import logging
from config import (
    MAX_HOST_CPU_ALLOCATION_PERCENT,
    MAX_HOST_RAM_ALLOCATION_PERCENT,
    LXC_TARGET_SWAP_MB,
    LXC_MIN_SWAP_MB,
    SWAP_FLUSH_THRESHOLD_PERCENT,
)
from proxmox_api import ProxmoxClient

logger = logging.getLogger("scaler")


class Scaler:
    def __init__(self, proxmox_client: ProxmoxClient):
        self.px = proxmox_client

        # Buffer percentages applied to the prediction to ensure we don't scale
        # too tightly to the absolute exact predicted Mb/Cpu, leaving overhead.
        self.ram_buffer_percent = 30.0
        self.cpu_buffer_percent = 20.0

    def evaluate_and_scale(
        self,
        entity_id: str,
        entity_type: str,
        baseline: dict,
        predicted: dict,
        current_metrics: dict,
    ):
        """
        Evaluates predictions against max/min baselines and overall node health.
        Triggers a scaling action if requirements change.
        """

        if not current_metrics:
            logger.warning(
                f"[{entity_type} {entity_id}] No current metrics to base scaling on. Skipping."
            )
            return

        # 1. Calculate the raw desired resources from the predictor.
        #    Use the higher of the ML forecast or the observed recent peak so
        #    that genuine RAM spikes are covered, not just the smoothed average.
        #    Then apply an overhead buffer on top of that peak value.
        peak_ram_mb = max(
            predicted["ram_usage_mb"], predicted.get("recent_peak_ram", 0.0)
        )
        desired_ram_mb = peak_ram_mb * (1 + self.ram_buffer_percent / 100.0)

        # CPU scaling heuristic - proportional to the predicted load:
        # Scale UP: add 1 core for every 15% above the 85% high-water mark
        # Scale DOWN: drop 1 core for every 30% below the 25% low-water mark
        desired_cpus = current_metrics["allocated_cpus"]
        if predicted["cpu_percent"] > 85.0:
            overshoot = predicted["cpu_percent"] - 85.0
            cores_to_add = max(1, int(overshoot / 15))
            desired_cpus += cores_to_add
        elif predicted["cpu_percent"] < 25.0 and current_metrics["cpu_percent"] < 25.0:
            undershoot = 25.0 - predicted["cpu_percent"]
            cores_to_remove = max(1, int(undershoot / 30))
            desired_cpus = max(1, desired_cpus - cores_to_remove)

        logger.info(
            f"[{entity_type} {entity_id}] Analyzing... Current State: "
            f"{current_metrics['allocated_cpus']} Cores, {current_metrics['allocated_ram_mb']} MB RAM. "
            f"Predicted Need: {desired_cpus} Cores ({predicted['cpu_percent']:.1f}%), "
            f"{predicted['ram_usage_mb']:.0f} MB RAM."
        )

        # 2. Bound against configured baselines (min/max for this entity)
        target_ram = max(
            baseline["min_ram_mb"], min(int(desired_ram_mb), baseline["max_ram_mb"])
        )
        target_cpus = max(baseline["min_cpus"], min(desired_cpus, baseline["max_cpus"]))

        # Proxmox hotplug mechanism requires at least 1024 MB for VMs, and hot-unplug is generally unreliable
        if entity_type == "VM":
            if target_cpus < current_metrics["allocated_cpus"]:
                # The prediction wanted to scale down CPU, but we block it
                logger.info(
                    f"[{entity_type} {entity_id}] VM CPU hot-unplug is generally unsupported by guest OS. Preventing CPU scale-down."
                )
                target_cpus = current_metrics["allocated_cpus"]

            if target_ram < 1024:
                if (
                    target_ram != current_metrics["allocated_ram_mb"]
                    and current_metrics["allocated_ram_mb"] >= 1024
                ):
                    logger.info(
                        f"[{entity_type} {entity_id}] Enforcing Proxmox VM hotplug minimum of 1024 MB RAM."
                    )
                target_ram = 1024
            if target_ram < current_metrics["allocated_ram_mb"]:
                # The prediction wanted to scale down RAM, but we block it
                logger.info(
                    f"[{entity_type} {entity_id}] VM Memory hot-unplug is unreliable. Preventing RAM scale-down."
                )
                target_ram = current_metrics["allocated_ram_mb"]

        # 3. Check physical node limits before scaling UP
        # Fetch live host node metrics
        host_metrics = self.px.get_host_usage()

        # Hardcoded emergency safeguard (caps user config at max 95%)
        safe_cpu_limit = min(MAX_HOST_CPU_ALLOCATION_PERCENT, 95.0)
        safe_ram_limit = min(MAX_HOST_RAM_ALLOCATION_PERCENT, 95.0)

        if (
            host_metrics["cpu_percent"] > safe_cpu_limit
            and target_cpus > current_metrics["allocated_cpus"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale CPU up. Host Node CPU is over "
                f"threshold ({host_metrics['cpu_percent']:.1f}% > {safe_cpu_limit}%)."
            )
            # Limit scale up to current allocation
            target_cpus = current_metrics["allocated_cpus"]

        if (
            host_metrics["ram_percent"] > safe_ram_limit
            and target_ram > current_metrics["allocated_ram_mb"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale RAM up. Host Node RAM is over "
                f"threshold ({host_metrics['ram_percent']:.1f}% > {safe_ram_limit}%)."
            )
            # But we can allow scaling down RAM, just not UP.
            target_ram = current_metrics["allocated_ram_mb"]

        # 4. Detect swap saturation on LXCs and schedule a post-scale flush.
        #    High swap means the container is IO-bound on disk; flushing it
        #    after a RAM scale-up reclaims pages back into the newly freed RAM.
        flush_swap = False
        if entity_type == "LXC":
            swap_used = current_metrics.get("swap_mb", 0.0)
            swap_alloc = current_metrics.get("allocated_swap_mb", 0.0)
            if swap_alloc > 0 and (swap_used / swap_alloc * 100) > SWAP_FLUSH_THRESHOLD_PERCENT:
                logger.warning(
                    f"[LXC {entity_id}] Swap saturation detected "
                    f"({swap_used:.0f}/{swap_alloc:.0f} MB used). "
                    "Will flush swap after scale-up."
                )
                flush_swap = True
            elif swap_used > 0 and swap_alloc == 0:
                # Residual swap from before the autoscaler took over
                logger.warning(
                    f"[LXC {entity_id}] Residual active swap detected "
                    f"({swap_used:.0f} MB). Will flush."
                )
                flush_swap = True

        # 5. Compute the target swap cap for this LXC.
        #    Auto mode (-1): size swap like RAM — use observed peak + 30% buffer,
        #    floored at LXC_MIN_SWAP_MB so no container is ever left fully swapless
        #    during the model cold-start period.
        if entity_type == "LXC":
            if LXC_TARGET_SWAP_MB == -1:
                peak_swap = max(
                    predicted.get("predicted_swap_mb", 0.0),
                    predicted.get("recent_peak_swap", 0.0),
                )
                target_swap = max(
                    int(peak_swap * (1 + self.ram_buffer_percent / 100.0)),
                    LXC_MIN_SWAP_MB,
                )
            else:
                target_swap = max(LXC_TARGET_SWAP_MB, 0)
        else:
            target_swap = 0  # VMs manage swap internally; we don't set this

        # 6. Apply changes if different from currently allocated.
        #    Triggers on: CPU change, RAM change (>=32 MB), swap cap change (>=32 MB),
        #    or swap saturation detected — whichever comes first.
        ram_diff = abs(target_ram - current_metrics["allocated_ram_mb"])
        swap_diff = abs(target_swap - current_metrics.get("allocated_swap_mb", 0.0))

        if (
            target_cpus != current_metrics["allocated_cpus"]
            or ram_diff >= 32
            or swap_diff >= 32
            or flush_swap
        ):
            cpu_action = "UNCHANGED"
            if target_cpus > current_metrics["allocated_cpus"]:
                cpu_action = "UP"
            elif target_cpus < current_metrics["allocated_cpus"]:
                cpu_action = "DOWN"

            ram_action = "UNCHANGED"
            if target_ram > current_metrics["allocated_ram_mb"]:
                ram_action = "UP"
            elif target_ram < current_metrics["allocated_ram_mb"]:
                ram_action = "DOWN"

            logger.info(
                f"[{entity_type} {entity_id}] Scaling Required. "
                f"CPU: {cpu_action} to {target_cpus} (was {current_metrics['allocated_cpus']}), "
                f"RAM: {ram_action} to {target_ram} MB (was {current_metrics['allocated_ram_mb']} MB), "
                f"Swap: {target_swap} MB"
            )

            if entity_type == "LXC":
                self.px.update_lxc_resources(
                    entity_id, target_cpus, target_ram, swap_mb=target_swap
                )
                if flush_swap:
                    # Safe-flush guard: only call swapoff if RAM headroom can
                    # physically absorb the in-swap pages without triggering OOM.
                    ram_headroom = target_ram - current_metrics.get("ram_usage_mb", 0.0)
                    swap_used = current_metrics.get("swap_mb", 0.0)
                    if ram_headroom >= swap_used * 1.1:
                        self.px.flush_lxc_swap(entity_id)
                    else:
                        logger.warning(
                            f"[LXC {entity_id}] Skipping swap flush: insufficient RAM headroom "
                            f"({ram_headroom:.0f} MB available, {swap_used:.0f} MB in swap). "
                            "Will retry after RAM scales up further."
                        )
            elif entity_type == "VM":
                self.px.update_vm_resources(entity_id, target_cpus, target_ram)
        else:
            logger.debug(
                f"[{entity_type} {entity_id}] Resources adequate, no significant scaling required."
            )
