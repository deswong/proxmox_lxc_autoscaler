import logging
import storage
from config import (
    MAX_HOST_CPU_ALLOCATION_PERCENT,
    MAX_HOST_RAM_ALLOCATION_PERCENT,
    MAX_HOST_SWAP_USAGE_PERCENT,
    LXC_TARGET_SWAP_MB,
    LXC_MIN_SWAP_MB,
)
from proxmox_api import ProxmoxClient

logger = logging.getLogger("scaler")

# Host RAM % above which we actively push idle containers down to reclaim headroom.
HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD = 90.0
# A container is "idle relative to allocation" when it uses less than this fraction.
CONTAINER_IDLE_ALLOCATION_RATIO = 0.5
# Target: scale idle containers to this multiple of current usage (+ 30% buffer).
ACTIVE_SCALEDOWN_HEADROOM_RATIO = 1.5


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

        Host pressure is handled at three tiers:
          < threshold     — normal operation
          threshold–90%   — block scale-ups (existing safety cap)
          > 90% RAM       — block scale-ups AND actively scale down idle containers
        """

        if not current_metrics:
            logger.warning(
                f"[{entity_type} {entity_id}] No current metrics to base scaling on. Skipping."
            )
            return

        # Fetch live host node metrics early so they are available for all safety guards
        host_metrics = self.px.get_host_usage()
        host_ram_pct = host_metrics.get("ram_percent", 0.0)
        host_cpu_pct = host_metrics["cpu_percent"]
        host_swap_pct = host_metrics.get("swap_percent", 0.0)

        # Hardcoded emergency safeguards (caps user config at max 95%)
        safe_cpu_limit = min(MAX_HOST_CPU_ALLOCATION_PERCENT, 95.0)
        safe_ram_limit = min(MAX_HOST_RAM_ALLOCATION_PERCENT, 95.0)
        safe_swap_limit = min(MAX_HOST_SWAP_USAGE_PERCENT, 95.0)

        # 0. Pre-calculate swap status (Needed for RAM headroom planning)
        swap_used = 0.0
        swap_alloc = 0.0

        if entity_type == "LXC":
            swap_used = current_metrics.get("swap_mb", 0.0)
            swap_alloc = current_metrics.get("allocated_swap_mb", 0.0)

        # 1. Calculate the raw desired resources from the predictor.
        #    Use the higher of the ML forecast, the observed recent peak, or 
        #    the current actual usage to ensure we never under-allocate.
        current_usage_mb = current_metrics.get("ram_usage_mb", 0.0)
        peak_ram_mb = max(
            predicted["ram_usage_mb"], 
            predicted.get("recent_peak_ram", 0.0),
            current_usage_mb
        )

        # "Natural Reclaim" RAM Boost: If the container is actively using swap, we
        # MUST increase its RAM headroom to give the OS enough physical space
        # to naturally page those swap blocks back into RAM on its own schedule.
        if swap_used > 5 and entity_type == "LXC":
            needed_for_swap = (current_usage_mb + swap_used) * 1.15
            if peak_ram_mb < needed_for_swap:
                logger.debug(
                    f"[{entity_type} {entity_id}] Boosting target RAM to {needed_for_swap:.0f} MB "
                    f"to allow natural reclaim of {swap_used:.0f} MB swap."
                )
                peak_ram_mb = needed_for_swap

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
        # Apply a hard system floor (64MB LXC, 1024MB VM) to prevent OS crashes
        system_floor = 64 if entity_type == "LXC" else 1024
        target_ram = max(
            baseline["min_ram_mb"],
            system_floor,
            min(int(desired_ram_mb), baseline["max_ram_mb"]),
        )

        # Final Guard: Ensure we never shrink RAM if container is heavily swapped
        # Priority 1: Ensure enough headroom (usage + swap).
        if swap_used > 5 and entity_type == "LXC":
            # Ensure physical capacity for pages
            target_ram = max(target_ram, int(current_usage_mb + swap_used + 128))
            # Prevent scale-down unless host is in emergency state (>95%)
            if host_ram_pct < 95.0:
                target_ram = max(target_ram, int(current_metrics["allocated_ram_mb"]))

        target_cpus = max(baseline["min_cpus"], min(desired_cpus, baseline["max_cpus"]))

        # 3. Check physical node limits before scaling UP
        # Hardcoded emergency safeguard (caps user config at max 95%)

        # Apply Host Swap Safety Cap
        # If the host is heavily swapping, completely block all scale-ups to prevent
        # exacerbating an already memory-starved hypervisor.
        if (
            host_swap_pct > safe_swap_limit
            and (target_cpus > current_metrics["allocated_cpus"] or target_ram > current_metrics["allocated_ram_mb"])
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale up. Host Node Swap is over "
                f"threshold ({host_swap_pct:.1f}% > {safe_swap_limit}%)."
            )
            # Limit scale up to current allocation for both
            target_cpus = min(target_cpus, current_metrics["allocated_cpus"])
            target_ram = min(target_ram, current_metrics["allocated_ram_mb"])

        if (
            host_cpu_pct > safe_cpu_limit
            and target_cpus > current_metrics["allocated_cpus"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale CPU up. Host Node CPU is over "
                f"threshold ({host_cpu_pct:.1f}% > {safe_cpu_limit}%)."
            )
            # Limit scale up to current allocation
            target_cpus = current_metrics["allocated_cpus"]

        if (
            host_ram_pct > safe_ram_limit
            and target_ram > current_metrics["allocated_ram_mb"]
        ):
            logger.warning(
                f"[{entity_type} {entity_id}] SAFETY CAP: Cannot scale RAM up. Host Node RAM is over "
                f"threshold ({host_ram_pct:.1f}% > {safe_ram_limit}%)."
            )
            # But we can allow scaling down RAM, just not UP.
            target_ram = current_metrics["allocated_ram_mb"]

        # 3b. Active scale-down: when the host is critically RAM-stressed AND this
        #     container is genuinely idle relative to its allocation, reclaim headroom.
        #     Guard: both conditions must be true simultaneously so a busy container
        #     during a host-wide load event is never aggressively shrunk.
        if (
            entity_type == "LXC"
            and host_ram_pct > HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
        ):
            ram_usage = current_metrics.get("ram_usage_mb", 0.0)
            alloc_ram = current_metrics["allocated_ram_mb"]
            if (
                alloc_ram > 0
                and ram_usage / alloc_ram < CONTAINER_IDLE_ALLOCATION_RATIO
            ):
                # Container is using < 50% of its allocation while the host is struggling.
                # Nudge it down to usage × 1.5, floored at min_ram_mb.
                reclaimed_target = max(
                    int(ram_usage * ACTIVE_SCALEDOWN_HEADROOM_RATIO),
                    baseline["min_ram_mb"],
                )
                if reclaimed_target < alloc_ram:
                    logger.warning(
                        f"[{entity_type} {entity_id}] HOST PRESSURE RECLAIM: Host RAM at "
                        f"{host_ram_pct:.1f}% (>{HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD:.0f}%). "
                        f"Container idle ({ram_usage:.0f}/{alloc_ram:.0f} MB used). "
                        f"Actively reducing RAM to {reclaimed_target} MB to ease host pressure."
                    )
                    target_ram = reclaimed_target

        if entity_type == "LXC":
            if swap_used > 5:
                logger.info(
                    f"[LXC {entity_id}] Natural Reclaim active ({swap_used:.0f}/{swap_alloc:.0f} MB used). "
                    "Waiting for OS to page back to RAM."
                )

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
                target_swap = max(LXC_TARGET_SWAP_MB, LXC_MIN_SWAP_MB)
            
            # NATURAL RECLAIM "DO NO HARM" FLOOR:
            # Never set the swap limit lower than the active swap usage plus a 32MB buffer.
            # Why? Because lowering the cgroup limit below usage forces the Linux kernel into 
            # a synchronous reclaim (pausing the container, reading disk sequentially to RAM). 
            # This causes catastrophic I/O stalls in Proxmox. We MUST let the OS page it back 
            # gently on its own schedule.
            safe_floor = int(swap_used + 32)
            if target_swap < safe_floor and swap_used > 5:
                logger.debug(f"[LXC {entity_id}] Adjusting target swap from {target_swap} MB to safe floor {safe_floor} MB to prevent cgroup stall.")
                target_swap = safe_floor
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
            or swap_diff > 0
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
                trigger = (
                    "host_pressure"
                    if host_metrics.get("ram_percent", 0) > HOST_RAM_ACTIVE_SCALEDOWN_THRESHOLD
                    else "prediction"
                )
                try:
                    storage.log_scale_event(
                        entity_id=entity_id,
                        entity_type="LXC",
                        cpus_before=current_metrics["allocated_cpus"],
                        cpus_after=target_cpus,
                        ram_before_mb=current_metrics["allocated_ram_mb"],
                        ram_after_mb=target_ram,
                        trigger=trigger,
                        swap_before_mb=float(current_metrics.get("swap_mb", 0.0)),
                        swap_after_mb=float(target_swap),
                    )
                except Exception as log_err:
                    logger.debug(f"[LXC {entity_id}] Scale event log failed: {log_err}")
            elif entity_type == "VM":
                self.px.update_vm_resources(entity_id, target_cpus, target_ram)
        else:
            logger.debug(
                f"[{entity_type} {entity_id}] Resources adequate, no significant scaling required."
            )

    def apply_vm_pending_config(
        self,
        vm_id: str,
        baseline: dict,
        predicted: dict,
        current_metrics: dict,
        rolling_peaks: dict,
    ):
        """
        Computes the optimal CPU / RAM sizing for a VM using a 14-day rolling peak
        from the telemetry log plus a 30% safety headroom, then writes that as a
        *pending* Proxmox config entry. The change takes effect on the next reboot—
        no live hotplug is ever attempted.

        Sizing formula
        --------------
        peak_ram_mb  = MAX(rolling 14-day observed RAM, prediction recent_peak)
        peak_cpu_pct = MAX(rolling 14-day observed CPU%, prediction recent_peak_cpu)

        target_ram   = clamp(int(peak_ram * 1.30), max(min_ram_mb, 1024), max_ram_mb)
        needed_cores = int(peak_cpu_pct / 100 x current_cpus x 1.30) + 1
        target_cpus  = clamp(needed_cores, min_cpus, max_cpus)

        Config is only written when recommendation differs from current allocation
        by > 5% RAM or >= 1 CPU core.
        """
        if not current_metrics:
            logger.warning(f"[VM {vm_id}] No current metrics. Skipping pending config.")
            return

        sample_count = rolling_peaks.get("sample_count", 0)
        alloc_cpus   = current_metrics["allocated_cpus"]
        alloc_ram_mb = current_metrics["allocated_ram_mb"]

        # Fetch actual configuration from API to avoid logging changes that are already pending
        current_config = self.px.get_vm_config(vm_id)
        config_cpus = current_config.get("cpus", alloc_cpus)
        config_ram_mb = current_config.get("ram_mb", alloc_ram_mb)

        if sample_count > 0:
            # Primary path: real observed peaks from the telemetry log
            peak_ram_mb  = rolling_peaks["peak_ram_mb"]
            peak_cpu_pct = rolling_peaks["peak_cpu_pct"]
            source_label = f"{sample_count} telemetry samples"
        else:
            # Bootstrap: no log data yet (day one). Use the ML prediction peaks.
            peak_ram_mb  = max(
                predicted["ram_usage_mb"],
                predicted.get("recent_peak_ram", 0.0),
            )
            peak_cpu_pct = max(
                predicted["cpu_percent"],
                predicted.get("recent_peak_cpu", 0.0),
            )
            source_label = "ML prediction peaks (no log data yet)"

        logger.info(
            f"[VM {vm_id}] Rolling peaks ({source_label}): "
            f"{peak_ram_mb:.0f} MB RAM / {peak_cpu_pct:.1f}% CPU"
        )

        # Apply 30% headroom above peak
        headroom   = 1 + self.ram_buffer_percent / 100.0
        target_ram = int(peak_ram_mb * headroom)
        target_ram = max(target_ram, 1024)              # Proxmox VM floor
        target_ram = max(target_ram, baseline["min_ram_mb"])
        target_ram = min(target_ram, baseline["max_ram_mb"])

        needed_cores = int(
            (peak_cpu_pct / 100.0) * alloc_cpus * (1 + self.cpu_buffer_percent / 100.0)
        ) + 1  # +1 ensures at least one core always recommended
        target_cpus = max(baseline["min_cpus"], min(needed_cores, baseline["max_cpus"]))

        # Only write when change is significant compared to existing CONFIG
        ram_delta_pct = abs(target_ram - config_ram_mb) / max(config_ram_mb, 1) * 100
        cpu_changed   = target_cpus != config_cpus

        if ram_delta_pct < 5.0 and not cpu_changed:
            logger.debug(
                f"[VM {vm_id}] Pending config unchanged "
                f"(delta {ram_delta_pct:.1f}% RAM, CPU same). Skipping write."
            )
            return

        logger.info(
            f"[VM {vm_id}] PENDING CONFIG (applies on next reboot): "
            f"{target_cpus} CPUs (was {config_cpus}), "
            f"{target_ram} MB RAM (was {config_ram_mb:.0f} MB). "
            f"Basis: {source_label} — 14-day peak {peak_ram_mb:.0f} MB / "
            f"{peak_cpu_pct:.1f}% CPU + {self.ram_buffer_percent:.0f}% headroom."
        )
        self.px.update_vm_resources(vm_id, target_cpus, target_ram)
        try:
            storage.log_scale_event(
                entity_id=vm_id,
                entity_type="VM",
                cpus_before=float(alloc_cpus),
                cpus_after=float(target_cpus),
                ram_before_mb=float(alloc_ram_mb),
                ram_after_mb=float(target_ram),
                trigger="vm_pending_config",
            )
        except Exception as log_err:
            logger.debug(f"[VM {vm_id}] Scale event log failed: {log_err}")
