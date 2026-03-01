import logging
from config import MAX_HOST_CPU_ALLOCATION_PERCENT, MAX_HOST_RAM_ALLOCATION_PERCENT
from proxmox_api import ProxmoxClient

logger = logging.getLogger("scaler")


class Scaler:
    def __init__(self, proxmox_client: ProxmoxClient):
        self.px = proxmox_client

        # Buffer percentages applied to the prediction to ensure we don't scale
        # too tightly to the absolute exact predicted Mb/Cpu, leaving overhead.
        self.ram_buffer_percent = 20.0
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
        logger.info(
            f"[{entity_type} {entity_id}] Analyzing... Current State: "
            f"{current_metrics['allocated_cpus']} Cores, {current_metrics['allocated_ram_mb']} MB RAM. "
            f"Predicted Need: {predicted['cpu_percent']:.1f}% CPU, {predicted['ram_usage_mb']:.0f} MB RAM."
        )

        if not current_metrics:
            logger.warning(
                f"[{entity_type} {entity_id}] No current metrics to base scaling on. Skipping."
            )
            return

        # 1. Calculate the raw desired resources from the predictor
        #    Add an overhead buffer to the predicted peak
        desired_ram_mb = predicted["ram_usage_mb"] * (
            1 + self.ram_buffer_percent / 100.0
        )

        # CPU predictor returns a percent (0-100+) of currently allocated cores based on recent telemetry.
        # This is trickier depending on how Proxmox reports CPU over multiple cores.
        # For simplicity, if predicted cpu percent > 80, we want another core.
        # If predicted cpu percent < 30, we can drop a core.
        # It's safer to base CPU scaling up on the current metrics + predicted trend rather than an exact math conversion here.
        desired_cpus = current_metrics["allocated_cpus"]

        # CPU scaling heuristic:
        if predicted["cpu_percent"] > 85.0:
            desired_cpus += 1
        elif predicted["cpu_percent"] < 25.0 and current_metrics["cpu_percent"] < 25.0:
            desired_cpus -= 1

        # 2. Bound against configured baselines (min/max for this entity)
        target_ram = max(
            baseline["min_ram_mb"], min(int(desired_ram_mb), baseline["max_ram_mb"])
        )
        target_cpus = max(baseline["min_cpus"], min(desired_cpus, baseline["max_cpus"]))

        # Proxmox hotplug mechanism requires at least 1024 MB for VMs, and hot-unplug is generally unreliable
        if entity_type == "VM":
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

        # 4. Apply changes if different from currently allocated
        # We also check if the change is significant enough to warrant an API call (e.g., +/- 128 MB RAM, or any CPU change)
        ram_diff = abs(target_ram - current_metrics["allocated_ram_mb"])

        if target_cpus != current_metrics["allocated_cpus"] or ram_diff >= 64:
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
                f"RAM: {ram_action} to {target_ram} MB (was {current_metrics['allocated_ram_mb']} MB)"
            )

            if entity_type == "LXC":
                self.px.update_lxc_resources(entity_id, target_cpus, target_ram)
            elif entity_type == "VM":
                self.px.update_vm_resources(entity_id, target_cpus, target_ram)
        else:
            logger.debug(
                f"[{entity_type} {entity_id}] Resources adequate, no significant scaling required."
            )
