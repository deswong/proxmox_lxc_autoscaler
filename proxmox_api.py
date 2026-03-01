import logging
from proxmoxer import ProxmoxAPI
import urllib3
from config import (
    PROXMOX_HOST,
    PROXMOX_USER,
    PROXMOX_TOKEN_ID,
    PROXMOX_TOKEN_SECRET,
    NODE_NAME,
)

# Suppress insecure request warnings if Proxmox uses self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("proxmox_api")


class ProxmoxClient:
    def __init__(self):
        try:
            self.proxmox = ProxmoxAPI(
                PROXMOX_HOST,
                user=PROXMOX_USER,
                token_name=PROXMOX_TOKEN_ID,
                token_value=PROXMOX_TOKEN_SECRET,
                verify_ssl=False,
                timeout=30,
            )
            self.node = self.proxmox.nodes(NODE_NAME)
            logger.info(
                f"Successfully connected to Proxmox node {NODE_NAME} at {PROXMOX_HOST}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Proxmox API: {e}")
            self.proxmox = None

    def get_host_usage(self) -> dict:
        """Fetches the current CPU and RAM usage of the host node."""
        if not self.proxmox:
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "total_ram_mb": 0.0}

        try:
            status = self.node.status.get()

            # memory
            mem_total = status.get("memory", {}).get("total", 0)
            mem_used = status.get("memory", {}).get("used", 0)
            ram_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

            # cpu
            cpu_percent = status.get("cpu", 0) * 100

            return {
                "cpu_percent": float(cpu_percent),
                "ram_percent": float(ram_percent),
                "total_ram_mb": mem_total / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Failed to fetch host usage: {e}")
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "total_ram_mb": 0.0}

    def update_lxc_resources(self, lxc_id: str, cpus: int, ram_mb: int):
        """Updates the CPU cores and RAM allocation of a running LXC."""
        if not self.proxmox:
            return False

        import time

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # We hotplug the CPU and Memory via the API.
                # Proxmox config API expects memory in MB
                self.node.lxc(lxc_id).config.put(cores=int(cpus), memory=int(ram_mb))
                logger.info(
                    f"[LXC {lxc_id}] Successfully hotplugged resources: {cpus} cores, {ram_mb} MB RAM"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to update resources for LXC {lxc_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return False

    def get_lxc_rrd_history(self, lxc_id: str, timeframe: str = "hour") -> list:
        """
        Fetches the RRD historical graph data for an LXC.
        timeframe can be: hour, day, week, month, year.
        Returns a list of dicts: [{'time': UNIX_EPOCH, 'cpu': 0.05, 'mem': bytes, ...}]
        """
        if not self.proxmox:
            return []

        try:
            # Proxmox API returns an array of data points for the given timeframe.
            rrd_data = self.node.lxc(lxc_id).rrddata.get(timeframe=timeframe)
            return rrd_data
        except Exception as e:
            logger.error(f"Failed to fetch RRD history for LXC {lxc_id}: {e}")
            return []

    def get_all_lxc_metrics(self) -> dict:
        """
        Returns a dictionary mapping LXC IDs to their parsed current telemetry.
        Eliminates the need to sequentially API ping every single LXC for baseline status.
        """
        if not self.proxmox:
            return {}

        try:
            lxcs = self.node.lxc.get()
            metrics_dict = {}
            for lxc in lxcs:
                if lxc.get("status") != "running":
                    continue

                vmid = str(lxc["vmid"])
                metrics_dict[vmid] = {
                    "cpu_percent": float(lxc.get("cpu", 0) * 100),
                    "ram_usage_mb": float(lxc.get("mem", 0) / (1024 * 1024)),
                    "allocated_cpus": int(lxc.get("cpus", 1)),
                    "allocated_ram_mb": float(lxc.get("maxmem", 0) / (1024 * 1024)),
                    "uptime": int(lxc.get("uptime", 0)),
                }
            return metrics_dict
        except Exception as e:
            logger.error(
                f"Failed to fetch bulk telemetry for LXCs from node {NODE_NAME}: {e}"
            )
            return {}

    def update_vm_resources(self, vm_id: str, cpus: int, ram_mb: int):
        """Updates the CPU cores and RAM allocation of a running VM."""
        if not self.proxmox:
            return False

        import time

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Requires Hotplug to be enabled in Proxmox VM Hardware configs
                self.node.qemu(vm_id).config.put(cores=int(cpus), memory=int(ram_mb))
                logger.info(
                    f"[VM {vm_id}] Successfully hotplugged resources: {cpus} cores, {ram_mb} MB RAM"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to update resources for VM {vm_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return False

    def get_vm_rrd_history(self, vm_id: str, timeframe: str = "hour") -> list:
        """
        Fetches the RRD historical graph data for a VM.
        """
        if not self.proxmox:
            return []

        try:
            rrd_data = self.node.qemu(vm_id).rrddata.get(timeframe=timeframe)
            return rrd_data
        except Exception as e:
            logger.error(f"Failed to fetch RRD history for VM {vm_id}: {e}")
            return []

    def get_all_vm_metrics(self) -> dict:
        """
        Returns a dictionary mapping VM IDs to their parsed current telemetry.
        """
        if not self.proxmox:
            return {}

        try:
            vms = self.node.qemu.get()
            metrics_dict = {}
            for vm in vms:
                if vm.get("status") != "running":
                    continue

                vmid = str(vm["vmid"])
                metrics_dict[vmid] = {
                    "cpu_percent": float(vm.get("cpu", 0) * 100),
                    "ram_usage_mb": float(vm.get("mem", 0) / (1024 * 1024)),
                    "allocated_cpus": int(vm.get("cpus", 1)),
                    "allocated_ram_mb": float(vm.get("maxmem", 0) / (1024 * 1024)),
                    "uptime": int(vm.get("uptime", 0)),
                }
            return metrics_dict
        except Exception as e:
            logger.error(
                f"Failed to fetch bulk telemetry for VMs from node {NODE_NAME}: {e}"
            )
            return {}
