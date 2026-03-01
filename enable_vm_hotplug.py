import sys
import logging
from proxmox_api import ProxmoxClient
from config import EXCLUDED_VMS

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - HOTPLUG - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enable_vm_hotplug")


def enable_hotplug():
    px_client = ProxmoxClient()
    if not px_client.proxmox:
        logger.error("Failed to connect to Proxmox API.")
        sys.exit(1)

    logger.info("Discovering all VMs on the node...")
    # Fetch all VMs to get their IDs
    all_vm_metrics = px_client.get_all_vm_metrics()

    if not all_vm_metrics:
        logger.warning("No running VMs found to configure.")
        sys.exit(0)

    success_count = 0
    skipped_count = 0
    error_count = 0

    for vm_id in all_vm_metrics.keys():
        if vm_id in EXCLUDED_VMS:
            logger.info(f"Skipping VM {vm_id} (Listed in EXCLUDED_VMS).")
            skipped_count += 1
            continue

        logger.info(f"Attempting to enable hotplug for VM {vm_id}...")

        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # The 'hotplug' property in Proxmox accepts a string of options.
                # 'disk,network,usb,memory,cpu' enables it for all commonly hot-pluggable hardware.
                px_client.node.qemu(vm_id).config.put(
                    hotplug="disk,network,usb,memory,cpu"
                )
                logger.info(
                    f"Successfully enabled hotplug (disk,network,usb,memory,cpu) for VM {vm_id}."
                )
                success_count += 1
                break
            except Exception as e:
                logger.error(
                    f"Failed to enable hotplug for VM {vm_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    error_count += 1

    logger.info(
        f"Hotplug Configuration Complete: {success_count} Success, {skipped_count} Skipped, {error_count} Errors."
    )


if __name__ == "__main__":
    enable_hotplug()
