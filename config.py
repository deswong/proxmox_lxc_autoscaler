import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

# Proxmox Credentials
PROXMOX_HOST = os.getenv("PROXMOX_HOST", "127.0.0.1")
PROXMOX_USER = os.getenv("PROXMOX_USER", "root@pam")
PROXMOX_TOKEN_ID = os.getenv("PROXMOX_TOKEN_ID", "autoscaler")
PROXMOX_TOKEN_SECRET = os.getenv("PROXMOX_TOKEN_SECRET", "")
NODE_NAME = os.getenv("NODE_NAME", "pve")

# Scaling Settings
MAX_HOST_CPU_ALLOCATION_PERCENT = float(os.getenv("MAX_HOST_CPU_ALLOCATION_PERCENT", 85.0))
MAX_HOST_RAM_ALLOCATION_PERCENT = float(os.getenv("MAX_HOST_RAM_ALLOCATION_PERCENT", 85.0))

# Initial Baselines
INITIAL_LXC_CONFIGS = {}
for key, value in os.environ.items():
    if key.startswith("LXC_") and len(key) > 4:
        try:
            lxc_id = key.split("_")[1]
            parts = value.split(",")
            if len(parts) == 4:
                INITIAL_LXC_CONFIGS[lxc_id] = {
                    "min_cpus": int(parts[0].strip()),
                    "min_ram_mb": int(parts[1].strip()),
                    "max_cpus": int(parts[2].strip()),
                    "max_ram_mb": int(parts[3].strip())
                }
            else:
                logger.warning(f"Skipping {key}: Must have exactly 4 values (min_cpu, min_ram, max_cpu, max_ram)")
        except Exception as e:
            logger.error(f"Failed to parse environment variable {key}={value}. Error: {e}")

DATABASE_PATH = os.getenv("DATABASE_PATH", "autoscaler.db")
POLL_INTERVAL_SECONDS = 60
