import os
import logging
import logging.handlers
from dotenv import load_dotenv

load_dotenv()

# Setup logging
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxmox_autoscaler")
logger.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Attempt to configure Rotating File Handler for /var/log
LOG_FILE_PATH = "/var/log/proxmox_ai_autoscaler.log"
try:
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except PermissionError:
    logger.warning(
        f"Permission denied to write to {LOG_FILE_PATH}. Logging to console only until permissions are fixed by install script."
    )
except Exception as e:
    logger.error(f"Failed to setup file logging at {LOG_FILE_PATH}: {e}")

# Map logger globally for this module
logger = logging.getLogger("config")

# Proxmox Credentials
PROXMOX_HOST = os.getenv("PROXMOX_HOST", "127.0.0.1")
PROXMOX_USER = os.getenv("PROXMOX_USER", "root@pam")
PROXMOX_TOKEN_ID = os.getenv("PROXMOX_TOKEN_ID", "autoscaler")
PROXMOX_TOKEN_SECRET = os.getenv("PROXMOX_TOKEN_SECRET", "")
NODE_NAME = os.getenv("NODE_NAME", "pve")

# Scaling Settings
MAX_HOST_CPU_ALLOCATION_PERCENT = min(
    float(os.getenv("MAX_HOST_CPU_ALLOCATION_PERCENT", 85.0)), 95.0
)
MAX_HOST_RAM_ALLOCATION_PERCENT = min(
    float(os.getenv("MAX_HOST_RAM_ALLOCATION_PERCENT", 85.0)), 95.0
)

# Training Settings
TRAINING_DAYS_LOOKBACK = int(os.getenv("TRAINING_DAYS_LOOKBACK", 7))

# Initial Baselines
INITIAL_LXC_CONFIGS = {}
INITIAL_VM_CONFIGS = {}

for key, value in os.environ.items():
    if key.startswith("LXC_") or key.startswith("VM_"):
        try:
            parts = value.split(",")
            if len(parts) == 4:
                prefix, entity_id = key.split("_", 1)
                config_dict = {
                    "min_cpus": int(parts[0].strip()),
                    "min_ram_mb": int(parts[1].strip()),
                    "max_cpus": int(parts[2].strip()),
                    "max_ram_mb": int(parts[3].strip()),
                }
                if prefix == "LXC":
                    INITIAL_LXC_CONFIGS[entity_id] = config_dict
                else:
                    INITIAL_VM_CONFIGS[entity_id] = config_dict
            else:
                logger.warning(
                    f"Skipping {key}: Must have exactly 4 values (min_cpu, min_ram, max_cpu, max_ram)"
                )
        except Exception as e:
            logger.error(
                f"Failed to parse environment variable {key}={value}. Error: {e}"
            )

# Excluded Containers and VMs
# Comma-separated list of IDs to never autoscale
_excluded_lxc_str = os.getenv("EXCLUDED_LXCS", "")
EXCLUDED_LXCS = [x.strip() for x in _excluded_lxc_str.split(",") if x.strip()]

_excluded_vm_str = os.getenv("EXCLUDED_VMS", "")
EXCLUDED_VMS = [x.strip() for x in _excluded_vm_str.split(",") if x.strip()]

DATABASE_PATH = os.getenv("DATABASE_PATH", "autoscaler.db")
POLL_INTERVAL_SECONDS = 60
