# Proxmox Universal AI Autoscaler 🚀

**Stop reacting to server crashes. Start predicting them.**

A lightweight, purely Python-based service that brings proactive AI autoscaling natively to your **Proxmox Virtual Machines (QEMU/KVM)** and **Linux Containers (LXC)**.

Instead of waiting for your servers to hit 100% saturation and stall, this daemon polls Proxmox's native historical RRD telemetry APIs and uses a lightning-fast **XGBoost engine** to forecast computing trends. It intelligently hotplugs CPU cores and RAM allocations *before* the spike hits, all while strictly enforcing host-level limiters to ensure you never starve your hypervisor.

Zero database tuning, zero complex Prometheus stacks—just one service keeping your Proxmox instances fast and your host safe.

## Features
- **Proactive Scaling**: Looks up to 2 minutes into the future to predict impending spikes, allocating resources before they are needed.
- **Batched Reinforcement Learning:** The underlying Machine Learning engine trains offline automatically every night, meaning the live scaling engine uses zero RAM and CPU on your Proxmox host.
- **Initial Allocation Baselining**: Automatically records your original CPU and RAM configuration to the Proxmox UI Notes before making any automated adjustments.
- **Boot Storm Protection**: Automatically enforces a 15-minute grace period on recently restarted VMs and LXCs to prevent the AI from learning from artificial startup CPU spikes.
- **Native Proxmox Integration**: Uses Proxmox's internal `rrddata` graph APIs to pull usage metrics without needing custom local telemetry agents or guest-agents.
- **Universal Hotplugging**: Dynamically adjusts CPU cores and Memory allocation on-the-fly without restarting the VMs or containers.
  - *Note: For VMs, you MUST explicitly enable "Hotplug: Memory, CPU" in the Proxmox UI under the VM Hardware settings.*

## How it Works
### 1. The XGBoost Engine & Batched Learning
Machine learning requires significant RAM and CPU to build reliable matrices. Rather than stalling your hypervisor by training models continuously, this autoscaler is strictly separated into two components:
1. **The Fast Inference Daemon (`main.py`)**: Runs every 60 seconds. It fetches the last 15 minutes of RRD metrics and passes them through a pre-trained `.json` model. This takes milliseconds and costs ~0% CPU. It logs the accuracy of its prediction into a local SQLite DB for later review.
2. **The Batched Trainer (`train_models.py`)**: Runs once a day via cron (e.g. 3:00 AM). It downloads the last week (or month) of metrics and the logged prediction errors, mathematically calculating the Mean Absolute Error (MAE). It uses these heavy datasets to train a fresh `xgboost` regressor for every single VM and LXC organically, writing the `.json` weights to disk for the inference daemon to seamlessly pick up the next morning.

### 2. Overcommitting & The 95% Host Safeguard
Proxmox allows you to allocate more CPU cores and RAM to containers than you physically possess on the motherboard (overcommitting). While this is great for virtualization, an aggressive AI could easily allocate 200% of your RAM across your instances and instantly crash the Proxmox Kernel out of memory (OOM).

To prevent this, the autoscaler implements a **Hard 95.0% Emergency Stop**. During every single 60-second cycle, the daemon asks the Proxmox Node for its *true physical utilization*. If your server is currently using >95.0% of its physical RAM or CPU, the AI is mathematically forbidden from scaling any instance *upward*, no matter how badly it needs it. It will continue to scale instances *downward* to free up resources, eventually relieving the node. You can control this threshold in the `.env` via `MAX_HOST_CPU_ALLOCATION_PERCENT`.

### 3. Zero-Config Discovery
You do not need to tell the autoscaler which instances to manage. By default, it queries the Proxmox API for every VM and LXC on the node, regardless of whether they are currently powered on or off.

If it finds an instance that is not strictly defined in your `.env` file, it assigns it a **Dynamic Baseline**:
- **Min CPU:** 1
- **Min RAM:** 512MB
- **Max CPU:** Current Cores + 4
- **Max RAM:** Current RAM * 2

If you want to explicitly override these dynamic baselines for a specific container/VM, or completely hide it from the AI, edit the `.env` file mapping.

---

## 🛠️ Step 1: Getting your Proxmox API Token

To allow the autoscaler to magically adjust your hardware, you need to create a secure API token inside Proxmox.

1. Log into your Proxmox Web GUI.
2. Navigate to **Datacenter** -> **Permissions** -> **API Tokens**.
3. Click **Add**.
4. Select the user `root@pam`.
5. Name the Token ID something memorable, like `autoscaler`.
6. Uncheck "Privilege Separation" so the autoscaler has permission to adjust hardware.
7. Click **Add**.
8. ⚠️ **IMPORTANT:** A window will pop up with your **Secret**. Once you close this window, you can *never* see the secret again. Copy it down immediately!

---

## 🚀 Step 2: One-Line Installation

Run the automated one-liner script. This securely clones the repository to the standard `/opt/proxmox-ai-autoscaler` directory, safely configures the Python Machine Learning environment, registers the daemon, and provisions your nightly XGBoost automated-training chron-jobs implicitly.

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/install.sh | bash
```

---

## ⚙️ Step 3: Configuration & Start

The script installs the daemon, but holds off on starting it until you connect your Proxmox API.

Open the newly generated `.env` file using a text editor like `nano`:
```bash
nano /opt/proxmox-ai-autoscaler/.env
```

### The `.env` File Explained
* **Authentication**: Paste the `PROXMOX_TOKEN_ID` and `PROXMOX_TOKEN_SECRET` you created in Step 1.
* **`NODE_NAME`**: Set this to the name of your Proxmox server (usually `pve` by default).
* **System Limits:** 
  * `MAX_HOST_CPU_ALLOCATION_PERCENT=85` safely prevents the autoscaler from assigning more than 85% of your host's physical cores. Do not set this to 100, or your hypervisor itself may stall.
* **Resource Baselines**: 
  * You can explicitly define boundaries by prefixing your Proxmox node ID with `VM_` or `LXC_`.
  * Multiplier Format: `<TYPE>_<ID>=min_cpu_cores,min_ram_mb,max_cpu_cores,max_ram_mb`
  * Example: `VM_100=1,1024,4,4096` means Virtual Machine #100 will never drop below 1 CPU / 1024MB RAM, and will never boost past 4 CPUs / 4096MB RAM, regardless of what the AI predicts.
  * ⚠️ **Note 1 (Proxmox Limit):** Proxmox enforces a strict minimum limit of `1024` MB for VM Memory Hotplugging. If you set a VM's `min_ram_mb` lower than this, the autoscaler will safely floor it at 1024MB to prevent API crashes.
  * ⚠️ **Note 2 (Guest OS Limit):** To prevent severe kernel crashes and file corruption, most Guest Operating Systems actively reject CPU and Memory Hot-Unplugging. Because of this, **the autoscaler natively prevents automatically scaling down a VM's CPU and RAM.** If a VM scales UP to 4 Cores and 8GB RAM during a spike, it will safely remain there permanently until you manually shut down and shrink the machine from the Proxmox UI. LXC Containers do not suffer from either of these limitations and will perfectly scale up and down symmetrically on the fly!
* **Blacklisting (Ignored Entities)**: You can completely block auto-discovery for isolated environments by listing their IDs in `EXCLUDED_VMS=101,102` or `EXCLUDED_LXCS=105`.

Once configured, tell systemd to start the autoscaler:
```bash
systemctl start proxmox-ai-autoscaler
```

You can watch the AI actively predicting and explicitly scaling your instances (e.g., `UP to 4 Cores`) by observing the universal log file:
```bash
tail -f /var/log/proxmox_ai_autoscaler.log
```

---

## �️ Uninstallation

If you wish to completely remove the autoscaler from your Proxmox server, you can run the one-line uninstall script. It will safely stop the systemd service, remove the nightly cron jobs, and delete all logs and configurations generated in `/opt/`.

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/uninstall.sh | bash
```

---

## �🚑 Troubleshooting

**Q: The logs say it's skipping my VM/LXC or failing to scale.**
* Double-check that your ID is correctly mapped in the `.env` file (e.g. `VM_105=...`).
* Ensure you restarted the service after editing the `.env` file! (`systemctl restart proxmox-ai-autoscaler`). 
* If a **VM** is failing to scale locally but LXCs are working, you MUST enable "Hotplug" in the Proxmox UI for that specific VM under its Hardware options.

**Q: "No XGBoost models found yet... Falling back to live metrics"**
* This is completely normal on your very first day! The nightly trainer hasn't run yet. It will use the raw metric data to scale today, and tomorrow morning it will seamlessly switch to the intelligent gradient-boosting matrices once the 3:00 AM cron script finishes. If you want to force it to train *right now*, run `source venv/bin/activate && python train_models.py` manually.

## Acknowledgments
This project was inspired by [fabriziosalmi/proxmox-lxc-autoscale-ml](https://github.com/fabriziosalmi/proxmox-lxc-autoscale-ml).
