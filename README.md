# Proxmox Universal AI Autoscaler 🚀

**Stop reacting to server crashes. Start predicting them.**

A lightweight, pure-Python service that brings proactive AI autoscaling natively to your **Proxmox Virtual Machines (QEMU/KVM)** and **Linux Containers (LXC)**.

Instead of waiting for servers to hit 100% saturation and stall, this daemon polls Proxmox's native historical RRD telemetry APIs and uses a lightning-fast **XGBoost engine** to forecast resource needs 2 minutes ahead. It hotplugs CPU cores and RAM *before* the spike hits, while actively protecting the host hypervisor from overload.

Zero database tuning, zero Prometheus stacks — just one service keeping your Proxmox instances fast and your host safe.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Proactive ML Scaling** | XGBoost forecasts resource needs 2 minutes ahead — before spikes degrade performance |
| **107-Feature Prediction Engine** | Reads CPU, RAM, disk I/O, network I/O, host load averages, overcommit ratios, time-of-day, and rate-of-change trends simultaneously |
| **Host-Aware Scaling** | Three-tier host pressure response: normal → block scale-ups → actively reclaim RAM from idle containers when host RAM > 90% |
| **Batched Nightly Learning** | Offline GPU/CPU-heavy training runs at 3AM via cron. Live daemon uses pre-trained `.json` weights — costs ~0% host CPU |
| **Intelligent Swap Management** | ML-driven LXC swap cap sizing with Safe Flush: drops cap to 0 to force the kernel to reclaim pages into RAM, then restores |
| **Rich Telemetry Storage** | Every prediction logs 17 environment fields to SQLite (hour, load avg, overcommit ratios, actual usage) for increasingly accurate future training |
| **Zero-Config Discovery** | No manual container lists required. Dynamic baselines auto-assigned to unknown containers |
| **Boot Storm Protection** | 15-minute grace period after reboots prevents AI from learning from artificial startup spikes |
| **Universal Hotplugging** | Adjusts CPU and RAM live — no container/VM restarts required |

---

## 🧠 How it Works

### 1. The XGBoost Prediction Engine

The system learns *patterns*, not just thresholds. The feature vector fed to XGBoost has **107 inputs** per prediction:

```
[0-89]   Per-container history (15 intervals × 6 metrics):
          cpu%, ram_mb, disk_read_bps, disk_write_bps, net_in_bps, net_out_bps

[90-99]  Hypervisor health context:
          host_cpu%, host_ram%, host_swap%,
          load_avg_1m, load_avg_5m, ksm_sharing_mb,
          cpu_overcommit_ratio, ram_overcommit_ratio, container_count, (reserved)

[100-101] Temporal context:
           hour_of_day (0–23), day_of_week (0=Mon…6=Sun)

[101-106] Rate-of-change deltas (last − first of the 15-minute window):
           Δcpu%, Δram_mb, Δdiskread, Δdiskwrite, Δnetin, Δnetout
```

This means the model understands *not only* how loaded a container is right now, but whether that load is rising or falling, how stressed the overall hypervisor is, and whether it's 9AM Monday (typically high load) or 3AM Sunday (typically quiet).

### 2. Two-Component Architecture

Training is computationally expensive. Inference is not. They are strictly separated so your hypervisor is never burdened by ML training during peak hours:

1. **Live Inference Daemon (`main.py`)** — runs every 60 seconds. Fetches the last 15 minutes of RRD metrics, runs them through pre-trained `.json` model weights in milliseconds, and hotplugs resources if a significant change is predicted. Logs the full environment context to SQLite for the trainer to use later.

2. **Nightly Batch Trainer (`train_models.py`)** — runs at 3AM via cron. Downloads the last week of RRD history, correlates it against node-level RRD data, builds a 107-column feature matrix, applies time-based exponential sample weights (recent data matters more), and trains a fresh XGBoost regressor for every LXC and VM. New weights are available to the daemon by dawn.

### 3. Host-Aware, Three-Tier Pressure Response

Every 60-second cycle, the daemon checks the physical node's CPU, RAM, and swap utilization. The response is *graduated*, not binary:

| Host RAM % | Action |
|---|---|
| < 85% | Normal operation |
| 85–90% | Block scale-ups to prevent making things worse |
| **> 90%** | **Block scale-ups** *and* **actively push idle containers down** — if a container is using < 50% of its RAM allocation, it's shrunk to `usage × 1.5` (floored at `min_ram_mb`) |

The active reclaim guard is conservative: **both** conditions must be true simultaneously (host RAM > 90% AND container usage < 50% of allocation), so a busy container is never shrunk during a host-wide load event.

### 4. Zero-Config Discovery

No need to list containers in a config file. The daemon queries the Proxmox API every cycle and assigns **dynamic baselines** to unrecognised instances:

| Limit | Dynamic Value |
|---|---|
| Min CPU | 1 core |
| Min RAM | Current allocated RAM (won't shrink below today's allocation) |
| Max CPU | Current cores + 4 |
| Max RAM | Current RAM × 2 |

Override these for specific containers in the `.env` file, or add to the exclusion list to ignore them entirely.

---

## 🛠️ Step 1: Create a Proxmox API Token

1. Log into your Proxmox Web GUI.
2. Navigate to **Datacenter** → **Permissions** → **API Tokens**.
3. Click **Add**.
4. Select user `root@pam`, name the token (e.g. `autoscaler`).
5. Uncheck **Privilege Separation**.
6. Click **Add** and immediately **copy the Secret** — it is shown only once.

---

## 🚀 Step 2: One-Line Installation

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/install.sh | bash
```

Installs to `/opt/proxmox-ai-autoscaler`, registers the systemd service, and provisions the nightly training cron job automatically.

---

## ⚙️ Step 3: Configure and Start

Open the generated `.env` file:
```bash
nano /opt/proxmox-ai-autoscaler/.env
```

### Authentication
```env
PROXMOX_HOST=192.168.1.10
PROXMOX_TOKEN_ID=root@pam!autoscaler
PROXMOX_TOKEN_SECRET=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
NODE_NAME=pve
```

### Host Safety Limits
```env
# Scale-up blocked when host CPU/RAM exceeds these thresholds (hard cap: 95%)
MAX_HOST_CPU_ALLOCATION_PERCENT=85
MAX_HOST_RAM_ALLOCATION_PERCENT=85

# All scale-ups blocked when host swap exceeds this (host is IO-starved)
MAX_HOST_SWAP_USAGE_PERCENT=20
```

> ⚠️ Never set the CPU/RAM thresholds to 100% — the hypervisor kernel needs headroom to operate.

### Swap Management (LXC Only)
```env
LXC_TARGET_SWAP_MB=-1        # -1 = auto-size from ML peak, 0 = disable swap, N = fixed MB
LXC_MIN_SWAP_MB=256          # Floor applied in auto mode
SWAP_FLUSH_THRESHOLD_PERCENT=50  # Flush swap when usage > 50% of cap
```

### Per-Container Baselines
```env
# Format: <TYPE>_<ID>=min_cpus,min_ram_mb,max_cpus,max_ram_mb
LXC_200=1,512,4,4096
VM_100=1,1024,4,8192   # VMs: min 1024 MB enforced by Proxmox hotplug

# Exclude from autoscaling entirely
EXCLUDED_LXCS=101,102
EXCLUDED_VMS=200
```

> ⚠️ **VM limitations:** Proxmox enforces a 1024 MB minimum for VM memory hotplug. Most guest operating systems also reject CPU/RAM *hot-unplugging*, so the autoscaler only scales VMs **upward** — manual shutdown is required to shrink a VM.

### Start the Service
```bash
systemctl start proxmox-ai-autoscaler
```

Watch the AI make decisions live:
```bash
tail -f /var/log/proxmox_ai_autoscaler.log
```

---

## 🔧 Step 4: Host Kernel Tuning (Optional)

By default, Linux kernels eagerly push idle pages to swap (swappiness=60) even when RAM is available. For a hypervisor running ML autoscaling, this creates unnecessary disk I/O. Apply the included tuning script to configure the kernel to only swap as a last resort:

```bash
sudo bash /opt/proxmox-ai-autoscaler/tools/tune_host_swappiness.sh
```

This sets `vm.swappiness=1` and `vm.vfs_cache_pressure=50`.

---

## 🗃️ Telemetry Database

The autoscaler stores a running log of every prediction in a local SQLite file (`autoscaler.db`). Each row captures:

- **Predicted values**: `predicted_cpu`, `predicted_ram`, `predicted_swap`, `pred_disk_read/write`, `pred_net_in/out`
- **Environment context**: `ctx_hour`, `ctx_dow`, `ctx_host_load_1m`, `ctx_host_load_5m`, `ctx_cpu_overcommit`, `ctx_ram_overcommit`, `ctx_container_count`
- **Actual observed usage**: `ctx_actual_cpu`, `ctx_actual_ram`

Logs are retained for 14 days (configurable) and pruned automatically. As the log fills with real production data, nightly training runs become progressively more accurate — particularly for load_avg and overcommit-aware predictions.

---

## 🗑️ Uninstallation

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/uninstall.sh | bash
```

Stops the service, removes cron jobs, and deletes all files under `/opt/proxmox-ai-autoscaler/`.

---

## 🚑 Troubleshooting

**"No XGBoost models found yet… Falling back to live metrics"**
> Normal on day one — the nightly trainer hasn't run yet. The daemon uses the latest live RRD reading as a safe fallback. Run training manually to bootstrap immediately:
> ```bash
> cd /opt/proxmox-ai-autoscaler && source venv/bin/activate && python train_models.py
> ```

**A VM is failing to scale but LXCs work fine**
> Enable **Hotplug: Memory, CPU** in the Proxmox UI under that VM's Hardware settings.

**Scaling seems too conservative after host pressure events**
> The autoscaler intentionally holds back during host stress. Once host RAM drops below 85% the scaler resumes normal operation. Check `MAX_HOST_RAM_ALLOCATION_PERCENT` in `.env` if you want to adjust the threshold.

**A container was scaled down too aggressively**
> Check if `min_ram_mb` is set correctly in `.env`. By default the dynamic baseline anchors `min_ram_mb` to the container's current allocation, preventing shrinkage below the provisioned size.

---

## Acknowledgments
Inspired by [fabriziosalmi/proxmox-lxc-autoscale-ml](https://github.com/fabriziosalmi/proxmox-lxc-autoscale-ml).
