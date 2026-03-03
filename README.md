# Proxmox Universal AI Autoscaler 🚀

**Stop reacting to server crashes. Start predicting them.**

A lightweight, pure-Python service that brings proactive AI autoscaling natively to your **Proxmox Virtual Machines (QEMU/KVM)** and **Linux Containers (LXC)**.

Instead of waiting for servers to hit 100% saturation and stall, this daemon polls Proxmox's native historical RRD telemetry APIs and uses a lightning-fast **XGBoost engine** to forecast resource needs 2 minutes ahead. It hotplugs CPU cores and RAM for LXCs *before* the spike hits, and right-sizes VMs for their next reboot — all while actively protecting the host hypervisor from overload.

Zero database tuning, zero Prometheus stacks — just one service keeping your Proxmox instances fast and your host safe.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Proactive ML Scaling (LXC)** | XGBoost forecasts resource needs 2 minutes ahead — before spikes degrade performance |
| **VM Right-Sizing (next reboot)** | Computes optimal CPU/RAM from 14-day observed peaks + 30% headroom; writes as pending Proxmox config |
| **107-Feature Prediction Engine** | Reads CPU, RAM, disk I/O, network I/O, host load averages, overcommit ratios, time-of-day, and rate-of-change trends simultaneously |
| **MAE Penalty-Weighted Training** | Nightly retraining boosts sample weight (1×–3×) for intervals where predictions were most wrong — models self-correct over time |
| **Host-Aware Scaling** | Three-tier host pressure response: normal → block scale-ups → actively reclaim RAM from idle containers |
| **Rich Telemetry Storage** | Every prediction logs 17 environment fields to SQLite (hour, load avg, overcommit, actual usage) powering future training |
| **Zero-Config Discovery** | No manual container lists required. Dynamic baselines auto-assigned to unknown containers |
| **Boot Storm Protection** | 15-minute grace period after reboots prevents AI from learning from artificial startup spikes |
| **Intelligent Swap Management** | ML-driven LXC swap cap sizing with Safe Flush — drops cap to 0 to force the kernel to reclaim pages |
| **Universal LXC Hotplugging** | Adjusts CPU and RAM live on containers — no restarts required |

---

## 🧠 How it Works

### 1. The XGBoost Prediction Engine (107 Features)

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

### 2. LXC vs VM: Different Strategies

LXC containers and VMs are fundamentally different in how Proxmox handles live resource changes:

| | **LXC** | **VM** |
|---|---|---|
| **Scaling strategy** | Live hotplug (applies immediately) | Pending config (applies on next reboot) |
| **Basis** | 2-min XGBoost forecast + 30% headroom | 14-day rolling observed peak + 30% headroom |
| **Scale down** | ✅ Yes (kernel supports hot-unplug) | N/A — sized correctly for the next boot |
| **Swap management** | ✅ ML-driven + safe flush | N/A (VMs manage swap internally) |

**VM Right-Sizing formula:**
```
peak_ram_mb  = MAX(ctx_actual_ram)  over last 14 days
peak_cpu_pct = MAX(ctx_actual_cpu%) over last 14 days

target_ram  = clamp( int(peak_ram_mb × 1.30), max(min_ram, 1024), max_ram )
target_cpus = clamp( int(peak_cpu_pct/100 × cores × 1.20) + 1, min_cpu, max_cpu )
```

The config is only written when the recommendation differs from current allocation by > 5% RAM or ≥ 1 CPU core, preventing redundant API calls every cycle. On day one (no telemetry yet), it bootstraps from the current-cycle XGBoost prediction peaks.

### 3. Self-Correcting Training (MAE Penalty Weights)

Each nightly training run compounds two signals into the XGBoost sample weights:

1. **Time-recency** — exponential ramp from 1× (oldest) to ~20× (most recent), so the model weights current patterns over old history
2. **Error penalty** — for each training interval, the gap between the model's last prediction and the actual observed value (from `prediction_logs`) is normalised to a multiplier of **1×–3×**

```
sample_weight[i] = time_weight[i] × error_penalty[i]   (normalised)

penalty = 1.0 + 2.0 × normalised_MAE
   0% error  →  1× (no boost)
   100% error →  3× (train 3× harder on this interval)
```

The training log shows: `342/1440 intervals boosted above 1× (max penalty: 2.73×)` so you can observe the reinforcement signal growing over time as the model self-corrects recurring prediction mistakes.

### 4. Two-Component Architecture

Training is computationally expensive. Inference is not. They are strictly separated:

1. **Live Inference Daemon (`main.py`)** — runs every 60 seconds. Fetches the last 15 minutes of RRD metrics, runs them through pre-trained `.json` model weights in milliseconds, hotplugs LXC resources if a significant change is predicted, and writes VM pending configs when the rolling peak warrants a change. Logs the full environment context (including actual observed usage) to SQLite on every cycle.

2. **Nightly Batch Trainer (`train_models.py`)** — runs at 3AM via cron. Downloads the last week of RRD history, joins it against the telemetry log to build MAE-penalty weights, and trains a fresh XGBoost regressor for every LXC and VM. New weights are available to the daemon by dawn.

### 5. Host-Aware, Three-Tier Pressure Response

Every 60-second cycle, the daemon checks the physical node's CPU, RAM, and swap utilization. The response is *graduated*, not binary:

| Host RAM % | Action |
|---|---|
| < 85% | Normal operation |
| 85–90% | Block scale-ups to prevent making things worse |
| **> 90%** | **Block scale-ups** *and* **actively push idle containers down** — if a container uses < 50% of its RAM allocation, shrink it to `usage × 1.5` (floored at `min_ram_mb`) |

The active reclaim guard is conservative: **both** conditions must be true simultaneously, so a busy container is never shrunk during a host-wide load event.

---

## 🛠️ Step 1: Create a Proxmox API Token

1. Log into your Proxmox Web GUI.
2. Navigate to **Datacenter** → **Permissions** → **API Tokens**.
3. Click **Add**, select user `root@pam`, name the token (e.g. `autoscaler`).
4. Uncheck **Privilege Separation**.
5. Click **Add** and immediately **copy the Secret** — it is shown only once.

---

## 🚀 Step 2: One-Line Installation

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/install.sh | bash
```

Installs to `/opt/proxmox-ai-autoscaler`, registers the systemd service, provisions the nightly training cron job, and offers to tune host kernel swappiness for optimal performance — all in one shot.

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
VM_100=1,1024,4,8192

# Exclude from autoscaling entirely
EXCLUDED_LXCS=101,102
EXCLUDED_VMS=200
```

> **VM note:** Proxmox enforces a 1024 MB minimum for VM memory. The autoscaler does *not* live-hotplug VMs — it writes optimal sizing as a pending config that applies on the next reboot, based on the VM's observed peak usage over the last 14 days.

### Start the Service
```bash
systemctl start proxmox-ai-autoscaler
```

Watch the AI make decisions live:
```bash
tail -f /var/log/proxmox_ai_autoscaler.log
```

---

## 🔧 Step 4: Host Kernel Tuning

The installer will detect your current `vm.swappiness` value and offer to tune it automatically. If you skipped that step, you can apply it manually:

```bash
sudo bash /opt/proxmox-ai-autoscaler/tools/tune_host_swappiness.sh
```

This sets `vm.swappiness=1` and `vm.vfs_cache_pressure=50`, telling the kernel to only use swap as a last resort and keeping RAM available for your containers.

---

## 🗃️ Telemetry Database

Every prediction is logged to a local SQLite file (`autoscaler.db`) capturing:

| Column group | Fields |
|---|---|
| Predicted values | `predicted_cpu`, `predicted_ram`, `predicted_swap`, `pred_disk_*`, `pred_net_*` |
| Environment snapshot | `ctx_hour`, `ctx_dow`, `ctx_host_load_1m`, `ctx_host_load_5m`, `ctx_cpu_overcommit`, `ctx_ram_overcommit`, `ctx_container_count` |
| Actual observed | `ctx_actual_cpu`, `ctx_actual_ram` |

The `ctx_actual_*` columns are the key feedback loop — the nightly trainer computes `|predicted - actual|` per interval and boosts sample weights for intervals where the model previously erred, so training accuracy improves automatically over time.

Logs are retained for 14 days and pruned automatically each training run.

---

## 🗑️ Uninstallation

```bash
curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/uninstall.sh | bash
```

Stops the service, removes cron jobs, and deletes all files under `/opt/proxmox-ai-autoscaler/`.

---

## 🚑 Troubleshooting

**"No XGBoost models found yet… Falling back to live metrics"**
> Normal on day one — the nightly trainer hasn't run yet. The daemon uses the latest live RRD reading as a safe fallback. Bootstrap manually:
> ```bash
> cd /opt/proxmox-ai-autoscaler && source venv/bin/activate && python train_models.py
> ```

**A VM is not being live-scaled — only LXCs change**
> This is by design. VMs receive a pending-config update (CPU/RAM optimised for next reboot) rather than live hotplug, which avoids guest OS kernel instability. Check the log for `PENDING CONFIG` lines to confirm the autoscaler is working for your VMs.

**Scaling seems too conservative after host pressure events**
> The autoscaler holds back during host stress. Once host RAM drops below 85% normal operation resumes. Adjust `MAX_HOST_RAM_ALLOCATION_PERCENT` in `.env` if needed.

**A container was scaled down too aggressively**
> Check that `min_ram_mb` is configured in `.env`. By default the dynamic baseline anchors `min_ram_mb` to the container's current allocation, preventing shrinkage below provisioned size.

---

## Acknowledgments
Inspired by [fabriziosalmi/proxmox-lxc-autoscale-ml](https://github.com/fabriziosalmi/proxmox-lxc-autoscale-ml).
