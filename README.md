# Proxmox LXC AI Autoscaler 🚀

A lightweight, purely Python-based service that brings proactive AI autoscaling to your Proxmox Linux Containers (LXC). 

Instead of waiting for your containers to hit 100% saturation and stall, this daemon polls Proxmox's native historical RRD telemetry APIs and uses an **XGBoost Regressor** to forecast computing trends into the future. It intelligently hotplugs CPU cores and RAM allocations *before* the spike hits, while strictly enforcing host-level limiters to ensure you never starve your hypervisor of overhead resources. 

Zero database tuning, zero complex Prometheus stacks—just one service keeping your Proxmox containers fast and your host safe.

## Features
- **Proactive Scaling**: Looks up to 2 minutes into the future to predict impending spikes, allocating resources before they are needed.
- **Batched Reinforcement Learning:** The underlying Machine Learning engine trains offline automatically every night, meaning the live scaling engine uses zero RAM and CPU on your Proxmox host.
- **Native Proxmox Integration**: Uses Proxmox's internal `rrddata` graph APIs to pull usage metrics without needing custom local telemetry agents.
- **Hotpluggable Safety**: Dynamically adjusts CPU cores and Memory allocation on-the-fly without container restarts.
- **Persistent Baselines**: Minimum and maximum boundaries per LXC are easily defined in the `.env` using simple variables.

---

## 🛠️ Step 1: Getting your Proxmox API Token

To allow the autoscaler to magically adjust your containers, you need to create a secure API token inside Proxmox.

1. Log into your Proxmox Web GUI.
2. Navigate to **Datacenter** -> **Permissions** -> **API Tokens**.
3. Click **Add**.
4. Select the user `root@pam`.
5. Name the Token ID something memorable, like `autoscaler`.
6. Uncheck "Privilege Separation" so the autoscaler has permission to adjust container hardware.
7. Click **Add**.
8. ⚠️ **IMPORTANT:** A window will pop up with your **Secret**. Once you close this window, you can *never* see the secret again. Copy it down immediately!

---

## ⚙️ Step 2: Configuration

Log into your Proxmox host via SSH and clone this repository:
```bash
git clone https://github.com/deswong/proxmox_lxc_autoscaler.git
cd proxmox_lxc_autoscaler
cp .env.example .env
```

Open the `.env` file using a text editor like `nano`:
```bash
nano .env
```

### The `.env` File Explained
* **Authentication**: Paste the `PROXMOX_TOKEN_ID` and `PROXMOX_TOKEN_SECRET` you created in Step 1.
* **`NODE_NAME`**: Set this to the name of your Proxmox server (usually `pve` by default).
* **System Limits:** 
  * `MAX_HOST_CPU_ALLOCATION_PERCENT=85` safely prevents the autoscaler from assigning more than 85% of your host's physical cores. Do not set this to 100, or your hypervisor itself may stall.
* **LXC Baselines**: 
  * You must strictly define the boundaries for the containers you want to autoscale. 
  * The format is: `LXC_<ID>=min_cpu_cores,min_ram_mb,max_cpu_cores,max_ram_mb`
  * Example: `LXC_100=1,512,4,4096` means Container #100 will never drop below 1 CPU / 512MB RAM, and will never boost past 4 CPUs / 4096MB RAM, regardless of what the AI predicts.

---

## 🚀 Step 3: Installation

Run the automated systemd install script. This will download the Python Machine Learning libraries and start the fast-inference loop in the background:

```bash
sudo ./install_service.sh
```

You can watch the AI actively scaling your containers by viewing the logs:
```bash
tail -f /var/log/proxmox_lxc_autoscaler.log
```

---

## 🧠 Step 4: The Nightly Machine Learning (Required)
The autoscaler uses a fast "Inference loop" to make guesses, but it needs to learn from its mistakes! We provide a `train_models.py` script that looks back at the entire week of Proxmox data to build precise, custom XGBoost Machine Learning models for each of your LXCs.

Because training is heavy, **you must schedule it to run at night when your server is quiet.**

1. Open the cron editor:
```bash
crontab -e
```
2. Paste this at the bottom of the file (adjust the `/path/to/` to where you actually downloaded the folder!):
```bash
0 3 * * * cd /root/proxmox_lxc_autoscaler && /root/proxmox_lxc_autoscaler/venv/bin/python /root/proxmox_lxc_autoscaler/train_models.py >> /var/log/proxmox_lxc_autoscaler.log 2>&1
```
*(This tells the server to train the AI at exactly 3:00 AM every night).*

---

## 🚑 Troubleshooting

**Q: The logs say it's skipping my LXC or failing to scale.**
* Double-check that your LXC ID is correctly mapped in the `.env` file (`LXC_105=...`).
* Ensure you restarted the service after editing the `.env` file! (`systemctl restart lxc-autoscaler`). 

**Q: "No XGBoost models found yet... Falling back to live metrics"**
* This is completely normal on your very first day! The nightly trainer hasn't run yet. It will use the raw metric data to scale today, and tomorrow morning it will seamlessly switch to the intelligent gradient-boosting matrices once the 3:00 AM cron script finishes. If you want to force it to train *right now*, run `source venv/bin/activate && python train_models.py` manually.

## Acknowledgments
This project was inspired by [fabriziosalmi/proxmox-lxc-autoscale-ml](https://github.com/fabriziosalmi/proxmox-lxc-autoscale-ml).
