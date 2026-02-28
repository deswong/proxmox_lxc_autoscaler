# Proxmox LXC AI Autoscaler 🚀

A lightweight, purely Python-based service that brings proactive AI autoscaling to your Proxmox Linux Containers (LXC). 

Instead of waiting for your containers to hit 100% saturation and stall, this daemon polls Proxmox's native historical RRD telemetry APIs and uses **Scikit-Learn** linear regression to forecast computing trends into the future. It intelligently hotplugs CPU cores and RAM allocations *before* the spike hits, while strictly enforcing host-level limiters to ensure you never starve your hypervisor of overhead resources. 

Zero database tuning, zero complex Prometheus stacks—just one service keeping your Proxmox containers fast and your host safe.

## Features
- **Proactive Scaling**: Looks into the future to predict impending spikes, allocating resources proactively.
- **Native Proxmox Integration**: Uses Proxmox's internal `rrddata` graph APIs. No local metrics databases to manage or bloat over time.
- **Hotpluggable Safety**: Dynamically adjusts CPU cores and Memory allocation on-the-fly without container restarts.
- **Hypervisor Aware**: Bounded by customizable maximum host limits (e.g., stops scaling if the node exceeds 85% total resource allocation).
- **Persistent Baselines**: Minimum and maximum boundaries per LXC are easily defined in the `.env` via simple `LXC_100=min_cpu,min_ram,max_cpu,max_ram` variables. These baselines are automatically seeded into a local SQLite database to outlive power cycles.

## Quickstart
1. Clone the repository into your Proxmox Host.
2. `cp .env.example .env` and edit your API tokens and base limits.
3. Run `sudo ./install_service.sh` to install the requirements and launch the `systemd` daemon!

## Acknowledgments
This project was inspired by [fabriziosalmi/proxmox-lxc-autoscale-ml](https://github.com/fabriziosalmi/proxmox-lxc-autoscale-ml).
