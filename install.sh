#!/bin/bash
# Proxmox AI Autoscaler 🚀
# One-Line Automated Installer
# Usage: curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/install.sh | bash

set -e

echo "========================================="
echo "  Proxmox Universal AI Autoscaler Setup  "
echo "========================================="

if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: This script must be run as root (or with sudo)."
  exit 1
fi

APP_DIR="/opt/proxmox-ai-autoscaler"
REPO_URL="https://github.com/deswong/proxmox_ai_autoscaler.git"
SERVICE_NAME="proxmox-ai-autoscaler"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_FILE="/var/log/proxmox_ai_autoscaler.log"

# 1. Install Dependencies
echo "📦 Installing system dependencies (git, python3-venv, cron)..."
apt-get update -yqq
apt-get install -yqq git python3-venv python3-pip cron

# 2. Clone or Update Repository
if [ -d "$APP_DIR" ]; then
    echo "🔄 Updating existing installation at $APP_DIR..."
    cd "$APP_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "📥 Cloning repository to $APP_DIR..."
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# 3. Setup Virtual Environment
echo "🐍 Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Configure Environment Variables
if [ ! -f .env ]; then
  echo "⚙️ Creating default .env configuration..."
  cp .env.example .env
  echo "⚠️  IMPORTANT: You must edit $APP_DIR/.env with your Proxmox API credentials!"
else
  echo "✅ Existing .env configuration found. Leaving intact."
fi

# 5. Setup Logging
echo "📝 Setting up centralized logging at $LOG_FILE..."
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
fi
chown root:root "$LOG_FILE"
chmod 644 "$LOG_FILE"

# 6. Setup Systemd Service
echo "⚙️ Creating Systemd service..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Proxmox Universal AI Autoscaler Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
# Note: We do not start it automatically because the user MUST configure .env first.

# 7. Setup Backup/Trainer Cron Job
echo "🕒 Configuring nightly XGBoost batch training (Cron)..."
CRON_JOB="0 3 * * * cd ${APP_DIR} && ${APP_DIR}/venv/bin/python ${APP_DIR}/train_models.py >> ${LOG_FILE} 2>&1"
# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "${APP_DIR}/train_models.py"; then
    echo "✅ Cron job already configured."
else
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Nightly training configured for 3:00 AM."
fi

# 8. Offer Host Kernel Swappiness Tuning
CURRENT_SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null || echo "60")
if [ "$CURRENT_SWAPPINESS" -eq 1 ]; then
    echo "✅ Host kernel already tuned (vm.swappiness=1). Skipping."
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  💡 Recommended: Host Kernel Swappiness Tuning"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Your host kernel is currently set to vm.swappiness=$CURRENT_SWAPPINESS."
    echo "  The Linux default (60) eagerly moves idle memory pages to the swap disk"
    echo "  even when physical RAM is available, causing unnecessary I/O on your"
    echo "  Proxmox hypervisor. For a host running ML-driven autoscaling, this"
    echo "  interferes with the autoscaler's own swap management for containers."
    echo ""
    echo "  Setting vm.swappiness=1 tells the kernel to only use swap as an absolute"
    echo "  last resort, keeping your RAM available for containers and reducing disk I/O."
    echo ""
    read -r -p "  Apply optimised kernel settings now? (vm.swappiness=1, vm.vfs_cache_pressure=50) [Y/n]: " TUNE_SWAP
    TUNE_SWAP="${TUNE_SWAP:-Y}"
    if [[ "$TUNE_SWAP" =~ ^[Yy]$ ]]; then
        bash "${APP_DIR}/tools/tune_host_swappiness.sh"
        echo "✅ Host kernel tuned for optimal autoscaler performance."
    else
        echo "⏭️  Skipped. You can apply this later:"
        echo "   sudo bash ${APP_DIR}/tools/tune_host_swappiness.sh"
    fi
fi

echo ""
echo "🎉 Installation Complete!"
echo "--------------------------------------------------------"
echo "1. Edit the configuration file with your API Token:"
echo "   nano $APP_DIR/.env"
echo ""
echo "2. Start the service (Once Configured):"
echo "   systemctl start $SERVICE_NAME"
echo ""
echo "3. View the live inference logs:"
echo "   tail -f $LOG_FILE"
echo "--------------------------------------------------------"
