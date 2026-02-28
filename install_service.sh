#!/bin/bash

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

APP_DIR=$(pwd)
SERVICE_NAME="lxc-autoscaler"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Creating python virtual environment and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env from .env.example if it doesn't exist
if [ ! -f .env ]; then
  echo "Copying .env.example to .env (Remember to edit it!)"
  cp .env.example .env
fi

echo "Setting up centralized logging..."
LOG_FILE="/var/log/proxmox_lxc_autoscaler.log"
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
fi
chown root:root "$LOG_FILE"
chmod 644 "$LOG_FILE"

echo "Creating systemd service file..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Proxmox LXC AI Autoscaler Service
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

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling $SERVICE_NAME to start on boot..."
systemctl enable "$SERVICE_NAME"

echo ""
echo "Installation complete!"
echo ""
echo "Please edit the .env file with your actual Proxmox API connection details:"
echo "  nano ${APP_DIR}/.env"
echo ""
echo "Then start the fast-inference loop service:"
echo "  systemctl start $SERVICE_NAME"
echo ""
echo "To view live inference and scaling logs, use:"
echo "  tail -f /var/log/proxmox_lxc_autoscaler.log"
echo ""
echo "--- IMPORTANT: BATCH TRAINING ---"
echo "To allow the XGBoost models to learn from historical data natively,"
echo "you must configure the nightly batch trainer using cron."
echo "Run 'crontab -e' and add the following line to train every day at 3:00 AM:"
echo "0 3 * * * cd ${APP_DIR} && ${APP_DIR}/venv/bin/python ${APP_DIR}/train_models.py >> /var/log/proxmox_lxc_autoscaler.log 2>&1"
echo "---------------------------------"

