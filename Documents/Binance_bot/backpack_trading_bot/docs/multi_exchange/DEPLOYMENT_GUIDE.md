# 🚀 Complete Deployment Guide - Active Trading Engine v6.0

## Step-by-Step Instructions to Replace Old Bot with New v6.0 System

### 📋 Overview

This guide will help you safely transition from your current paper trading v4 bot to the new **Active Trading Engine v6.0** on your Contabo server without disrupting your existing setup.

---

## 🔄 PHASE 1: BACKUP AND PREPARATION

### Step 1: Connect to Your Contabo Server
```bash
ssh your_username@your_contabo_ip
```

### Step 2: Backup Current Setup
```bash
# Create backup directory
mkdir -p ~/backups/$(date +%Y%m%d_%H%M%S)
cd ~/backups/$(date +%Y%m%d_%H%M%S)

# Backup current bot and configuration
cp -r ~/binance-bot* ./
cp -r ~/trading* ./
cp ~/.env ./env_backup 2>/dev/null || echo "No .env file found"
cp ~/config.json ./config_backup.json 2>/dev/null || echo "No config.json found"

# Backup any running services
sudo systemctl list-units --type=service --state=running | grep -i trading > running_services.txt
sudo systemctl list-units --type=service --state=running | grep -i binance >> running_services.txt

echo "✅ Backup completed in ~/backups/$(basename $(pwd))"
```

### Step 3: Stop Current Trading Bot
```bash
# Find running trading processes
ps aux | grep -E "(trading|binance|bot)" | grep -v grep

# Stop trading services (adjust service names as needed)
sudo systemctl stop trading-bot 2>/dev/null || echo "No trading-bot service"
sudo systemctl stop binance-bot 2>/dev/null || echo "No binance-bot service"
sudo systemctl stop paper-trading 2>/dev/null || echo "No paper-trading service"

# Kill any remaining python trading processes
pkill -f "python.*trading"
pkill -f "python.*binance"
pkill -f "python.*bot"

echo "✅ Old trading processes stopped"
```

---

## 📦 PHASE 2: INSTALL NEW TRADING ENGINE

### Step 4: Create New Directory Structure
```bash
# Create organized directory structure
mkdir -p ~/trading_systems/v6_active_engine
cd ~/trading_systems/v6_active_engine

# Create subdirectories
mkdir -p {logs,data,config,backups}
```

### Step 5: Clone New Repository
```bash
# Clone the new v6.0 repository (replace with actual repo URL)
git clone https://github.com/yourusername/active-trading-engine-v6.git .

# Or transfer files from local machine using scp
# scp -r /Users/tetsu/Documents/Binance_bot/v0.3/binance-bot-v4-atr-enhanced/integrated_multi_exchange_system/* your_username@your_contabo_ip:~/trading_systems/v6_active_engine/
```

### Step 6: Install Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ if not available
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed"
```

---

## ⚙️ PHASE 3: CONFIGURATION

### Step 7: Configure API Keys
```bash
# Copy configuration template
cp config.example.json config.json

# Edit configuration (use nano, vim, or your preferred editor)
nano config.json
```

**Configuration Template:**
```json
{
  "api_keys": {
    "binance": {
      "api_key": "YOUR_BINANCE_API_KEY",
      "api_secret": "YOUR_BINANCE_SECRET",
      "testnet": true
    },
    "backpack": {
      "api_key": "YOUR_BACKPACK_API_KEY", 
      "api_secret": "YOUR_BACKPACK_SECRET",
      "testnet": true
    }
  },
  "trading_config": {
    "initial_capital": 10000,
    "confidence_threshold": 0.6,
    "max_position_size": 0.05
  }
}
```

### Step 8: Set File Permissions
```bash
# Set secure permissions for config files
chmod 600 config.json
chmod 700 ~/trading_systems/v6_active_engine

# Make main script executable
chmod +x ACTIVE_TRADING_ENGINE_v6.py
```

---

## 🖥️ PHASE 4: SERVICE SETUP

### Step 9: Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/trading-engine-v6.service
```

**Service Configuration:**
```ini
[Unit]
Description=Active Trading Engine v6.0
After=network.target
StartLimitIntervalSec=500
StartLimitBurst=5

[Service]
Type=simple
Restart=on-failure
RestartSec=5s
User=your_username
WorkingDirectory=/home/your_username/trading_systems/v6_active_engine
Environment=PATH=/home/your_username/trading_systems/v6_active_engine/venv/bin
ExecStart=/home/your_username/trading_systems/v6_active_engine/venv/bin/python ACTIVE_TRADING_ENGINE_v6.py
StandardOutput=append:/home/your_username/trading_systems/v6_active_engine/logs/trading.log
StandardError=append:/home/your_username/trading_systems/v6_active_engine/logs/error.log

[Install]
WantedBy=multi-user.target
```

### Step 10: Enable and Configure Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trading-engine-v6

# Create log files with proper permissions
touch logs/trading.log logs/error.log
chmod 644 logs/*.log
```

---

## 🧪 PHASE 5: TESTING

### Step 11: Test Configuration
```bash
# Activate virtual environment
source venv/bin/activate

# Test configuration (dry run)
python ACTIVE_TRADING_ENGINE_v6.py --test-config

# Test API connections
python -c "
import json
from integrated_trading_system.exchanges.binance_adapter import BinanceAdapter
from integrated_trading_system.exchanges.backpack_adapter import BackpackAdapter

with open('config.json') as f:
    config = json.load(f)

print('Testing Binance connection...')
binance = BinanceAdapter(config['api_keys']['binance']['api_key'], 
                        config['api_keys']['binance']['api_secret'], 
                        testnet=True)
print('✅ Binance adapter created')

print('Testing Backpack connection...')
backpack = BackpackAdapter(config['api_keys']['backpack']['api_key'],
                          config['api_keys']['backpack']['api_secret'],
                          testnet=True)
print('✅ Backpack adapter created')
print('✅ All connections tested successfully')
"
```

### Step 12: Run Initial Test
```bash
# Start in test mode first
python ACTIVE_TRADING_ENGINE_v6.py --testnet --dry-run

# Monitor logs in another terminal
tail -f logs/trading.log
```

---

## 🚀 PHASE 6: PRODUCTION DEPLOYMENT

### Step 13: Start Trading Service
```bash
# Start the service
sudo systemctl start trading-engine-v6

# Check service status
sudo systemctl status trading-engine-v6

# Monitor real-time logs
sudo journalctl -u trading-engine-v6 -f
```

### Step 14: Verify Operation
```bash
# Check if service is running
sudo systemctl is-active trading-engine-v6

# Monitor system resources
htop

# Check trading activity
tail -f logs/trading.log | grep -E "(SIGNAL|TRADE|ARBITRAGE)"
```

---

## 📊 PHASE 7: MONITORING SETUP

### Step 15: Create Monitoring Scripts
```bash
# Create health check script
cat > scripts/health_check.sh << 'EOF'
#!/bin/bash
SERVICE_NAME="trading-engine-v6"
LOG_FILE="/home/$(whoami)/trading_systems/v6_active_engine/logs/trading.log"

# Check service status
if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo "❌ Service $SERVICE_NAME is not running"
    exit 1
fi

# Check recent activity (last 5 minutes)
RECENT_ACTIVITY=$(tail -n 100 $LOG_FILE | grep $(date -d '5 minutes ago' '+%Y-%m-%d %H:%M') | wc -l)
if [ $RECENT_ACTIVITY -eq 0 ]; then
    echo "⚠️ No recent activity detected"
else
    echo "✅ Service is active with $RECENT_ACTIVITY recent log entries"
fi
EOF

chmod +x scripts/health_check.sh
```

### Step 16: Setup Automated Monitoring
```bash
# Add to crontab for regular health checks
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/$(whoami)/trading_systems/v6_active_engine/scripts/health_check.sh") | crontab -

# Create log rotation
sudo nano /etc/logrotate.d/trading-engine-v6
```

**Log Rotation Configuration:**
```
/home/your_username/trading_systems/v6_active_engine/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 your_username your_username
    postrotate
        systemctl reload trading-engine-v6
    endscript
}
```

---

## 🔧 PHASE 8: FINAL VALIDATION

### Step 17: Performance Verification
```bash
# Check performance metrics
python -c "
import json
import time
from datetime import datetime

# Monitor for 5 minutes
print('📊 Monitoring trading activity for 5 minutes...')
start_time = time.time()
while time.time() - start_time < 300:  # 5 minutes
    with open('logs/trading.log') as f:
        lines = f.readlines()
        signals = [l for l in lines[-100:] if 'SIGNAL' in l]
        trades = [l for l in lines[-100:] if 'TRADE' in l]
        arbitrage = [l for l in lines[-100:] if 'ARBITRAGE' in l]
    
    print(f'📈 Signals: {len(signals)}, Trades: {len(trades)}, Arbitrage: {len(arbitrage)}')
    time.sleep(60)
"
```

### Step 18: Security Hardening
```bash
# Set up firewall rules
sudo ufw allow ssh
sudo ufw enable

# Secure config files
chmod 600 config.json
chown $(whoami):$(whoami) config.json

# Set up automatic security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## 🆘 TROUBLESHOOTING

### Common Issues and Solutions

#### Service Won't Start
```bash
# Check service logs
sudo journalctl -u trading-engine-v6 -n 50

# Check Python environment
source venv/bin/activate
python --version
pip list | grep -E "(pandas|numpy|requests)"
```

#### API Connection Issues
```bash
# Test API connectivity
curl -X GET "https://testnet.binance.vision/api/v3/ping"
curl -X GET "https://api.backpack.exchange/api/v1/status"

# Check API key permissions
python -c "
import requests
api_key = 'YOUR_API_KEY'
headers = {'X-MBX-APIKEY': api_key}
response = requests.get('https://testnet.binance.vision/api/v3/account', headers=headers)
print(f'Binance API Response: {response.status_code}')
"
```

#### Performance Issues
```bash
# Monitor system resources
free -h
df -h
top -p $(pgrep -f trading-engine)

# Check log file size
du -sh logs/
```

---

## 📱 MAINTENANCE COMMANDS

### Daily Operations
```bash
# Check service status
sudo systemctl status trading-engine-v6

# View recent performance
tail -n 100 logs/trading.log | grep "PERFORMANCE"

# Check system health
./scripts/health_check.sh
```

### Weekly Maintenance
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Backup logs and data
tar -czf backups/weekly_backup_$(date +%Y%m%d).tar.gz logs/ data/

# Check log rotation
sudo logrotate -d /etc/logrotate.d/trading-engine-v6
```

### Emergency Procedures
```bash
# Emergency stop
sudo systemctl stop trading-engine-v6

# Quick restart
sudo systemctl restart trading-engine-v6

# Restore from backup
cp ~/backups/YYYYMMDD_HHMMSS/config_backup.json ./config.json
sudo systemctl restart trading-engine-v6
```

---

## ✅ SUCCESS CHECKLIST

- [ ] Old bot safely stopped and backed up
- [ ] New v6.0 engine installed and configured
- [ ] API keys properly configured and tested
- [ ] Systemd service created and enabled
- [ ] Initial testing completed successfully
- [ ] Production deployment verified
- [ ] Monitoring and health checks active
- [ ] Security measures implemented
- [ ] Emergency procedures documented

**🎉 Congratulations! Your Active Trading Engine v6.0 is now deployed and operational!**

---

## 📞 Support

If you encounter any issues during deployment:

1. **Check Logs**: `tail -f logs/trading.log`
2. **Service Status**: `sudo systemctl status trading-engine-v6`
3. **Health Check**: `./scripts/health_check.sh`
4. **Emergency Stop**: `sudo systemctl stop trading-engine-v6`

**Remember**: Always test with small amounts and testnet first before deploying with real capital!