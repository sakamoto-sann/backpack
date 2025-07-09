#!/bin/bash
"""
🚀 GEMINI-OPTIMIZED UBUNTU 20.04 DEPLOYMENT STRATEGY
High-Performance Deployment for Backpack HF Delta-Neutral Bot

Strategy:
- Python 3.11 for 10-60% performance boost
- System-level optimization for low-latency
- Memory optimization for reduced GC impact
- CPU affinity for consistent performance
- Async optimization for high-frequency trading
"""

set -e  # Exit on any error

echo "🚀 GEMINI-OPTIMIZED BACKPACK BOT DEPLOYMENT"
echo "==========================================="
echo "Target: Ubuntu 20.04 (Focal) - Production Ready"
echo "Optimization: High-Frequency Trading Performance"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   exit 1
fi

# System Information
log_info "System Information:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  Kernel: $(uname -r)"
echo "  CPU: $(nproc) cores"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  Architecture: $(uname -m)"
echo ""

# Phase 1: System Optimization
log_info "Phase 1: System-Level Optimization for HFT"
echo "============================================"

# Update system
log_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
log_info "Installing essential system packages..."
sudo apt install -y \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    htop \
    iotop \
    nethogs \
    tmux \
    vim \
    jq \
    unzip

# System optimization for low-latency trading
log_info "Applying system optimizations for low-latency trading..."

# Disable swap for consistent performance
sudo swapoff -a
log_info "Swap disabled for consistent performance"

# Configure kernel parameters for low-latency
sudo tee /etc/sysctl.d/99-trading-bot.conf > /dev/null <<EOF
# Trading bot optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_no_metrics_save = 1
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0
EOF

# Apply kernel parameters
sudo sysctl -p /etc/sysctl.d/99-trading-bot.conf
log_info "Low-latency kernel parameters applied"

# Phase 2: Python 3.11 Installation (Gemini-Optimized)
log_info "Phase 2: Python 3.11 Installation with Performance Optimization"
echo "================================================================"

# Add deadsnakes PPA for Python 3.11
log_info "Adding deadsnakes PPA for Python 3.11..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.11 and performance-related packages
log_info "Installing Python 3.11 with performance optimizations..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    python3.11-full \
    libjemalloc2 \
    libjemalloc-dev

# Install pip for Python 3.11
log_info "Installing pip for Python 3.11..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Verify Python 3.11 installation
PYTHON_VERSION=$(python3.11 --version)
log_info "Python installation verified: $PYTHON_VERSION"

# Phase 3: Repository Setup and Optimization
log_info "Phase 3: Repository Setup and Code Optimization"
echo "==============================================="

# Clone repository if not exists
if [ ! -d "backpack" ]; then
    log_info "Cloning Backpack repository..."
    git clone https://github.com/sakamoto-sann/backpack.git
else
    log_info "Repository already exists, updating..."
    cd backpack && git pull && cd ..
fi

cd backpack

# Create optimized virtual environment
log_info "Creating optimized virtual environment..."
python3.11 -m venv venv --upgrade-deps

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install performance tools
log_info "Installing performance optimization tools..."
python3.11 -m pip install --upgrade pip setuptools wheel
python3.11 -m pip install \
    cython \
    numba \
    uvloop \
    orjson \
    ujson \
    msgpack \
    lz4 \
    psutil \
    memory_profiler \
    line_profiler

# Install bot dependencies
log_info "Installing bot dependencies..."
cd backpack_trading_bot
python3.11 -m pip install -r requirements.txt

# Install additional HF dependencies with performance focus
log_info "Installing high-frequency trading dependencies..."
python3.11 -m pip install \
    aiohttp[speedups] \
    websockets \
    numpy \
    pandas \
    pyyaml \
    asyncio \
    aiodns \
    cchardet \
    brotli

# Phase 4: Performance Configuration
log_info "Phase 4: Performance Configuration and Memory Optimization"
echo "========================================================="

# Create performance-optimized configuration
log_info "Creating performance-optimized configuration..."
cat > config/performance_config.py << 'EOF'
"""
Performance Configuration for Backpack HF Delta-Neutral Bot
Optimized for Ubuntu 20.04 with Python 3.11
"""

import os
import gc
import asyncio
import uvloop
from numba import jit

# Performance optimizations
class PerformanceOptimizer:
    def __init__(self):
        self.setup_performance_optimizations()
    
    def setup_performance_optimizations(self):
        """Setup performance optimizations"""
        # Use uvloop for better async performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Optimize garbage collection
        gc.set_threshold(700, 10, 10)
        
        # Set process priority
        os.nice(-5)  # Higher priority (requires privileges)
        
        # Memory optimization
        self.optimize_memory()
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Use jemalloc if available
        if os.path.exists('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2'):
            os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libjemalloc.so.2'
        
        # Optimize Python memory
        os.environ['PYTHONMALLOC'] = 'malloc'
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
        os.environ['MALLOC_TOP_PAD_'] = '131072'
        os.environ['MALLOC_MMAP_MAX_'] = '65536'

# Initialize performance optimizer
performance_optimizer = PerformanceOptimizer()

# JIT-compiled functions for critical path
@jit(nopython=True)
def calculate_delta_fast(spot_position, futures_position):
    """JIT-compiled delta calculation"""
    return spot_position + futures_position

@jit(nopython=True)
def calculate_grid_levels_fast(price, spacing, levels):
    """JIT-compiled grid level calculation"""
    import numpy as np
    levels_array = np.empty(levels)
    for i in range(levels):
        levels_array[i] = price * (1 + (i - levels//2) * spacing)
    return levels_array

@jit(nopython=True)
def calculate_volume_efficiency_fast(volume, capital):
    """JIT-compiled volume efficiency calculation"""
    return volume / capital if capital > 0 else 0

EOF

# Phase 5: Systemd Service Configuration
log_info "Phase 5: Systemd Service Configuration for Production"
echo "=================================================="

# Create systemd service file
log_info "Creating systemd service for production deployment..."
sudo tee /etc/systemd/system/backpack-hf-bot.service > /dev/null <<EOF
[Unit]
Description=Backpack High-Frequency Delta-Neutral Trading Bot
After=network.target
Requires=network.target

[Service]
Type=simple
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=PYTHONPATH=$(pwd)
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONMALLOC=malloc
Environment=LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ExecStart=$(pwd)/venv/bin/python3.11 launch_hf_competition_bot.py --config config/hf_delta_neutral_config.yaml --hf-mode --competition-mode
Restart=always
RestartSec=10
KillMode=process
TimeoutStopSec=30
LimitNOFILE=65536
Nice=-5
IOSchedulingClass=1
IOSchedulingPriority=4
CPUSchedulingPolicy=2
CPUSchedulingPriority=50

[Install]
WantedBy=multi-user.target
EOF

# Phase 6: Monitoring and Logging Setup
log_info "Phase 6: Monitoring and Logging Setup"
echo "====================================="

# Create directories
mkdir -p logs data data/mcp_reports data/performance

# Create performance monitoring script
cat > performance_monitor.sh << 'EOF'
#!/bin/bash
# Performance monitoring for trading bot

echo "$(date): Performance Monitoring Started"
echo "======================================="

while true; do
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # Network connections
    network_conns=$(netstat -an | wc -l)
    
    # Bot process info
    bot_pid=$(pgrep -f "launch_hf_competition_bot")
    if [ ! -z "$bot_pid" ]; then
        bot_cpu=$(ps -p $bot_pid -o %cpu --no-headers)
        bot_mem=$(ps -p $bot_pid -o %mem --no-headers)
        echo "$(date): CPU:${cpu_usage}% MEM:${memory_usage}% NET:${network_conns} BOT_CPU:${bot_cpu}% BOT_MEM:${bot_mem}%"
    else
        echo "$(date): CPU:${cpu_usage}% MEM:${memory_usage}% NET:${network_conns} BOT:STOPPED"
    fi
    
    sleep 30
done
EOF

chmod +x performance_monitor.sh

# Phase 7: Security Configuration
log_info "Phase 7: Security Configuration"
echo "=============================="

# Create environment file for secrets
cat > .env << 'EOF'
# Backpack API Configuration
BACKPACK_API_KEY=your_api_key_here
BACKPACK_API_SECRET=your_api_secret_here

# Bot Configuration
STARTING_SOL_AMOUNT=1.0
COMPETITION_MODE=true
HIGH_FREQUENCY_ENABLED=true
DELTA_NEUTRAL_ENABLED=true

# Performance Settings
JEMALLOC_ENABLED=true
UVLOOP_ENABLED=true
NUMBA_JIT_ENABLED=true

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# Set secure permissions
chmod 600 .env

# Phase 8: Testing and Validation
log_info "Phase 8: Testing and Validation"
echo "=============================="

# Test Python 3.11 performance
log_info "Testing Python 3.11 performance..."
python3.11 -c "
import time
import asyncio
import uvloop

# Test uvloop performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def test_performance():
    start = time.time()
    tasks = [asyncio.sleep(0.001) for _ in range(1000)]
    await asyncio.gather(*tasks)
    end = time.time()
    print(f'1000 async tasks completed in {end - start:.3f}s')

asyncio.run(test_performance())
print('Python 3.11 with uvloop: Performance test completed')
"

# Test numba compilation
log_info "Testing Numba JIT compilation..."
python3.11 -c "
from numba import jit
import time
import numpy as np

@jit(nopython=True)
def calculate_fast(data):
    return np.sum(data ** 2)

# Test compilation
test_data = np.random.random(10000)
start = time.time()
result = calculate_fast(test_data)
end = time.time()

print(f'Numba JIT compilation test: {end - start:.3f}s')
print('Numba optimization: Ready')
"

# Test bot import
log_info "Testing bot import and initialization..."
python3.11 -c "
try:
    from backpack_hf_delta_neutral_bot import BackpackHFDeltaNeutralBot
    print('✅ Bot import successful')
except Exception as e:
    print(f'❌ Bot import failed: {e}')
    exit(1)
"

# Final Phase: Deployment Summary
log_info "Deployment Summary"
echo "================="
echo "✅ System optimized for low-latency trading"
echo "✅ Python 3.11 installed with performance enhancements"
echo "✅ Virtual environment created with optimized dependencies"
echo "✅ Performance configuration applied"
echo "✅ Systemd service configured"
echo "✅ Monitoring and logging setup"
echo "✅ Security configuration applied"
echo "✅ Testing completed successfully"
echo ""
echo "🚀 DEPLOYMENT READY!"
echo "===================="
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Start the service: sudo systemctl start backpack-hf-bot"
echo "3. Enable auto-start: sudo systemctl enable backpack-hf-bot"
echo "4. Monitor logs: sudo journalctl -u backpack-hf-bot -f"
echo "5. Run MCP tests: python3.11 test_mergeraft_mcp.py"
echo ""
echo "Performance monitoring: ./performance_monitor.sh"
echo "Service status: sudo systemctl status backpack-hf-bot"
echo ""
echo "🎯 Bot optimized for 10-60% better performance with Python 3.11!"
echo "🔥 Ready for high-frequency delta-neutral trading competition!"

log_info "Deployment completed successfully!"