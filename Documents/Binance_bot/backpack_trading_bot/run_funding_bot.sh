#!/bin/bash
# 🎯 Funding Rate Capture Bot - Startup Script
# Quick launcher for the specialized funding rate capture bot

echo "🎯 FUNDING RATE CAPTURE BOT"
echo "=============================="
echo "Specialized Delta-Neutral Bot for Funding Rate Earnings"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if config files exist
if [ ! -f "config/trading_config.yaml" ]; then
    echo "⚠️  Configuration files not found. Please set up config/ directory"
    exit 1
fi

# Display startup options
echo "🚀 Starting Funding Rate Capture Bot..."
echo "📊 Features:"
echo "   • Delta-neutral positioning"
echo "   • Automatic funding rate capture"
echo "   • Real-time risk monitoring"
echo "   • Multi-exchange support"
echo ""

# Start the bot
echo "🔄 Initializing bot..."
python funding_rate_bot.py

echo "✅ Bot shutdown complete"