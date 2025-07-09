#!/bin/bash
# MergeRaft MCP Testing Runner
# Comprehensive testing suite for Backpack HF Delta-Neutral Bot

echo "🧪 MERGERAFT MCP TESTING SUITE"
echo "=============================="

# Ensure directories exist
mkdir -p logs data data/mcp_reports

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install required dependencies
echo "📦 Installing test dependencies..."
pip install asyncio aiohttp websockets numpy pandas pyyaml

# Run MCP tests
echo "🚀 Starting MergeRaft MCP comprehensive testing..."
python3 test_mergeraft_mcp.py

# Check test results
if [ $? -eq 0 ]; then
    echo "✅ MCP testing completed successfully"
    echo "📊 Test report saved to data/mcp_reports/"
    echo "📋 Logs saved to logs/mergeraft_mcp_test.log"
else
    echo "❌ MCP testing failed"
    echo "📋 Check logs for details: logs/mergeraft_mcp_test.log"
fi

# Display recent test results
echo ""
echo "📊 RECENT TEST RESULTS:"
echo "====================="
tail -n 20 logs/mergeraft_mcp_test.log

# Display test summary if available
if [ -f "data/mcp_reports/mcp_test_report_*.json" ]; then
    echo ""
    echo "📈 TEST SUMMARY:"
    echo "==============="
    latest_report=$(ls -t data/mcp_reports/mcp_test_report_*.json | head -1)
    python3 -c "import json; report=json.load(open('$latest_report')); print(f'Success Rate: {report[\"test_summary\"][\"success_rate\"]:.2%}'); print(f'Grade: {report[\"overall_assessment\"][\"grade\"]}'); print(f'Recommendation: {report[\"overall_assessment\"][\"recommendation\"]}')"
fi

echo ""
echo "🎯 MCP Testing Complete!"
echo "========================"