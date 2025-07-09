#!/usr/bin/env python3
"""
Unified Binance Trading Bot
Main entry point for all trading strategies and bot operations
"""

import asyncio
import logging
import signal
import sys
import argparse
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from core.utils.config_manager import load_config, BotConfig
from core.strategies.grid_trading.market_analyzer import MarketAnalyzer
from core.strategies.arbitrage.triangular_arbitrage import ArbitrageBot
from core.execution.order_manager import OrderManager
from core.execution.risk_manager import RiskManager
from core.data.market_data import MarketDataManager
from core.data.analytics.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: BotConfig):
        """
        Initialize the trading bot.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.is_running = False
        self.strategy = None
        self.market_analyzer = None
        self.order_manager = None
        self.risk_manager = None
        self.market_data_manager = None
        self.performance_tracker = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Trading bot initialized with strategy: {config.strategy.type}")
    
    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # Market analyzer (shared by all strategies)
            self.market_analyzer = MarketAnalyzer(self.config)
            
            # Order management
            self.order_manager = OrderManager(self.config)
            
            # Risk management
            self.risk_manager = RiskManager(self.config)
            
            # Market data management
            self.market_data_manager = MarketDataManager(self.config)
            
            # Performance tracking
            self.performance_tracker = PerformanceTracker(self.config)
            
            # Initialize strategy based on configuration
            self._initialize_strategy()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_strategy(self):
        """Initialize the selected trading strategy"""
        try:
            strategy_type = self.config.strategy.type
            
            if strategy_type == 'grid_trading':
                from core.strategies.grid_trading.adaptive_grid import AdaptiveGridStrategy
                self.strategy = AdaptiveGridStrategy(
                    config=self.config,
                    market_analyzer=self.market_analyzer,
                    order_manager=self.order_manager,
                    risk_manager=self.risk_manager
                )
            
            elif strategy_type == 'arbitrage':
                self.strategy = ArbitrageBot(self.config)
            
            elif strategy_type == 'delta_neutral':
                from core.strategies.delta_neutral.delta_neutral_manager import DeltaNeutralManager
                self.strategy = DeltaNeutralManager(
                    config=self.config,
                    market_analyzer=self.market_analyzer,
                    order_manager=self.order_manager,
                    risk_manager=self.risk_manager
                )
            
            elif strategy_type == 'advanced':
                from core.strategies.advanced.multi_strategy_manager import MultiStrategyManager
                self.strategy = MultiStrategyManager(
                    config=self.config,
                    market_analyzer=self.market_analyzer,
                    order_manager=self.order_manager,
                    risk_manager=self.risk_manager
                )
            
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            logger.info(f"Strategy initialized: {strategy_type}")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            raise
    
    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting trading bot...")
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Initialize strategy
            if hasattr(self.strategy, 'initialize'):
                await self.strategy.initialize()
            
            # Start market data feed
            await self.market_data_manager.start()
            
            # Start performance tracking
            await self.performance_tracker.start()
            
            # Start risk monitoring
            await self.risk_manager.start_monitoring()
            
            # Set running flag
            self.is_running = True
            
            # Start main trading loop
            await self._run_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def _run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check if risk manager allows trading
                if not await self.risk_manager.can_trade():
                    logger.warning("Risk manager blocking trading")
                    await asyncio.sleep(10)
                    continue
                
                # Get latest market data
                market_data = await self.market_data_manager.get_latest_data()
                
                if market_data is None:
                    logger.warning("No market data available")
                    await asyncio.sleep(5)
                    continue
                
                # Run strategy
                if hasattr(self.strategy, 'execute'):
                    await self.strategy.execute(market_data)
                elif hasattr(self.strategy, 'start_websocket'):
                    # For strategies that handle their own loop (like arbitrage)
                    await self.strategy.start_websocket()
                
                # Update performance tracking
                await self.performance_tracker.update()
                
                # Brief pause to prevent excessive CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        
        self.is_running = False
        
        try:
            # Stop strategy
            if hasattr(self.strategy, 'stop'):
                await self.strategy.stop()
            
            # Stop market data manager
            if self.market_data_manager:
                await self.market_data_manager.stop()
            
            # Stop performance tracker
            if self.performance_tracker:
                await self.performance_tracker.stop()
            
            # Stop risk manager
            if self.risk_manager:
                await self.risk_manager.stop_monitoring()
            
            # Cancel all pending orders
            if self.order_manager:
                await self.order_manager.cancel_all_orders()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            status = {
                'is_running': self.is_running,
                'strategy': self.config.strategy.type,
                'paper_trading': self.config.paper_trading,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add strategy-specific status
            if hasattr(self.strategy, 'get_status'):
                strategy_status = await self.strategy.get_status()
                status.update(strategy_status)
            
            # Add risk manager status
            if self.risk_manager:
                risk_status = await self.risk_manager.get_status()
                status['risk'] = risk_status
            
            # Add performance data
            if self.performance_tracker:
                performance = await self.performance_tracker.get_current_performance()
                status['performance'] = performance
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

# Placeholder classes for components not yet implemented
class OrderManager:
    def __init__(self, config: BotConfig):
        self.config = config
        logger.info("Order manager initialized")
    
    async def cancel_all_orders(self):
        logger.info("Canceling all orders (placeholder)")

class RiskManager:
    def __init__(self, config: BotConfig):
        self.config = config
        logger.info("Risk manager initialized")
    
    async def can_trade(self) -> bool:
        return True
    
    async def start_monitoring(self):
        logger.info("Starting risk monitoring (placeholder)")
    
    async def stop_monitoring(self):
        logger.info("Stopping risk monitoring (placeholder)")
    
    async def get_status(self) -> Dict[str, Any]:
        return {'status': 'active'}

class MarketDataManager:
    def __init__(self, config: BotConfig):
        self.config = config
        logger.info("Market data manager initialized")
    
    async def start(self):
        logger.info("Starting market data feed (placeholder)")
    
    async def stop(self):
        logger.info("Stopping market data feed (placeholder)")
    
    async def get_latest_data(self):
        return {'symbol': 'BTCUSDT', 'price': 50000}

class PerformanceTracker:
    def __init__(self, config: BotConfig):
        self.config = config
        logger.info("Performance tracker initialized")
    
    async def start(self):
        logger.info("Starting performance tracking (placeholder)")
    
    async def stop(self):
        logger.info("Stopping performance tracking (placeholder)")
    
    async def update(self):
        pass
    
    async def get_current_performance(self) -> Dict[str, Any]:
        return {'pnl': 0.0, 'trades': 0}

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Binance Trading Bot')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--strategy', '-s', help='Strategy type', 
                       choices=['grid_trading', 'arbitrage', 'delta_neutral', 'advanced'])
    parser.add_argument('--paper', '-p', action='store_true', help='Enable paper trading')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--status', action='store_true', help='Show bot status and exit')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_files = [args.config] if args.config else None
        config = load_config(config_files)
        
        # Override configuration with command line arguments
        if args.strategy:
            config.strategy.type = args.strategy
        
        if args.paper:
            config.paper_trading = True
        
        if args.debug:
            config.debug_mode = True
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create and start bot
        bot = TradingBot(config)
        
        if args.status:
            # Just show status and exit
            status = await bot.get_status()
            print(f"Bot Status: {status}")
            return
        
        # Start the bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the main function
    asyncio.run(main())