#!/usr/bin/env python3
"""
🎯 FUNDING RATE CAPTURE BOT
Specialized Delta-Neutral Bot Optimized for Funding Rate Earnings

Key Features:
- 💰 Funding Rate Capture: Earn from positive funding rates
- ⚖️ Delta Neutrality: Market direction neutral positions  
- 🔄 Dynamic Hedging: Automatic futures hedging
- 📊 Real-time Monitoring: Continuous funding rate analysis
- 🛡️ Risk Controls: Multi-layer risk management
- 🎯 High Performance: Optimized for funding earnings
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.utils.config_manager import load_config, BotConfig
from core.strategies.delta_neutral.delta_neutral_manager import (
    DeltaNeutralManager, GridConfiguration, RiskLimits
)
from core.strategies.delta_neutral.funding_rate_collector import CompliantFundingFeeCollector
from core.execution.order_manager import OrderManager
from core.execution.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/funding_rate_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FundingRateBot:
    """
    Specialized Funding Rate Capture Bot
    
    Optimized for earning from funding rates while maintaining delta neutrality.
    Uses advanced delta-neutral strategies to capture funding fees consistently.
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize funding rate capture bot.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.is_running = False
        
        # Core components
        self.delta_neutral_manager = None
        self.funding_collector = None
        self.order_manager = None
        self.risk_manager = None
        
        # Performance tracking
        self.total_funding_earned = 0.0
        self.daily_funding_target = 0.0
        self.positions_count = 0
        self.start_time = datetime.now()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("🎯 Funding Rate Capture Bot initialized")
    
    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # Risk limits for funding strategy
            risk_limits = RiskLimits(
                max_delta_exposure=0.05,  # Very tight delta control for neutral strategy
                max_daily_loss=self.config.risk.max_daily_loss,
                max_position_size=self.config.position.max_position_size,
                emergency_stop_loss=0.03,  # 3% emergency stop
                funding_rate_threshold=0.0001  # 0.01% minimum funding rate
            )
            
            # Grid configuration optimized for funding capture
            grid_config = GridConfiguration(
                symbol=self.config.strategy.delta_neutral.spot_symbol,
                grid_levels=self.config.strategy.delta_neutral.grid_levels,
                grid_spacing=self.config.strategy.delta_neutral.grid_spacing,
                base_order_size=self.config.strategy.delta_neutral.base_position_size / 20,  # Smaller orders
                max_total_position=self.config.strategy.delta_neutral.base_position_size,
                rebalance_threshold=self.config.strategy.delta_neutral.rebalance_threshold,
                funding_optimization=True  # Enable funding optimization
            )
            
            # Initialize placeholder clients (would be real exchange clients)
            self.spot_client = PlaceholderSpotClient()
            self.futures_client = PlaceholderFuturesClient()
            self.rate_limiter = PlaceholderRateLimiter()
            
            # Initialize funding collector
            funding_config = {
                'funding_symbols': [self.config.strategy.delta_neutral.spot_symbol],
                'min_funding_rate': 0.0001,  # 0.01%
                'max_funding_position': self.config.strategy.delta_neutral.base_position_size,
                'funding_check_interval': 1800  # 30 minutes
            }
            
            self.funding_collector = CompliantFundingFeeCollector(
                binance_client=self.futures_client,
                rate_limiter=self.rate_limiter,
                config=funding_config
            )
            
            # Initialize delta neutral manager
            self.delta_neutral_manager = DeltaNeutralManager(
                risk_limits=risk_limits,
                grid_config=grid_config,
                spot_client=self.spot_client,
                futures_client=self.futures_client,
                funding_collector=self.funding_collector
            )
            
            # Initialize order and risk managers
            self.order_manager = OrderManager(self.config)
            self.risk_manager = RiskManager(self.config)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def start(self):
        """Start the funding rate capture bot"""
        try:
            logger.info("🚀 Starting Funding Rate Capture Bot")
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start funding collector
            await self.funding_collector.start_monitoring()
            
            # Start delta neutral strategy
            success = await self.delta_neutral_manager.start_strategy()
            if not success:
                logger.error("Failed to start delta neutral strategy")
                return
            
            # Start risk monitoring
            await self.risk_manager.start_monitoring()
            
            # Set running flag
            self.is_running = True
            
            # Display initial status
            await self._display_startup_info()
            
            # Start main trading loop
            await self._run_funding_capture_loop()
            
        except Exception as e:
            logger.error(f"Error starting funding rate bot: {e}")
            await self.stop()
            raise
    
    async def _display_startup_info(self):
        """Display startup information and current opportunities"""
        try:
            logger.info("=" * 60)
            logger.info("🎯 FUNDING RATE CAPTURE BOT - ACTIVE")
            logger.info("=" * 60)
            
            # Get current funding opportunities
            opportunities = self.funding_collector.get_best_funding_opportunities(min_annual_rate=0.05)
            
            if opportunities:
                logger.info("💰 Current High-Value Funding Opportunities:")
                for opp in opportunities[:3]:
                    logger.info(
                        f"   {opp['symbol']}: {opp['annualized_rate']:.2%} annual "
                        f"({opp['funding_rate']:.4f} per period) - "
                        f"Suggested: {opp['suggested_side']}"
                    )
            else:
                logger.info("📊 Monitoring funding rates - no high-value opportunities currently")
            
            # Display strategy configuration
            logger.info(f"⚙️ Configuration:")
            logger.info(f"   Primary Symbol: {self.config.strategy.delta_neutral.spot_symbol}")
            logger.info(f"   Base Position: ${self.config.strategy.delta_neutral.base_position_size}")
            logger.info(f"   Max Delta: {self.delta_neutral_manager.risk_limits.max_delta_exposure:.1%}")
            logger.info(f"   Grid Levels: {self.delta_neutral_manager.grid_config.grid_levels}")
            logger.info(f"   Paper Trading: {self.config.paper_trading}")
            
            logger.info("🔄 Strategy is now actively monitoring and capturing funding rates...")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error displaying startup info: {e}")
    
    async def _run_funding_capture_loop(self):
        """Main funding capture monitoring loop"""
        logger.info("🔄 Starting funding capture monitoring loop")
        
        while self.is_running:
            try:
                # Update funding performance metrics
                await self._update_funding_metrics()
                
                # Check for new high-value opportunities
                await self._check_new_opportunities()
                
                # Monitor strategy performance
                await self._monitor_strategy_performance()
                
                # Log periodic status
                await self._log_periodic_status()
                
                # Sleep for monitoring interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in funding capture loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_funding_metrics(self):
        """Update funding rate capture metrics"""
        try:
            # Get current strategy status
            status = self.delta_neutral_manager.get_status()
            
            if 'position' in status:
                self.total_funding_earned = status['position'].get('funding_collected', 0.0)
                
                # Calculate daily funding rate
                hours_running = (datetime.now() - self.start_time).total_seconds() / 3600
                if hours_running > 0:
                    daily_rate = (self.total_funding_earned / hours_running) * 24
                    self.daily_funding_target = daily_rate
            
        except Exception as e:
            logger.error(f"Error updating funding metrics: {e}")
    
    async def _check_new_opportunities(self):
        """Check for new high-value funding opportunities"""
        try:
            # Get current best opportunities
            opportunities = self.funding_collector.get_best_funding_opportunities(min_annual_rate=0.10)
            
            for opp in opportunities:
                if opp['annualized_rate'] > 0.20:  # 20%+ annual rate
                    logger.info(
                        f"🎯 HIGH-VALUE OPPORTUNITY: {opp['symbol']} "
                        f"{opp['annualized_rate']:.1%} annual rate - "
                        f"Consider {opp['suggested_side']} position"
                    )
            
        except Exception as e:
            logger.error(f"Error checking new opportunities: {e}")
    
    async def _monitor_strategy_performance(self):
        """Monitor overall strategy performance"""
        try:
            # Get strategy status
            status = self.delta_neutral_manager.get_status()
            
            if not status.get('is_active', False):
                logger.warning("⚠️ Delta neutral strategy is not active")
                return
            
            # Check delta exposure
            delta_exposure = status.get('position', {}).get('delta_exposure', 0)
            max_delta = self.delta_neutral_manager.risk_limits.max_delta_exposure
            
            if abs(delta_exposure) > max_delta * 0.8:  # 80% of limit
                logger.warning(f"⚠️ High delta exposure: {delta_exposure:.4f} (limit: {max_delta:.4f})")
            
            # Check funding collection
            funding_collected = status.get('position', {}).get('funding_collected', 0)
            if funding_collected > 0:
                logger.info(f"💰 Total funding earned: ${funding_collected:.2f}")
            
        except Exception as e:
            logger.error(f"Error monitoring strategy performance: {e}")
    
    async def _log_periodic_status(self):
        """Log periodic status update"""
        try:
            # Log status every 30 minutes
            current_time = datetime.now()
            if current_time.minute in [0, 30]:
                
                status = self.delta_neutral_manager.get_status()
                funding_summary = self.funding_collector.get_funding_summary()
                
                logger.info("📊 FUNDING BOT STATUS UPDATE:")
                logger.info(f"   ⏰ Runtime: {current_time - self.start_time}")
                logger.info(f"   💰 Total Funding: ${status.get('position', {}).get('funding_collected', 0):.2f}")
                logger.info(f"   ⚖️ Delta Exposure: {status.get('position', {}).get('delta_exposure', 0):.4f}")
                logger.info(f"   🎯 Strategy Active: {status.get('is_active', False)}")
                
                # Log current funding rates
                for symbol, data in funding_summary.items():
                    rate = data.get('annualized_rate', 0)
                    logger.info(f"   📈 {symbol}: {rate:.2%} annual funding rate")
            
        except Exception as e:
            logger.error(f"Error logging periodic status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.is_running = False
    
    async def stop(self):
        """Stop the funding rate capture bot"""
        logger.info("🛑 Stopping Funding Rate Capture Bot...")
        
        self.is_running = False
        
        try:
            # Stop delta neutral strategy
            if self.delta_neutral_manager:
                await self.delta_neutral_manager.stop_strategy()
            
            # Stop funding collector
            if self.funding_collector:
                await self.funding_collector.shutdown("Bot shutdown")
            
            # Stop risk manager
            if self.risk_manager:
                await self.risk_manager.stop_monitoring()
            
            # Final performance summary
            await self._display_final_summary()
            
            logger.info("✅ Funding Rate Capture Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    async def _display_final_summary(self):
        """Display final performance summary"""
        try:
            runtime = datetime.now() - self.start_time
            
            logger.info("=" * 60)
            logger.info("📊 FUNDING RATE BOT - FINAL SUMMARY")
            logger.info("=" * 60)
            logger.info(f"⏰ Total Runtime: {runtime}")
            logger.info(f"💰 Total Funding Earned: ${self.total_funding_earned:.2f}")
            
            if runtime.total_seconds() > 0:
                hourly_rate = (self.total_funding_earned / runtime.total_seconds()) * 3600
                daily_rate = hourly_rate * 24
                logger.info(f"📈 Hourly Funding Rate: ${hourly_rate:.4f}")
                logger.info(f"📈 Daily Funding Rate: ${daily_rate:.2f}")
            
            logger.info("🎯 Thank you for using Funding Rate Capture Bot!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error displaying final summary: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            delta_status = self.delta_neutral_manager.get_status() if self.delta_neutral_manager else {}
            funding_summary = self.funding_collector.get_funding_summary() if self.funding_collector else {}
            
            return {
                'is_running': self.is_running,
                'bot_type': 'funding_rate_capture',
                'runtime': str(datetime.now() - self.start_time),
                'total_funding_earned': self.total_funding_earned,
                'daily_funding_target': self.daily_funding_target,
                'delta_neutral_status': delta_status,
                'funding_opportunities': self.funding_collector.get_best_funding_opportunities() if self.funding_collector else [],
                'current_funding_rates': funding_summary,
                'paper_trading': self.config.paper_trading
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

# Placeholder classes for demonstration
class PlaceholderSpotClient:
    async def get_symbol_ticker(self, symbol):
        return {'price': '50000.00'}

class PlaceholderFuturesClient:
    async def get_premium_index(self, symbol):
        return {
            'lastFundingRate': '0.0001',
            'nextFundingTime': '1640995200000',
            'markPrice': '50000.00',
            'indexPrice': '50000.00'
        }

class PlaceholderRateLimiter:
    async def wait_for_request_weight(self, weight):
        return True

class OrderManager:
    def __init__(self, config):
        self.config = config
    
    async def cancel_all_orders(self):
        pass

class RiskManager:
    def __init__(self, config):
        self.config = config
    
    async def start_monitoring(self):
        pass
    
    async def stop_monitoring(self):
        pass

async def main():
    """Main entry point for funding rate capture bot"""
    try:
        # Load configuration
        config = load_config()
        
        # Ensure delta neutral strategy is selected
        config.strategy.type = 'delta_neutral'
        
        # Create and start funding rate bot
        bot = FundingRateBot(config)
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the funding rate capture bot
    asyncio.run(main())