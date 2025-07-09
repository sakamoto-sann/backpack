#!/usr/bin/env python3
"""
🏆 BACKPACK COMPETITION BOT LAUNCHER
Advanced launcher with real-time optimization and competition monitoring

Features:
- Real-time performance monitoring
- Dynamic parameter optimization
- Competition ranking tracking
- Emergency procedures
- Performance analytics
"""

import asyncio
import logging
import json
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backpack_sol_bot import BackpackSOLBot, TradingMode
from config.backpack_config import BackpackConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/competition_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompetitionBotLauncher:
    """
    Advanced Competition Bot Launcher
    
    Manages bot deployment, monitoring, and optimization for Backpack competitions
    """
    
    def __init__(self, config_path: str = "config/backpack_config.yaml"):
        """
        Initialize competition bot launcher.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.bot = None
        self.is_running = False
        
        # Competition tracking
        self.competition_start_time = datetime.now()
        self.performance_history = []
        self.optimization_log = []
        
        # Performance targets
        self.volume_target = self.config.get('performance_targets', {}).get('volume_targets', {}).get('daily_volume', 50000)
        self.pnl_target = self.config.get('performance_targets', {}).get('pnl_targets', {}).get('daily_return', 0.008)
        
        logger.info("🏆 Competition Bot Launcher initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _substitute_env_vars(self, config: dict) -> dict:
        """Substitute environment variables in config"""
        import os
        
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    async def launch_competition_bot(self):
        """Launch the competition bot with monitoring"""
        try:
            logger.info("🚀 Launching Backpack Competition Bot")
            
            # Display competition information
            await self._display_competition_info()
            
            # Pre-launch checks
            await self._pre_launch_checks()
            
            # Initialize bot
            self.bot = BackpackSOLBot(self.config)
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            # Launch bot
            await self.bot.start()
            
        except Exception as e:
            logger.error(f"Error launching competition bot: {e}")
            raise
    
    async def _display_competition_info(self):
        """Display competition information and targets"""
        try:
            logger.info("=" * 80)
            logger.info("🏆 BACKPACK COMPETITION BOT LAUNCH")
            logger.info("=" * 80)
            
            # Competition targets
            volume_targets = self.config.get('performance_targets', {}).get('volume_targets', {})
            pnl_targets = self.config.get('performance_targets', {}).get('pnl_targets', {})
            
            logger.info("🎯 Competition Targets:")
            logger.info(f"   📊 Daily Volume: ${volume_targets.get('daily_volume', 0):,}")
            logger.info(f"   📈 Monthly Return: {pnl_targets.get('monthly_return', 0):.1%}")
            logger.info(f"   🏆 Volume Rank Target: #{volume_targets.get('volume_rank_target', 0)}")
            logger.info(f"   🏆 PnL Rank Target: #{pnl_targets.get('pnl_rank_target', 0)}")
            
            # SOL collateral info
            sol_config = self.config.get('sol_collateral', {})
            logger.info(f"💰 SOL Collateral: {sol_config.get('starting_amount', 0)} SOL")
            logger.info(f"🏦 Auto Lending: {sol_config.get('auto_lending', {}).get('lending_percentage', 0):.1%}")
            
            # Trading pairs
            pairs = self.config.get('grid_trading', {}).get('pairs', [])
            logger.info(f"📊 Trading Pairs: {len(pairs)} pairs")
            for pair in pairs:
                logger.info(f"   {pair['symbol']}: {pair['allocation']:.1%} allocation")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying competition info: {e}")
    
    async def _pre_launch_checks(self):
        """Perform pre-launch checks"""
        try:
            logger.info("🔍 Performing pre-launch checks...")
            
            # Check API credentials
            api_key = self.config.get('api', {}).get('key')
            if not api_key or api_key.startswith('${'):
                raise ValueError("BACKPACK_API_KEY environment variable not set")
            
            # Check required directories
            Path('logs').mkdir(exist_ok=True)
            Path('data').mkdir(exist_ok=True)
            
            # Validate configuration
            await self._validate_config()
            
            logger.info("✅ Pre-launch checks completed")
            
        except Exception as e:
            logger.error(f"Pre-launch check failed: {e}")
            raise
    
    async def _validate_config(self):
        """Validate configuration settings"""
        try:
            # Validate SOL collateral settings
            sol_config = self.config.get('sol_collateral', {})
            if sol_config.get('starting_amount', 0) != 1.0:
                logger.warning("Starting amount is not 1.0 SOL as specified")
            
            # Validate grid trading settings
            pairs = self.config.get('grid_trading', {}).get('pairs', [])
            total_allocation = sum(pair.get('allocation', 0) for pair in pairs)
            if abs(total_allocation - 1.0) > 0.01:
                logger.warning(f"Total allocation is {total_allocation:.2%}, should be 100%")
            
            # Validate risk settings
            risk_config = self.config.get('risk_management', {})
            if risk_config.get('position_limits', {}).get('max_daily_loss', 0) > 0.10:
                logger.warning("Daily loss limit is high (>10%)")
            
            logger.info("✅ Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    async def _start_monitoring_tasks(self):
        """Start monitoring and optimization tasks"""
        try:
            # Start monitoring tasks
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._competition_tracker())
            asyncio.create_task(self._dynamic_optimizer())
            asyncio.create_task(self._risk_monitor())
            asyncio.create_task(self._emergency_monitor())
            
            logger.info("🔄 All monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {e}")
            raise
    
    async def _performance_monitor(self):
        """Monitor bot performance"""
        while self.is_running:
            try:
                if self.bot:
                    # Get current performance
                    status = await self.bot.get_status()
                    
                    # Log performance metrics
                    await self._log_performance_metrics(status)
                    
                    # Store performance history
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'volume': status.get('total_volume', 0),
                        'pnl': status.get('total_pnl', 0),
                        'collateral_usage': status.get('collateral_usage', 0)
                    })
                    
                    # Keep only recent history
                    if len(self.performance_history) > 1440:  # 24 hours of minute data
                        self.performance_history = self.performance_history[-1440:]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def _log_performance_metrics(self, status: dict):
        """Log detailed performance metrics"""
        try:
            current_time = datetime.now()
            runtime = current_time - self.competition_start_time
            
            # Calculate performance ratios
            volume_ratio = status.get('total_volume', 0) / self.volume_target
            pnl_ratio = status.get('total_pnl', 0) / self.pnl_target
            
            # Log every 10 minutes
            if current_time.minute % 10 == 0:
                logger.info("📊 PERFORMANCE METRICS:")
                logger.info(f"   ⏰ Runtime: {runtime}")
                logger.info(f"   📈 Volume: ${status.get('total_volume', 0):,.2f} ({volume_ratio:.1%} of target)")
                logger.info(f"   💰 PnL: ${status.get('total_pnl', 0):.4f} ({pnl_ratio:.1%} of target)")
                logger.info(f"   🏦 Lending: {status.get('lending_income', 0):.6f} SOL")
                logger.info(f"   ⚖️ Collateral: {status.get('collateral_usage', 0):.1%}")
                logger.info(f"   🔄 Transactions: {status.get('transaction_count', 0)}")
                
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    async def _competition_tracker(self):
        """Track competition rankings and progress"""
        while self.is_running:
            try:
                if self.bot:
                    # Get competition metrics
                    competition_metrics = await self.bot.get_competition_metrics()
                    
                    # Log competition status
                    await self._log_competition_status(competition_metrics)
                    
                    # Check for ranking improvements
                    await self._check_ranking_changes(competition_metrics)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in competition tracker: {e}")
                await asyncio.sleep(300)
    
    async def _log_competition_status(self, metrics: dict):
        """Log competition status"""
        try:
            logger.info("🏆 COMPETITION STATUS:")
            logger.info(f"   📊 Volume Rank: #{metrics.get('volume_rank_estimate', 0)}")
            logger.info(f"   📈 PnL Rank: #{metrics.get('pnl_rank_estimate', 0)}")
            logger.info(f"   🎯 Daily Volume: ${metrics.get('daily_volume', 0):,.2f}")
            logger.info(f"   💰 Daily PnL: ${metrics.get('daily_pnl', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error logging competition status: {e}")
    
    async def _check_ranking_changes(self, metrics: dict):
        """Check for significant ranking changes"""
        try:
            volume_rank = metrics.get('volume_rank_estimate', 0)
            pnl_rank = metrics.get('pnl_rank_estimate', 0)
            
            # Check if we're meeting targets
            volume_target_rank = self.config.get('performance_targets', {}).get('volume_targets', {}).get('volume_rank_target', 10)
            pnl_target_rank = self.config.get('performance_targets', {}).get('pnl_targets', {}).get('pnl_rank_target', 5)
            
            if volume_rank <= volume_target_rank:
                logger.info(f"🎉 Volume rank target achieved: #{volume_rank}")
            
            if pnl_rank <= pnl_target_rank:
                logger.info(f"🎉 PnL rank target achieved: #{pnl_rank}")
            
        except Exception as e:
            logger.error(f"Error checking ranking changes: {e}")
    
    async def _dynamic_optimizer(self):
        """Dynamically optimize bot parameters"""
        while self.is_running:
            try:
                if self.bot:
                    # Get current performance
                    performance = await self.bot.get_performance_metrics()
                    
                    # Optimize parameters based on performance
                    optimizations = await self._calculate_optimizations(performance)
                    
                    # Apply optimizations
                    if optimizations:
                        await self._apply_optimizations(optimizations)
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in dynamic optimizer: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_optimizations(self, performance: dict) -> dict:
        """Calculate optimal parameter adjustments"""
        try:
            optimizations = {}
            
            # Volume optimization
            if performance.get('volume_efficiency', 0) < 0.8:
                optimizations['reduce_grid_spacing'] = 0.9  # Reduce by 10%
                optimizations['increase_rebalance_frequency'] = 1.2  # Increase by 20%
            
            # PnL optimization
            if performance.get('pnl_efficiency', 0) < 0.8:
                optimizations['increase_grid_spacing'] = 1.1  # Increase by 10%
                optimizations['optimize_position_sizing'] = True
            
            # Collateral optimization
            if performance.get('collateral_efficiency', 0) < 0.8:
                optimizations['increase_collateral_usage'] = 1.1  # Increase by 10%
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error calculating optimizations: {e}")
            return {}
    
    async def _apply_optimizations(self, optimizations: dict):
        """Apply calculated optimizations"""
        try:
            for optimization, value in optimizations.items():
                if optimization == 'reduce_grid_spacing':
                    await self.bot.adjust_grid_spacing(value)
                elif optimization == 'increase_rebalance_frequency':
                    await self.bot.adjust_rebalance_frequency(value)
                elif optimization == 'optimize_position_sizing':
                    await self.bot.optimize_position_sizing()
                elif optimization == 'increase_collateral_usage':
                    await self.bot.adjust_collateral_usage(value)
                
                # Log optimization
                self.optimization_log.append({
                    'timestamp': datetime.now(),
                    'optimization': optimization,
                    'value': value
                })
                
                logger.info(f"🔧 Applied optimization: {optimization} = {value}")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
    
    async def _risk_monitor(self):
        """Monitor risk metrics"""
        while self.is_running:
            try:
                if self.bot:
                    # Get risk metrics
                    risk_metrics = await self.bot.get_risk_metrics()
                    
                    # Check risk thresholds
                    await self._check_risk_thresholds(risk_metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(30)
    
    async def _check_risk_thresholds(self, risk_metrics: dict):
        """Check risk thresholds and trigger alerts"""
        try:
            # Check collateral usage
            collateral_usage = risk_metrics.get('collateral_usage', 0)
            if collateral_usage > 0.90:  # 90% usage
                logger.warning(f"⚠️ High collateral usage: {collateral_usage:.1%}")
            
            # Check daily loss
            daily_loss = risk_metrics.get('daily_loss', 0)
            if daily_loss > 0.03:  # 3% daily loss
                logger.warning(f"⚠️ High daily loss: {daily_loss:.2%}")
            
            # Check drawdown
            drawdown = risk_metrics.get('drawdown', 0)
            if drawdown > 0.10:  # 10% drawdown
                logger.warning(f"⚠️ High drawdown: {drawdown:.2%}")
            
        except Exception as e:
            logger.error(f"Error checking risk thresholds: {e}")
    
    async def _emergency_monitor(self):
        """Monitor for emergency conditions"""
        while self.is_running:
            try:
                if self.bot:
                    # Get emergency metrics
                    emergency_metrics = await self.bot.get_emergency_metrics()
                    
                    # Check emergency conditions
                    await self._check_emergency_conditions(emergency_metrics)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in emergency monitor: {e}")
                await asyncio.sleep(10)
    
    async def _check_emergency_conditions(self, metrics: dict):
        """Check for emergency conditions"""
        try:
            # Check for emergency stop conditions
            if metrics.get('emergency_stop_required', False):
                logger.error("🚨 EMERGENCY STOP REQUIRED!")
                await self._execute_emergency_stop()
            
            # Check for collateral liquidation risk
            if metrics.get('liquidation_risk', False):
                logger.error("🚨 LIQUIDATION RISK DETECTED!")
                await self._execute_emergency_position_reduction()
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
    
    async def _execute_emergency_stop(self):
        """Execute emergency stop procedures"""
        try:
            logger.error("🚨 EXECUTING EMERGENCY STOP")
            
            if self.bot:
                await self.bot.emergency_stop()
            
            self.is_running = False
            
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
    
    async def _execute_emergency_position_reduction(self):
        """Execute emergency position reduction"""
        try:
            logger.warning("⚠️ EXECUTING EMERGENCY POSITION REDUCTION")
            
            if self.bot:
                await self.bot.emergency_position_reduction()
            
        except Exception as e:
            logger.error(f"Error executing emergency position reduction: {e}")
    
    async def stop(self):
        """Stop the competition bot launcher"""
        try:
            logger.info("🛑 Stopping competition bot launcher...")
            
            self.is_running = False
            
            if self.bot:
                await self.bot.stop()
            
            # Generate final report
            await self._generate_final_report()
            
            logger.info("✅ Competition bot launcher stopped")
            
        except Exception as e:
            logger.error(f"Error stopping launcher: {e}")
    
    async def _generate_final_report(self):
        """Generate final competition report"""
        try:
            runtime = datetime.now() - self.competition_start_time
            
            if self.bot:
                final_status = await self.bot.get_status()
                
                logger.info("=" * 80)
                logger.info("🏆 FINAL COMPETITION REPORT")
                logger.info("=" * 80)
                logger.info(f"⏰ Total Runtime: {runtime}")
                logger.info(f"📊 Final Volume: ${final_status.get('total_volume', 0):,.2f}")
                logger.info(f"📈 Final PnL: ${final_status.get('total_pnl', 0):.4f}")
                logger.info(f"🏦 Lending Income: {final_status.get('lending_income', 0):.6f} SOL")
                logger.info(f"🔄 Total Transactions: {final_status.get('transaction_count', 0)}")
                logger.info(f"🏆 Estimated Volume Rank: #{final_status.get('volume_rank_estimate', 0)}")
                logger.info(f"🏆 Estimated PnL Rank: #{final_status.get('pnl_rank_estimate', 0)}")
                logger.info(f"🔧 Optimizations Applied: {len(self.optimization_log)}")
                logger.info("=" * 80)
                
                # Save detailed report
                await self._save_detailed_report(final_status, runtime)
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    async def _save_detailed_report(self, status: dict, runtime: timedelta):
        """Save detailed competition report"""
        try:
            report = {
                'competition_summary': {
                    'start_time': self.competition_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'runtime': str(runtime),
                    'final_status': status
                },
                'performance_history': self.performance_history,
                'optimization_log': self.optimization_log
            }
            
            # Save to file
            report_path = Path(f"data/competition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Detailed report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving detailed report: {e}")

async def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Backpack Competition Bot Launcher')
    parser.add_argument('--config', '-c', default='config/backpack_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--competition-mode', action='store_true',
                       help='Enable competition mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in simulation mode')
    
    args = parser.parse_args()
    
    try:
        # Create launcher
        launcher = CompetitionBotLauncher(args.config)
        
        # Set running flag
        launcher.is_running = True
        
        # Launch competition bot
        await launcher.launch_competition_bot()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        await launcher.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run launcher
    asyncio.run(main())