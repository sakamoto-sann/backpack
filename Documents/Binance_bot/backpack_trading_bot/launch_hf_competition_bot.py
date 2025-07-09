#!/usr/bin/env python3
"""
🏆 HIGH-FREQUENCY COMPETITION BOT LAUNCHER
Advanced launcher for the High-Frequency Delta-Neutral Competition Bot

Features:
- Ultra-fast deployment and monitoring
- Real-time performance optimization
- Competition ranking tracking
- Emergency procedures and risk management
- Advanced analytics and reporting
"""

import asyncio
import logging
import json
import yaml
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backpack_hf_delta_neutral_bot import BackpackHFDeltaNeutralBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_competition_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HFCompetitionLauncher:
    """
    High-Frequency Competition Bot Launcher
    
    Advanced launcher with real-time optimization and ultra-fast monitoring
    """
    
    def __init__(self, config_path: str = "config/hf_delta_neutral_config.yaml"):
        """
        Initialize HF competition launcher.
        
        Args:
            config_path: Path to HF configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.bot = None
        self.is_running = False
        
        # Competition tracking
        self.competition_start_time = datetime.now()
        self.performance_history = []
        self.optimization_log = []
        
        # HF Performance targets
        self.volume_target = self.config.get('performance_targets', {}).get('volume_targets', {}).get('daily_volume', 100000)
        self.transaction_target = self.config.get('performance_targets', {}).get('transaction_targets', {}).get('daily_transactions', 3000)
        self.pnl_target = self.config.get('performance_targets', {}).get('pnl_targets', {}).get('daily_return', 0.012)
        
        logger.info("🚀 HF Competition Bot Launcher initialized")
    
    def _load_config(self) -> dict:
        """Load HF configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading HF config: {e}")
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
    
    async def launch_hf_competition_bot(self):
        """Launch the high-frequency competition bot"""
        try:
            logger.info("🚀 Launching High-Frequency Delta-Neutral Competition Bot")
            
            # Display HF competition information
            await self._display_hf_competition_info()
            
            # Pre-launch checks
            await self._hf_pre_launch_checks()
            
            # Initialize HF bot
            self.bot = BackpackHFDeltaNeutralBot(self.config)
            
            # Start HF monitoring tasks
            await self._start_hf_monitoring_tasks()
            
            # Launch HF bot
            await self.bot.start()
            
        except Exception as e:
            logger.error(f"Error launching HF competition bot: {e}")
            raise
    
    async def _display_hf_competition_info(self):
        """Display HF competition information and targets"""
        try:
            logger.info("=" * 80)
            logger.info("🏆 HIGH-FREQUENCY DELTA-NEUTRAL COMPETITION BOT")
            logger.info("=" * 80)
            
            # HF Competition targets
            volume_targets = self.config.get('performance_targets', {}).get('volume_targets', {})
            transaction_targets = self.config.get('performance_targets', {}).get('transaction_targets', {})
            pnl_targets = self.config.get('performance_targets', {}).get('pnl_targets', {})
            
            logger.info("🎯 HF Competition Targets:")
            logger.info(f"   📊 Daily Volume: ${volume_targets.get('daily_volume', 0):,}")
            logger.info(f"   🔄 Daily Transactions: {transaction_targets.get('daily_transactions', 0):,}")
            logger.info(f"   📈 Daily Return: {pnl_targets.get('daily_return', 0):.1%}")
            logger.info(f"   🏆 Volume Rank Target: #{volume_targets.get('volume_rank_target', 0)}")
            logger.info(f"   🏆 PnL Rank Target: #{pnl_targets.get('pnl_rank_target', 0)}")
            
            # HF Trading parameters
            hf_config = self.config.get('high_frequency', {})
            logger.info(f"⚡ HF Parameters:")
            logger.info(f"   Grid Update: {hf_config.get('grid_update_interval', 0)}s")
            logger.info(f"   Rebalance: {hf_config.get('rebalance_interval', 0)}s")
            logger.info(f"   Arbitrage Scan: {hf_config.get('arbitrage_scan_interval', 0)}s")
            
            # SOL collateral info
            sol_config = self.config.get('sol_collateral', {})
            logger.info(f"💰 SOL Collateral: {sol_config.get('starting_amount', 0)} SOL")
            logger.info(f"🎯 Target Utilization: {sol_config.get('collateral_utilization_target', 0):.1%}")
            logger.info(f"🏦 Auto Lending: {sol_config.get('auto_lending', {}).get('lending_percentage', 0):.1%}")
            
            # Delta-neutral info
            delta_config = self.config.get('delta_neutral', {})
            logger.info(f"⚖️ Delta-Neutral: Target {delta_config.get('target_delta', 0):.1%}, Tolerance {delta_config.get('delta_tolerance', 0):.1%}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying HF competition info: {e}")
    
    async def _hf_pre_launch_checks(self):
        """Perform HF-specific pre-launch checks"""
        try:
            logger.info("🔍 Performing HF pre-launch checks...")
            
            # Check API credentials
            api_key = self.config.get('api', {}).get('key')
            if not api_key or api_key.startswith('${'):
                raise ValueError("BACKPACK_API_KEY environment variable not set")
            
            # Check HF configuration
            hf_config = self.config.get('high_frequency', {})
            if not hf_config.get('volume_target'):
                raise ValueError("HF volume target not configured")
            
            # Check delta-neutral configuration
            delta_config = self.config.get('delta_neutral', {})
            if not delta_config.get('enabled'):
                raise ValueError("Delta-neutral strategy not enabled")
            
            # Check required directories
            Path('logs').mkdir(exist_ok=True)
            Path('data').mkdir(exist_ok=True)
            Path('data/hf_performance').mkdir(exist_ok=True)
            
            # Validate HF configuration
            await self._validate_hf_config()
            
            logger.info("✅ HF pre-launch checks completed")
            
        except Exception as e:
            logger.error(f"HF pre-launch check failed: {e}")
            raise
    
    async def _validate_hf_config(self):
        """Validate HF-specific configuration settings"""
        try:
            # Validate HF intervals
            hf_config = self.config.get('high_frequency', {})
            grid_interval = hf_config.get('grid_update_interval', 5)
            if grid_interval < 3:
                logger.warning("Grid update interval very aggressive (<3s)")
            
            rebalance_interval = hf_config.get('rebalance_interval', 15)
            if rebalance_interval < 10:
                logger.warning("Rebalance interval very aggressive (<10s)")
            
            # Validate volume targets
            volume_target = hf_config.get('volume_target', 100000)
            if volume_target > 200000:
                logger.warning(f"Very high volume target: ${volume_target:,}")
            
            # Validate delta-neutral settings
            delta_config = self.config.get('delta_neutral', {})
            delta_tolerance = delta_config.get('delta_tolerance', 0.02)
            if delta_tolerance > 0.05:
                logger.warning(f"High delta tolerance: {delta_tolerance:.1%}")
            
            # Validate collateral utilization
            sol_config = self.config.get('sol_collateral', {})
            target_utilization = sol_config.get('collateral_utilization_target', 0.85)
            if target_utilization > 0.90:
                logger.warning(f"Very high collateral utilization target: {target_utilization:.1%}")
            
            logger.info("✅ HF configuration validation completed")
            
        except Exception as e:
            logger.error(f"HF configuration validation failed: {e}")
            raise
    
    async def _start_hf_monitoring_tasks(self):
        """Start HF-specific monitoring and optimization tasks"""
        try:
            # Start HF monitoring tasks
            asyncio.create_task(self._hf_performance_monitor())
            asyncio.create_task(self._delta_neutral_monitor())
            asyncio.create_task(self._volume_optimization_monitor())
            asyncio.create_task(self._competition_tracker())
            asyncio.create_task(self._funding_rate_monitor())
            asyncio.create_task(self._hf_risk_monitor())
            asyncio.create_task(self._emergency_monitor())
            
            logger.info("🔄 All HF monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Error starting HF monitoring tasks: {e}")
            raise
    
    async def _hf_performance_monitor(self):
        """Monitor HF bot performance with high frequency"""
        while self.is_running:
            try:
                if self.bot:
                    # Get current HF performance
                    status = await self.bot.get_status()
                    
                    # Log HF performance metrics
                    await self._log_hf_performance_metrics(status)
                    
                    # Store performance history
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'volume': status.get('total_volume', 0),
                        'transactions': status.get('transaction_count', 0),
                        'pnl': status.get('total_pnl', 0),
                        'delta_exposure': status.get('total_delta', 0),
                        'funding_income': status.get('funding_income', 0),
                        'collateral_usage': status.get('collateral_utilization', 0)
                    })
                    
                    # Keep only recent history (last 2 hours)
                    if len(self.performance_history) > 720:  # 2 hours of 10-second data
                        self.performance_history = self.performance_history[-720:]
                
                await asyncio.sleep(10)  # Check every 10 seconds for HF
                
            except Exception as e:
                logger.error(f"Error in HF performance monitor: {e}")
                await asyncio.sleep(10)
    
    async def _log_hf_performance_metrics(self, status: dict):
        """Log detailed HF performance metrics"""
        try:
            current_time = datetime.now()
            runtime = current_time - self.competition_start_time
            
            # Calculate performance ratios
            volume_ratio = status.get('total_volume', 0) / self.volume_target
            transaction_ratio = status.get('transaction_count', 0) / self.transaction_target
            pnl_ratio = status.get('total_pnl', 0) / self.pnl_target
            
            # Log every 2 minutes for HF
            if current_time.minute % 2 == 0 and current_time.second < 10:
                logger.info("📊 HF PERFORMANCE METRICS:")
                logger.info(f"   ⏰ Runtime: {runtime}")
                logger.info(f"   📈 Volume: ${status.get('total_volume', 0):,.2f} ({volume_ratio:.1%} of target)")
                logger.info(f"   🔄 Transactions: {status.get('transaction_count', 0)} ({transaction_ratio:.1%} of target)")
                logger.info(f"   💰 PnL: ${status.get('total_pnl', 0):.4f} ({pnl_ratio:.1%} of target)")
                logger.info(f"   ⚖️ Delta: {status.get('total_delta', 0):.2%}")
                logger.info(f"   🏦 Funding: ${status.get('funding_income', 0):.4f}")
                logger.info(f"   📊 Collateral: {status.get('collateral_utilization', 0):.1%}")
                
        except Exception as e:
            logger.error(f"Error logging HF performance metrics: {e}")
    
    async def _delta_neutral_monitor(self):
        """Monitor delta-neutral performance"""
        while self.is_running:
            try:
                if self.bot:
                    # Get delta-neutral metrics
                    delta_metrics = await self.bot.get_delta_metrics()
                    
                    # Check delta exposure
                    total_delta = delta_metrics.get('total_delta', 0)
                    if abs(total_delta) > 0.03:  # 3% warning threshold
                        logger.warning(f"⚠️ High delta exposure: {total_delta:.2%}")
                    
                    # Log delta performance
                    await self._log_delta_performance(delta_metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in delta-neutral monitor: {e}")
                await asyncio.sleep(30)
    
    async def _log_delta_performance(self, metrics: dict):
        """Log delta-neutral performance"""
        try:
            logger.info("⚖️ DELTA-NEUTRAL STATUS:")
            logger.info(f"   Delta Exposure: {metrics.get('total_delta', 0):.2%}")
            logger.info(f"   Hedge Ratio: {metrics.get('hedge_ratio', 0):.2%}")
            logger.info(f"   Rebalance Count: {metrics.get('rebalance_count', 0)}")
            logger.info(f"   Funding Income: ${metrics.get('funding_income', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error logging delta performance: {e}")
    
    async def _volume_optimization_monitor(self):
        """Monitor and optimize volume generation"""
        while self.is_running:
            try:
                if self.bot:
                    # Get volume metrics
                    volume_metrics = await self.bot.get_volume_metrics()
                    
                    # Check volume performance
                    current_volume = volume_metrics.get('current_volume', 0)
                    runtime_hours = (datetime.now() - self.competition_start_time).total_seconds() / 3600
                    hourly_volume = current_volume / max(runtime_hours, 0.1)
                    
                    # Optimize if below target
                    if hourly_volume < 4167:  # $4167/hour for $100k daily
                        await self._optimize_volume_generation()
                    
                    # Log volume optimization
                    await self._log_volume_optimization(volume_metrics, hourly_volume)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in volume optimization monitor: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_volume_generation(self):
        """Optimize volume generation strategies"""
        try:
            if self.bot:
                # Increase trading frequency
                await self.bot._increase_trading_frequency()
                
                # Optimize grid spacing
                await self.bot._optimize_grid_spacing_for_volume()
                
                # Increase cross-pair arbitrage
                await self.bot._increase_arbitrage_frequency()
                
                logger.info("📈 Applied volume optimization")
            
        except Exception as e:
            logger.error(f"Error optimizing volume generation: {e}")
    
    async def _log_volume_optimization(self, metrics: dict, hourly_volume: float):
        """Log volume optimization metrics"""
        try:
            logger.info("📈 VOLUME OPTIMIZATION:")
            logger.info(f"   Current Volume: ${metrics.get('current_volume', 0):,.2f}")
            logger.info(f"   Hourly Rate: ${hourly_volume:,.2f}/hour")
            logger.info(f"   Target Rate: $4,167/hour")
            logger.info(f"   Volume Efficiency: {metrics.get('volume_efficiency', 0):.1f}x")
            
        except Exception as e:
            logger.error(f"Error logging volume optimization: {e}")
    
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
        """Log HF competition status"""
        try:
            logger.info("🏆 HF COMPETITION STATUS:")
            logger.info(f"   📊 Volume Rank: #{metrics.get('volume_rank_estimate', 0)}")
            logger.info(f"   📈 PnL Rank: #{metrics.get('pnl_rank_estimate', 0)}")
            logger.info(f"   🎯 Daily Volume: ${metrics.get('daily_volume', 0):,.2f}")
            logger.info(f"   💰 Daily PnL: ${metrics.get('daily_pnl', 0):.4f}")
            logger.info(f"   🔄 Transactions: {metrics.get('transaction_count', 0)}")
            
        except Exception as e:
            logger.error(f"Error logging HF competition status: {e}")
    
    async def _check_ranking_changes(self, metrics: dict):
        """Check for significant ranking changes"""
        try:
            volume_rank = metrics.get('volume_rank_estimate', 0)
            pnl_rank = metrics.get('pnl_rank_estimate', 0)
            
            # Check if we're meeting HF targets
            if volume_rank <= 5:
                logger.info(f"🎉 HF Volume rank target achieved: #{volume_rank}")
            
            if pnl_rank <= 3:
                logger.info(f"🎉 HF PnL rank target achieved: #{pnl_rank}")
            
        except Exception as e:
            logger.error(f"Error checking HF ranking changes: {e}")
    
    async def _funding_rate_monitor(self):
        """Monitor funding rate opportunities"""
        while self.is_running:
            try:
                if self.bot:
                    # Get funding rate metrics
                    funding_metrics = await self.bot.get_funding_metrics()
                    
                    # Log funding performance
                    await self._log_funding_performance(funding_metrics)
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in funding rate monitor: {e}")
                await asyncio.sleep(120)
    
    async def _log_funding_performance(self, metrics: dict):
        """Log funding rate performance"""
        try:
            logger.info("💰 FUNDING RATE STATUS:")
            logger.info(f"   Total Funding Income: ${metrics.get('total_funding_income', 0):.4f}")
            logger.info(f"   Active Funding Positions: {metrics.get('funding_position_count', 0)}")
            logger.info(f"   Avg Funding Rate: {metrics.get('avg_funding_rate', 0):.4%}")
            
        except Exception as e:
            logger.error(f"Error logging funding performance: {e}")
    
    async def _hf_risk_monitor(self):
        """Monitor HF-specific risk metrics"""
        while self.is_running:
            try:
                if self.bot:
                    # Get HF risk metrics
                    risk_metrics = await self.bot.get_hf_risk_metrics()
                    
                    # Check HF risk thresholds
                    await self._check_hf_risk_thresholds(risk_metrics)
                
                await asyncio.sleep(15)  # Check every 15 seconds for HF
                
            except Exception as e:
                logger.error(f"Error in HF risk monitor: {e}")
                await asyncio.sleep(15)
    
    async def _check_hf_risk_thresholds(self, risk_metrics: dict):
        """Check HF-specific risk thresholds"""
        try:
            # Check execution speed
            avg_execution_time = risk_metrics.get('avg_execution_time', 0)
            if avg_execution_time > 200:  # 200ms threshold
                logger.warning(f"⚠️ Slow execution: {avg_execution_time}ms")
            
            # Check order fill rate
            fill_rate = risk_metrics.get('order_fill_rate', 1.0)
            if fill_rate < 0.90:  # 90% minimum
                logger.warning(f"⚠️ Low fill rate: {fill_rate:.1%}")
            
            # Check delta exposure
            total_delta = risk_metrics.get('total_delta', 0)
            if abs(total_delta) > 0.05:  # 5% max
                logger.warning(f"⚠️ High delta exposure: {total_delta:.2%}")
            
        except Exception as e:
            logger.error(f"Error checking HF risk thresholds: {e}")
    
    async def _emergency_monitor(self):
        """Monitor for emergency conditions"""
        while self.is_running:
            try:
                if self.bot:
                    # Get emergency metrics
                    emergency_metrics = await self.bot.get_emergency_metrics()
                    
                    # Check emergency conditions
                    await self._check_emergency_conditions(emergency_metrics)
                
                await asyncio.sleep(5)  # Check every 5 seconds for emergencies
                
            except Exception as e:
                logger.error(f"Error in emergency monitor: {e}")
                await asyncio.sleep(5)
    
    async def _check_emergency_conditions(self, metrics: dict):
        """Check for emergency conditions"""
        try:
            # Check for emergency stop conditions
            if metrics.get('emergency_stop_required', False):
                logger.error("🚨 EMERGENCY STOP REQUIRED!")
                await self._execute_emergency_stop()
            
            # Check for high delta exposure
            if metrics.get('high_delta_exposure', False):
                logger.error("🚨 HIGH DELTA EXPOSURE DETECTED!")
                await self._execute_emergency_delta_rebalance()
            
            # Check for API issues
            if metrics.get('api_errors', 0) > 5:
                logger.error("🚨 MULTIPLE API ERRORS DETECTED!")
                await self._execute_api_recovery()
            
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
    
    async def _execute_emergency_delta_rebalance(self):
        """Execute emergency delta rebalancing"""
        try:
            logger.warning("⚠️ EXECUTING EMERGENCY DELTA REBALANCE")
            
            if self.bot:
                await self.bot.emergency_delta_rebalance()
            
        except Exception as e:
            logger.error(f"Error executing emergency delta rebalance: {e}")
    
    async def _execute_api_recovery(self):
        """Execute API recovery procedures"""
        try:
            logger.warning("⚠️ EXECUTING API RECOVERY")
            
            if self.bot:
                await self.bot.recover_api_connections()
            
        except Exception as e:
            logger.error(f"Error executing API recovery: {e}")
    
    async def stop(self):
        """Stop the HF competition bot launcher"""
        try:
            logger.info("🛑 Stopping HF competition bot launcher...")
            
            self.is_running = False
            
            if self.bot:
                await self.bot.stop()
            
            # Generate final HF report
            await self._generate_final_hf_report()
            
            logger.info("✅ HF competition bot launcher stopped")
            
        except Exception as e:
            logger.error(f"Error stopping HF launcher: {e}")
    
    async def _generate_final_hf_report(self):
        """Generate final HF competition report"""
        try:
            runtime = datetime.now() - self.competition_start_time
            
            if self.bot:
                final_status = await self.bot.get_status()
                
                logger.info("=" * 80)
                logger.info("🏆 FINAL HF COMPETITION REPORT")
                logger.info("=" * 80)
                logger.info(f"⏰ Total Runtime: {runtime}")
                logger.info(f"📊 Final Volume: ${final_status.get('total_volume', 0):,.2f}")
                logger.info(f"🎯 Volume Target: ${self.volume_target:,.2f}")
                logger.info(f"📈 Volume Achievement: {(final_status.get('total_volume', 0)/self.volume_target)*100:.1f}%")
                logger.info(f"🔄 Total Transactions: {final_status.get('transaction_count', 0)}")
                logger.info(f"🎯 Transaction Target: {self.transaction_target}")
                logger.info(f"💰 Final PnL: ${final_status.get('total_pnl', 0):.4f}")
                logger.info(f"🏦 Funding Income: ${final_status.get('funding_income', 0):.4f}")
                logger.info(f"⚖️ Final Delta: {final_status.get('total_delta', 0):.2%}")
                logger.info(f"📊 Collateral Usage: {final_status.get('collateral_utilization', 0):.1%}")
                logger.info(f"🏆 Volume Rank Estimate: #{final_status.get('volume_rank_estimate', 0)}")
                logger.info(f"🏆 PnL Rank Estimate: #{final_status.get('pnl_rank_estimate', 0)}")
                logger.info(f"🔧 Optimizations Applied: {len(self.optimization_log)}")
                logger.info("=" * 80)
                
                # Save detailed HF report
                await self._save_detailed_hf_report(final_status, runtime)
            
        except Exception as e:
            logger.error(f"Error generating final HF report: {e}")
    
    async def _save_detailed_hf_report(self, status: dict, runtime: timedelta):
        """Save detailed HF competition report"""
        try:
            report = {
                'hf_competition_summary': {
                    'start_time': self.competition_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'runtime': str(runtime),
                    'final_status': status,
                    'targets': {
                        'volume_target': self.volume_target,
                        'transaction_target': self.transaction_target,
                        'pnl_target': self.pnl_target
                    }
                },
                'hf_performance_history': self.performance_history,
                'optimization_log': self.optimization_log
            }
            
            # Save to file
            report_path = Path(f"data/hf_performance/hf_competition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Detailed HF report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving detailed HF report: {e}")

async def main():
    """Main HF launcher function"""
    parser = argparse.ArgumentParser(description='Backpack HF Competition Bot Launcher')
    parser.add_argument('--config', '-c', default='config/hf_delta_neutral_config.yaml',
                       help='HF Configuration file path')
    parser.add_argument('--hf-mode', action='store_true',
                       help='Enable high-frequency mode')
    parser.add_argument('--competition-mode', action='store_true',
                       help='Enable competition mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in simulation mode')
    
    args = parser.parse_args()
    
    try:
        # Create HF launcher
        launcher = HFCompetitionLauncher(args.config)
        
        # Set running flag
        launcher.is_running = True
        
        # Launch HF competition bot
        await launcher.launch_hf_competition_bot()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down HF bot...")
        await launcher.stop()
    except Exception as e:
        logger.error(f"Fatal HF error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/hf_performance").mkdir(exist_ok=True)
    
    # Run HF launcher
    asyncio.run(main())
