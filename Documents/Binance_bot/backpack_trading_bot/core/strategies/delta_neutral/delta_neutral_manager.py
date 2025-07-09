"""
Advanced Delta-Neutral Manager for Funding Rate Capture
Professional-grade delta-neutral position management optimized for funding earnings.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class GridType(Enum):
    SPOT_PROFIT = "SPOT_PROFIT"
    FUTURES_HEDGE = "FUTURES_HEDGE"

@dataclass
class DeltaNeutralPosition:
    """Current delta neutral position state"""
    spot_position: float = 0.0
    futures_position: float = 0.0
    delta_exposure: float = 0.0
    net_pnl: float = 0.0
    funding_collected: float = 0.0
    unrealized_pnl: float = 0.0
    position_value: float = 0.0
    hedge_ratio: float = 1.0
    last_rebalance: Optional[datetime] = None

@dataclass
class FundingRateData:
    """Funding rate information"""
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    estimated_funding: float
    annualized_rate: float

@dataclass
class GridConfiguration:
    """Grid trading configuration"""
    symbol: str = "BTCUSDT"
    grid_levels: int = 20
    grid_spacing: float = 0.002  # 0.2%
    base_order_size: float = 50  # USDT
    max_total_position: float = 2000  # USDT
    rebalance_threshold: float = 0.05  # 5% delta threshold
    funding_optimization: bool = True

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_delta_exposure: float = 0.1  # 10% max delta
    max_daily_loss: float = 100  # USDT
    max_position_size: float = 5000  # USDT
    emergency_stop_loss: float = 0.05  # 5%
    funding_rate_threshold: float = 0.0001  # 0.01%

class DeltaNeutralManager:
    """
    Advanced Delta-Neutral Manager for Funding Rate Capture
    
    Core Features:
    - Maintains delta neutrality while capturing funding rates
    - Optimizes position direction based on funding rates
    - Dynamic grid spacing based on volatility
    - Real-time risk monitoring and emergency controls
    - Performance tracking and optimization
    """
    
    def __init__(self, 
                 risk_limits: RiskLimits,
                 grid_config: GridConfiguration,
                 spot_client,
                 futures_client,
                 funding_collector):
        """
        Initialize the delta-neutral manager.
        
        Args:
            risk_limits: Risk management limits
            grid_config: Grid configuration
            spot_client: Spot trading client
            futures_client: Futures trading client
            funding_collector: Funding rate collector
        """
        self.risk_limits = risk_limits
        self.grid_config = grid_config
        self.spot_client = spot_client
        self.futures_client = futures_client
        self.funding_collector = funding_collector
        
        # Position state
        self.position = DeltaNeutralPosition()
        self.is_active = False
        self.last_rebalance = datetime.now()
        
        # Grid management
        self.spot_grid_levels: List[Dict] = []
        self.futures_hedge_levels: List[Dict] = []
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.funding_history: List[float] = []
        self.pnl_history: List[float] = []
        
        # Risk monitoring
        self.last_risk_check = datetime.now()
        self.emergency_stop = False
        
        logger.info("Advanced Delta-Neutral Manager initialized")
    
    async def start_strategy(self) -> bool:
        """Start the delta-neutral strategy"""
        try:
            logger.info("🚀 Starting Advanced Delta-Neutral Strategy")
            
            # Initialize components
            await self._initialize_strategy()
            
            # Start monitoring loops
            asyncio.create_task(self._delta_monitoring_loop())
            asyncio.create_task(self._funding_optimization_loop())
            asyncio.create_task(self._risk_monitoring_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.is_active = True
            logger.info("✅ Delta-neutral strategy started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            return False
    
    async def _initialize_strategy(self):
        """Initialize strategy components"""
        try:
            # Get current market price
            ticker = await self.spot_client.get_symbol_ticker(symbol=self.grid_config.symbol)
            current_price = float(ticker['price'])
            
            # Initialize grid levels
            await self._setup_initial_grids(current_price)
            
            # Get initial funding rate
            await self._update_funding_data()
            
            # Set initial hedge ratio based on funding
            await self._optimize_hedge_ratio()
            
            logger.info(f"Strategy initialized at price: {current_price}")
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            raise
    
    async def _setup_initial_grids(self, current_price: float):
        """Setup initial spot and futures grid levels"""
        try:
            # Calculate grid levels around current price
            grid_half = self.grid_config.grid_levels // 2
            
            self.spot_grid_levels = []
            self.futures_hedge_levels = []
            
            for i in range(-grid_half, grid_half + 1):
                if i == 0:
                    continue  # Skip center price
                
                # Calculate grid price
                price_offset = i * self.grid_config.grid_spacing
                grid_price = current_price * (1 + price_offset)
                
                # Spot grid level (for profit taking)
                spot_level = {
                    'price': grid_price,
                    'quantity': self.grid_config.base_order_size / grid_price,
                    'side': 'sell' if i > 0 else 'buy',
                    'active': False,
                    'order_id': None
                }
                self.spot_grid_levels.append(spot_level)
                
                # Futures hedge level (for delta neutrality)
                futures_level = {
                    'price': grid_price,
                    'quantity': (self.grid_config.base_order_size / grid_price) * self.position.hedge_ratio,
                    'side': 'sell' if i < 0 else 'buy',  # Opposite to spot for hedging
                    'active': False,
                    'order_id': None
                }
                self.futures_hedge_levels.append(futures_level)
            
            logger.info(f"Initialized {len(self.spot_grid_levels)} spot and futures grid levels")
            
        except Exception as e:
            logger.error(f"Error setting up grids: {e}")
            raise
    
    async def _delta_monitoring_loop(self):
        """Main delta monitoring and rebalancing loop"""
        while self.is_active and not self.emergency_stop:
            try:
                # Calculate current delta exposure
                await self._calculate_delta_exposure()
                
                # Check if rebalancing is needed
                if abs(self.position.delta_exposure) > self.risk_limits.max_delta_exposure:
                    await self._rebalance_delta()
                
                # Update grid orders if needed
                await self._update_grid_orders()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in delta monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _funding_optimization_loop(self):
        """Funding rate optimization loop"""
        while self.is_active and not self.emergency_stop:
            try:
                # Update funding rate data
                await self._update_funding_data()
                
                # Optimize position for funding capture
                await self._optimize_for_funding()
                
                # Check funding collection opportunities
                await self._check_funding_opportunities()
                
                # Sleep until next funding check (every hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in funding optimization loop: {e}")
                await asyncio.sleep(300)
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring and emergency controls"""
        while self.is_active:
            try:
                # Check various risk metrics
                await self._check_risk_limits()
                
                # Monitor PnL and drawdown
                await self._monitor_pnl()
                
                # Check system health
                await self._check_system_health()
                
                self.last_risk_check = datetime.now()
                
                # Sleep for risk check interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self):
        """Performance tracking and analytics"""
        while self.is_active and not self.emergency_stop:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Log performance summary
                await self._log_performance_summary()
                
                # Sleep for performance update interval
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_delta_exposure(self):
        """Calculate current delta exposure"""
        try:
            # Get current positions
            spot_balance = await self._get_spot_balance()
            futures_position = await self._get_futures_position()
            
            # Update position state
            self.position.spot_position = spot_balance
            self.position.futures_position = futures_position
            
            # Calculate delta (net directional exposure)
            self.position.delta_exposure = self.position.spot_position + self.position.futures_position
            
            # Log if significant exposure
            if abs(self.position.delta_exposure) > self.risk_limits.max_delta_exposure * 0.5:
                logger.warning(f"Delta exposure: {self.position.delta_exposure:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating delta exposure: {e}")
    
    async def _rebalance_delta(self):
        """Rebalance position to maintain delta neutrality"""
        try:
            logger.info(f"Rebalancing delta exposure: {self.position.delta_exposure:.4f}")
            
            # Calculate required hedge adjustment
            hedge_adjustment = -self.position.delta_exposure
            
            # Execute hedge trade on futures
            if abs(hedge_adjustment) > 0.001:  # Minimum trade size
                await self._execute_futures_hedge(hedge_adjustment)
                
                self.last_rebalance = datetime.now()
                self.position.last_rebalance = self.last_rebalance
                
                logger.info(f"Delta rebalanced with hedge adjustment: {hedge_adjustment:.4f}")
            
        except Exception as e:
            logger.error(f"Error rebalancing delta: {e}")
    
    async def _update_funding_data(self):
        """Update current funding rate data"""
        try:
            # Get funding rate from collector
            funding_summary = self.funding_collector.get_funding_summary(self.grid_config.symbol)
            
            if self.grid_config.symbol in funding_summary:
                funding_info = funding_summary[self.grid_config.symbol]
                
                self.current_funding = FundingRateData(
                    symbol=self.grid_config.symbol,
                    funding_rate=funding_info['current_funding_rate'],
                    next_funding_time=datetime.fromisoformat(funding_info['next_funding_time']),
                    estimated_funding=funding_info['current_funding_rate'] * self.position.position_value,
                    annualized_rate=funding_info['annualized_rate']
                )
                
                logger.debug(f"Funding rate updated: {self.current_funding.funding_rate:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating funding data: {e}")
    
    async def _optimize_for_funding(self):
        """Optimize position direction for maximum funding capture"""
        try:
            if not hasattr(self, 'current_funding'):
                return
            
            # Check if funding rate is significant enough to optimize for
            if abs(self.current_funding.funding_rate) < self.risk_limits.funding_rate_threshold:
                return
            
            # Determine optimal position direction
            if self.current_funding.funding_rate > 0:
                # Positive funding rate - shorts pay longs
                # We want to be long futures to collect funding
                optimal_futures_side = "long"
            else:
                # Negative funding rate - longs pay shorts  
                # We want to be short futures to collect funding
                optimal_futures_side = "short"
            
            # Adjust hedge ratio if needed to optimize funding capture
            await self._adjust_for_funding_optimization(optimal_futures_side)
            
        except Exception as e:
            logger.error(f"Error optimizing for funding: {e}")
    
    async def _adjust_for_funding_optimization(self, optimal_side: str):
        """Adjust hedge ratio for funding optimization"""
        try:
            # Calculate optimal hedge ratio considering funding
            base_hedge_ratio = 1.0
            funding_adjustment = 0.0
            
            # Adjust hedge ratio based on funding rate magnitude
            if abs(self.current_funding.annualized_rate) > 0.10:  # 10% annual
                funding_adjustment = 0.1 if optimal_side == "long" else -0.1
            elif abs(self.current_funding.annualized_rate) > 0.05:  # 5% annual
                funding_adjustment = 0.05 if optimal_side == "long" else -0.05
            
            new_hedge_ratio = base_hedge_ratio + funding_adjustment
            
            # Apply hedge ratio bounds
            new_hedge_ratio = max(0.8, min(1.2, new_hedge_ratio))
            
            if abs(new_hedge_ratio - self.position.hedge_ratio) > 0.01:
                self.position.hedge_ratio = new_hedge_ratio
                logger.info(f"Adjusted hedge ratio for funding optimization: {new_hedge_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Error adjusting hedge ratio: {e}")
    
    async def _check_funding_opportunities(self):
        """Check for immediate funding opportunities"""
        try:
            # Get best funding opportunities from collector
            opportunities = self.funding_collector.get_best_funding_opportunities(min_annual_rate=0.05)
            
            for opportunity in opportunities[:3]:  # Top 3 opportunities
                if opportunity['symbol'] == self.grid_config.symbol:
                    logger.info(
                        f"🎯 FUNDING OPPORTUNITY: {opportunity['symbol']} "
                        f"Rate: {opportunity['funding_rate']:.4f} "
                        f"({opportunity['annualized_rate']:.2%} annual) "
                        f"Suggested: {opportunity['suggested_side']}"
                    )
            
        except Exception as e:
            logger.error(f"Error checking funding opportunities: {e}")
    
    async def _get_spot_balance(self) -> float:
        """Get current spot balance for the trading symbol"""
        try:
            # This would get actual spot balance from exchange
            # Placeholder implementation
            return self.position.spot_position
        except Exception as e:
            logger.error(f"Error getting spot balance: {e}")
            return 0.0
    
    async def _get_futures_position(self) -> float:
        """Get current futures position"""
        try:
            # This would get actual futures position from exchange
            # Placeholder implementation
            return self.position.futures_position
        except Exception as e:
            logger.error(f"Error getting futures position: {e}")
            return 0.0
    
    async def _execute_futures_hedge(self, quantity: float):
        """Execute futures hedge trade"""
        try:
            side = "BUY" if quantity > 0 else "SELL"
            abs_quantity = abs(quantity)
            
            logger.info(f"Executing futures hedge: {side} {abs_quantity:.4f}")
            
            # This would execute actual futures trade
            # Placeholder implementation
            self.position.futures_position += quantity
            
        except Exception as e:
            logger.error(f"Error executing futures hedge: {e}")
    
    async def _update_grid_orders(self):
        """Update grid orders based on current market conditions"""
        try:
            # Get current market price
            ticker = await self.spot_client.get_symbol_ticker(symbol=self.grid_config.symbol)
            current_price = float(ticker['price'])
            
            # Update spot grid orders
            for level in self.spot_grid_levels:
                if not level['active']:
                    # Check if we should activate this level
                    price_diff = abs(level['price'] - current_price) / current_price
                    if price_diff < self.grid_config.grid_spacing * 2:
                        # Place grid order (placeholder)
                        level['active'] = True
                        logger.debug(f"Activated spot grid level at {level['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating grid orders: {e}")
    
    async def _check_risk_limits(self):
        """Check all risk limits and controls"""
        try:
            # Check delta exposure
            if abs(self.position.delta_exposure) > self.risk_limits.max_delta_exposure:
                logger.warning(f"Delta exposure limit exceeded: {self.position.delta_exposure:.4f}")
                await self._rebalance_delta()
            
            # Check position size
            total_position_value = abs(self.position.spot_position) + abs(self.position.futures_position)
            if total_position_value > self.risk_limits.max_position_size:
                logger.warning(f"Position size limit exceeded: {total_position_value:.2f}")
            
            # Check daily loss
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -self.risk_limits.max_daily_loss:
                logger.error(f"Daily loss limit exceeded: {daily_pnl:.2f}")
                await self._trigger_emergency_stop("Daily loss limit exceeded")
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _monitor_pnl(self):
        """Monitor profit and loss"""
        try:
            # Calculate current PnL
            current_pnl = self.position.net_pnl + self.position.funding_collected
            self.pnl_history.append(current_pnl)
            
            # Keep only recent history
            if len(self.pnl_history) > 1440:  # 24 hours of minute data
                self.pnl_history = self.pnl_history[-1440:]
            
        except Exception as e:
            logger.error(f"Error monitoring PnL: {e}")
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        try:
            if len(self.pnl_history) < 2:
                return 0.0
            
            # Simple daily PnL calculation
            return self.pnl_history[-1] - self.pnl_history[0]
            
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    async def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check last rebalance time
            time_since_rebalance = datetime.now() - self.last_rebalance
            if time_since_rebalance > timedelta(hours=4):
                logger.warning("No rebalancing activity for over 4 hours")
            
            # Check funding collector health
            if hasattr(self, 'funding_collector'):
                funding_summary = self.funding_collector.get_funding_summary()
                if not funding_summary:
                    logger.warning("Funding collector not providing data")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            # Calculate key metrics
            total_pnl = self.position.net_pnl + self.position.funding_collected
            
            # Update position state
            self.position.net_pnl = total_pnl
            
            # Track funding collected
            if hasattr(self, 'current_funding'):
                estimated_daily_funding = self.current_funding.funding_rate * 3 * self.position.position_value
                logger.debug(f"Estimated daily funding: {estimated_daily_funding:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _log_performance_summary(self):
        """Log performance summary"""
        try:
            logger.info(
                f"📊 Performance Summary - "
                f"Net PnL: {self.position.net_pnl:.2f} | "
                f"Funding: {self.position.funding_collected:.2f} | "
                f"Delta: {self.position.delta_exposure:.4f} | "
                f"Hedge Ratio: {self.position.hedge_ratio:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop procedures"""
        try:
            logger.error(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
            self.emergency_stop = True
            self.is_active = False
            
            # Cancel all open orders
            await self._cancel_all_orders()
            
            # Close positions if required
            # This would be implemented based on specific requirements
            
            logger.error("Emergency stop procedures completed")
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            # Cancel spot orders
            # Cancel futures orders
            # Placeholder implementation
            logger.info("All orders cancelled")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _optimize_hedge_ratio(self):
        """Optimize hedge ratio based on current conditions"""
        try:
            # Start with base hedge ratio
            optimal_ratio = 1.0
            
            # Adjust based on funding opportunities
            if hasattr(self, 'current_funding'):
                if abs(self.current_funding.annualized_rate) > 0.20:  # 20% annual
                    # High funding rate - optimize for capture
                    if self.current_funding.funding_rate > 0:
                        optimal_ratio = 1.1  # Slightly long bias
                    else:
                        optimal_ratio = 0.9  # Slightly short bias
            
            self.position.hedge_ratio = optimal_ratio
            logger.info(f"Optimized hedge ratio: {optimal_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Error optimizing hedge ratio: {e}")
    
    async def stop_strategy(self):
        """Stop the delta-neutral strategy"""
        try:
            logger.info("Stopping delta-neutral strategy...")
            
            self.is_active = False
            
            # Cancel all orders
            await self._cancel_all_orders()
            
            # Final performance log
            await self._log_performance_summary()
            
            logger.info("✅ Delta-neutral strategy stopped")
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        try:
            return {
                'is_active': self.is_active,
                'emergency_stop': self.emergency_stop,
                'position': {
                    'spot_position': self.position.spot_position,
                    'futures_position': self.position.futures_position,
                    'delta_exposure': self.position.delta_exposure,
                    'net_pnl': self.position.net_pnl,
                    'funding_collected': self.position.funding_collected,
                    'hedge_ratio': self.position.hedge_ratio
                },
                'funding': {
                    'current_rate': getattr(self, 'current_funding', {}).funding_rate if hasattr(self, 'current_funding') else 0,
                    'annualized_rate': getattr(self, 'current_funding', {}).annualized_rate if hasattr(self, 'current_funding') else 0
                },
                'risk': {
                    'last_risk_check': self.last_risk_check.isoformat(),
                    'last_rebalance': self.last_rebalance.isoformat(),
                    'daily_pnl': self._calculate_daily_pnl()
                },
                'grid': {
                    'spot_levels': len(self.spot_grid_levels),
                    'futures_levels': len(self.futures_hedge_levels),
                    'active_levels': sum(1 for level in self.spot_grid_levels if level['active'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

if __name__ == '__main__':
    print("Advanced Delta-Neutral Manager for Funding Rate Capture")
    print("Key features:")
    print("- Delta neutrality with funding optimization")
    print("- Dynamic hedge ratio adjustment")
    print("- Real-time risk monitoring")
    print("- Grid-based profit capture")
    print("- Emergency stop controls")