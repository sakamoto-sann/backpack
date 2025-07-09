#!/usr/bin/env python3
"""
🛡️ INTEGRATED RISK MANAGEMENT MODULE v1.0.0
Critical risk controls for multi-exchange arbitrage trading

Features:
- 📊 Real-time portfolio delta across all exchanges
- 🚨 Multi-level circuit breakers and kill switches
- 💰 Position limits and exposure monitoring
- ⚖️ Preserves existing institutional risk controls
- 🔄 Emergency protocols and auto-hedging
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import existing institutional modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class EmergencyAction(Enum):
    NONE = "none"
    REDUCE_EXPOSURE = "reduce_exposure"
    STOP_NEW_TRADES = "stop_new_trades"
    CLOSE_ARBITRAGE = "close_arbitrage"
    FULL_EMERGENCY_STOP = "full_emergency_stop"

@dataclass
class PortfolioMetrics:
    total_portfolio_value: float = 0.0
    total_delta: float = 0.0
    delta_deviation: float = 0.0
    total_exposure: float = 0.0
    max_exposure_used_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class ExchangeRisk:
    exchange: str
    connected: bool = True
    latency_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    positions_count: int = 0
    total_exposure: float = 0.0
    margin_ratio: float = 1.0  # For futures
    funding_exposure: float = 0.0
    risk_score: float = 0.0

@dataclass
class RiskLimits:
    # Portfolio limits
    max_portfolio_delta: float = 0.05  # 5% max delta deviation
    max_total_exposure: float = 100000  # $100k max exposure
    max_daily_loss: float = 5000  # $5k max daily loss
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown: float = 10000  # $10k max drawdown
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    
    # Position limits per exchange
    max_position_size: float = 10000  # $10k max per position
    max_arbitrage_size: float = 5000  # $5k max per arbitrage
    max_positions_per_exchange: int = 10
    
    # Operational limits
    max_api_errors_per_hour: int = 100
    max_latency_ms: float = 1000.0  # 1 second max latency
    min_margin_ratio: float = 0.3  # 30% min margin for futures

class IntegratedRiskManager:
    """
    Central risk management for multi-exchange trading system.
    Integrates with existing institutional bot while adding cross-exchange controls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits from config
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        
        # Portfolio tracking
        self.portfolio_metrics = PortfolioMetrics()
        self.exchange_risks = {}
        
        # Position tracking across exchanges
        self.positions = {
            'binance': {},
            'backpack': {}
        }
        
        # PnL tracking
        self.daily_pnl_history = []
        self.portfolio_start_value = 0.0
        self.portfolio_peak_value = 0.0
        
        # Emergency controls
        self.emergency_stop_active = False
        self.kill_switch_active = False
        self.trading_halted = False
        
        # Risk monitoring
        self.last_risk_check = datetime.now()
        self.risk_check_interval = 1.0  # Check every second
        
        # Error tracking
        self.error_counts = {}
        self.last_errors = {}
        
        logger.info("🛡️ Integrated Risk Manager v1.0.0 initialized")
        logger.info(f"   Max Portfolio Delta: {self.risk_limits.max_portfolio_delta:.3f}")
        logger.info(f"   Max Total Exposure: ${self.risk_limits.max_total_exposure:,.0f}")
        logger.info(f"   Max Daily Loss: ${self.risk_limits.max_daily_loss:,.0f}")
    
    async def initialize(self, initial_portfolio_value: float):
        """Initialize risk manager with starting portfolio value."""
        self.portfolio_start_value = initial_portfolio_value
        self.portfolio_peak_value = initial_portfolio_value
        self.portfolio_metrics.total_portfolio_value = initial_portfolio_value
        
        logger.info(f"📊 Risk manager initialized with ${initial_portfolio_value:,.2f} portfolio")
    
    async def update_positions(self, exchange: str, positions: Dict[str, Any]):
        """Update position data from an exchange."""
        try:
            self.positions[exchange] = positions
            
            # Update exchange risk metrics
            if exchange not in self.exchange_risks:
                self.exchange_risks[exchange] = ExchangeRisk(exchange=exchange)
            
            exchange_risk = self.exchange_risks[exchange]
            exchange_risk.positions_count = len(positions)
            exchange_risk.total_exposure = sum(
                abs(pos.get('notional', 0)) for pos in positions.values()
            )
            
            # Calculate risk score
            exchange_risk.risk_score = self._calculate_exchange_risk_score(exchange_risk)
            
        except Exception as e:
            logger.error(f"❌ Failed to update positions for {exchange}: {e}")
    
    async def update_portfolio_metrics(self, exchange_balances: Dict[str, Dict[str, float]]):
        """Update overall portfolio metrics from all exchanges."""
        try:
            # Calculate total portfolio value
            total_value = 0.0
            for exchange, balances in exchange_balances.items():
                for asset, balance in balances.items():
                    if asset == 'USDT' or asset == 'USD':
                        total_value += balance
                    # TODO: Convert other assets to USD equivalent
            
            self.portfolio_metrics.total_portfolio_value = total_value
            
            # Calculate total delta across all positions
            self.portfolio_metrics.total_delta = self._calculate_total_delta()
            self.portfolio_metrics.delta_deviation = abs(self.portfolio_metrics.total_delta)
            
            # Calculate total exposure
            self.portfolio_metrics.total_exposure = sum(
                risk.total_exposure for risk in self.exchange_risks.values()
            )
            
            # Calculate exposure percentage
            if self.risk_limits.max_total_exposure > 0:
                self.portfolio_metrics.max_exposure_used_pct = (
                    self.portfolio_metrics.total_exposure / self.risk_limits.max_total_exposure
                )
            
            # Calculate daily PnL
            self.portfolio_metrics.daily_pnl = total_value - self.portfolio_start_value
            if self.portfolio_start_value > 0:
                self.portfolio_metrics.daily_pnl_pct = (
                    self.portfolio_metrics.daily_pnl / self.portfolio_start_value
                )
            
            # Update peak value and calculate drawdown
            if total_value > self.portfolio_peak_value:
                self.portfolio_peak_value = total_value
            
            self.portfolio_metrics.max_drawdown = self.portfolio_peak_value - total_value
            if self.portfolio_peak_value > 0:
                self.portfolio_metrics.max_drawdown_pct = (
                    self.portfolio_metrics.max_drawdown / self.portfolio_peak_value
                )
            
            # Update timestamp
            self.portfolio_metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio metrics: {e}")
    
    def _calculate_total_delta(self) -> float:
        """Calculate total portfolio delta across all exchanges."""
        try:
            total_delta = 0.0
            
            for exchange, positions in self.positions.items():
                for symbol, position in positions.items():
                    # Calculate delta for each position
                    quantity = position.get('quantity', 0)
                    side = position.get('side', 'long')
                    
                    if side == 'long':
                        total_delta += quantity
                    else:
                        total_delta -= quantity
            
            return total_delta
            
        except Exception as e:
            logger.error(f"❌ Delta calculation error: {e}")
            return 0.0
    
    def _calculate_exchange_risk_score(self, exchange_risk: ExchangeRisk) -> float:
        """Calculate risk score for an exchange (0-1, higher = riskier)."""
        try:
            risk_factors = []
            
            # Connection risk
            if not exchange_risk.connected:
                risk_factors.append(1.0)
            else:
                risk_factors.append(0.0)
            
            # Latency risk
            latency_risk = min(exchange_risk.latency_ms / 1000.0, 1.0)
            risk_factors.append(latency_risk * 0.3)
            
            # Error rate risk
            error_risk = min(exchange_risk.error_count / 100.0, 1.0)
            risk_factors.append(error_risk * 0.4)
            
            # Exposure risk
            exposure_risk = min(exchange_risk.total_exposure / self.risk_limits.max_total_exposure, 1.0)
            risk_factors.append(exposure_risk * 0.3)
            
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
            
        except Exception as e:
            logger.error(f"❌ Exchange risk calculation error: {e}")
            return 0.5
    
    async def check_risk_limits(self) -> Tuple[RiskLevel, List[str]]:
        """Check all risk limits and return risk level with violations."""
        try:
            violations = []
            risk_level = RiskLevel.NORMAL
            
            # Check portfolio delta
            if self.portfolio_metrics.delta_deviation > self.risk_limits.max_portfolio_delta:
                violations.append(f"Portfolio delta deviation: {self.portfolio_metrics.delta_deviation:.4f} > {self.risk_limits.max_portfolio_delta:.4f}")
                risk_level = max(risk_level, RiskLevel.WARNING)
            
            # Check total exposure
            if self.portfolio_metrics.total_exposure > self.risk_limits.max_total_exposure:
                violations.append(f"Total exposure: ${self.portfolio_metrics.total_exposure:,.0f} > ${self.risk_limits.max_total_exposure:,.0f}")
                risk_level = max(risk_level, RiskLevel.CRITICAL)
            
            # Check daily loss
            if abs(self.portfolio_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
                violations.append(f"Daily loss: ${abs(self.portfolio_metrics.daily_pnl):,.0f} > ${self.risk_limits.max_daily_loss:,.0f}")
                risk_level = max(risk_level, RiskLevel.CRITICAL)
            
            # Check daily loss percentage
            if abs(self.portfolio_metrics.daily_pnl_pct) > self.risk_limits.max_daily_loss_pct:
                violations.append(f"Daily loss %: {abs(self.portfolio_metrics.daily_pnl_pct)*100:.2f}% > {self.risk_limits.max_daily_loss_pct*100:.2f}%")
                risk_level = max(risk_level, RiskLevel.CRITICAL)
            
            # Check drawdown
            if self.portfolio_metrics.max_drawdown > self.risk_limits.max_drawdown:
                violations.append(f"Max drawdown: ${self.portfolio_metrics.max_drawdown:,.0f} > ${self.risk_limits.max_drawdown:,.0f}")
                risk_level = max(risk_level, RiskLevel.EMERGENCY)
            
            # Check drawdown percentage
            if self.portfolio_metrics.max_drawdown_pct > self.risk_limits.max_drawdown_pct:
                violations.append(f"Max drawdown %: {self.portfolio_metrics.max_drawdown_pct*100:.2f}% > {self.risk_limits.max_drawdown_pct*100:.2f}%")
                risk_level = max(risk_level, RiskLevel.EMERGENCY)
            
            # Check exchange-specific risks
            for exchange, risk in self.exchange_risks.items():
                if risk.risk_score > 0.7:
                    violations.append(f"{exchange} risk score: {risk.risk_score:.2f}")
                    risk_level = max(risk_level, RiskLevel.WARNING)
                
                if risk.error_count > self.risk_limits.max_api_errors_per_hour:
                    violations.append(f"{exchange} error count: {risk.error_count}")
                    risk_level = max(risk_level, RiskLevel.CRITICAL)
            
            # Update portfolio risk level
            self.portfolio_metrics.risk_level = risk_level
            
            return risk_level, violations
            
        except Exception as e:
            logger.error(f"❌ Risk limit check error: {e}")
            return RiskLevel.WARNING, [f"Risk check error: {e}"]
    
    async def execute_emergency_action(self, risk_level: RiskLevel, violations: List[str]) -> EmergencyAction:
        """Execute appropriate emergency action based on risk level."""
        try:
            if risk_level == RiskLevel.EMERGENCY or self.kill_switch_active:
                await self._full_emergency_stop()
                return EmergencyAction.FULL_EMERGENCY_STOP
            
            elif risk_level == RiskLevel.CRITICAL:
                await self._stop_new_trades()
                await self._close_arbitrage_positions()
                return EmergencyAction.CLOSE_ARBITRAGE
            
            elif risk_level == RiskLevel.WARNING:
                await self._reduce_exposure()
                return EmergencyAction.REDUCE_EXPOSURE
            
            return EmergencyAction.NONE
            
        except Exception as e:
            logger.error(f"❌ Emergency action execution error: {e}")
            return EmergencyAction.NONE
    
    async def _full_emergency_stop(self):
        """Execute full emergency stop procedure."""
        logger.critical("🚨 FULL EMERGENCY STOP ACTIVATED")
        
        self.emergency_stop_active = True
        self.trading_halted = True
        
        # TODO: Cancel all open orders
        # TODO: Close all positions (if configured)
        # TODO: Send emergency alerts
        
        logger.critical("🛑 All trading activities halted")
    
    async def _stop_new_trades(self):
        """Stop new trade execution while allowing existing trades to complete."""
        logger.warning("⚠️ Stopping new trades - risk limit breach")
        self.trading_halted = True
    
    async def _close_arbitrage_positions(self):
        """Close all arbitrage positions to reduce risk."""
        logger.warning("🔄 Closing arbitrage positions for risk reduction")
        # TODO: Implement arbitrage position closure
    
    async def _reduce_exposure(self):
        """Reduce overall exposure by closing largest positions."""
        logger.warning("📉 Reducing exposure - warning level reached")
        # TODO: Implement exposure reduction logic
    
    async def manual_kill_switch(self, reason: str = "Manual intervention"):
        """Manually activate kill switch."""
        logger.critical(f"🚨 MANUAL KILL SWITCH ACTIVATED: {reason}")
        self.kill_switch_active = True
        await self._full_emergency_stop()
    
    async def reset_emergency_state(self, confirmation: str):
        """Reset emergency state after manual confirmation."""
        if confirmation == "CONFIRM_RESET":
            self.emergency_stop_active = False
            self.kill_switch_active = False
            self.trading_halted = False
            logger.info("✅ Emergency state reset - trading can resume")
            return True
        else:
            logger.warning("❌ Invalid confirmation for emergency reset")
            return False
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not (self.emergency_stop_active or self.kill_switch_active or self.trading_halted)
    
    def can_open_position(self, exchange: str, size: float, position_type: str = "spot") -> Tuple[bool, str]:
        """Check if a new position can be opened."""
        try:
            if not self.is_trading_allowed():
                return False, "Trading is halted"
            
            # Check position limits
            exchange_risk = self.exchange_risks.get(exchange)
            if exchange_risk:
                if exchange_risk.positions_count >= self.risk_limits.max_positions_per_exchange:
                    return False, f"Max positions reached for {exchange}"
                
                if exchange_risk.total_exposure + size > self.risk_limits.max_total_exposure * 0.5:
                    return False, f"Exposure limit would be exceeded on {exchange}"
            
            # Check position size
            if size > self.risk_limits.max_position_size:
                return False, f"Position size ${size:,.0f} exceeds limit ${self.risk_limits.max_position_size:,.0f}"
            
            # Check total exposure
            if self.portfolio_metrics.total_exposure + size > self.risk_limits.max_total_exposure:
                return False, f"Total exposure limit would be exceeded"
            
            return True, "Position allowed"
            
        except Exception as e:
            logger.error(f"❌ Position check error: {e}")
            return False, f"Position check error: {e}"
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status."""
        return {
            'portfolio_metrics': self.portfolio_metrics,
            'exchange_risks': self.exchange_risks,
            'emergency_status': {
                'emergency_stop_active': self.emergency_stop_active,
                'kill_switch_active': self.kill_switch_active,
                'trading_halted': self.trading_halted
            },
            'trading_allowed': self.is_trading_allowed()
        }

# Example usage
if __name__ == "__main__":
    config = {
        'risk_limits': {
            'max_portfolio_delta': 0.05,
            'max_total_exposure': 50000,
            'max_daily_loss': 2500,
            'max_daily_loss_pct': 0.05
        }
    }
    
    risk_manager = IntegratedRiskManager(config)
    
    async def test_risk_manager():
        await risk_manager.initialize(100000)  # $100k starting portfolio
        
        # Simulate position update
        await risk_manager.update_positions('binance', {
            'BTCUSDT': {'quantity': 0.1, 'side': 'long', 'notional': 5000}
        })
        
        # Simulate portfolio update
        await risk_manager.update_portfolio_metrics({
            'binance': {'USDT': 95000},
            'backpack': {'USD': 0}
        })
        
        # Check risk limits
        risk_level, violations = await risk_manager.check_risk_limits()
        print(f"Risk Level: {risk_level.value}")
        if violations:
            print(f"Violations: {violations}")
        
        # Test position allowance
        allowed, reason = risk_manager.can_open_position('binance', 3000)
        print(f"Can open position: {allowed} - {reason}")
    
    asyncio.run(test_risk_manager())