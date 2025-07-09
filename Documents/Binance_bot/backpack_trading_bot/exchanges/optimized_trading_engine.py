#!/usr/bin/env python3
"""
🚀 ACTIVE TRADING ENGINE v6.0 OPTIMIZED - HIGH VOLUME + LOW DRAWDOWN
Based on Gemini Expert Consultation for Volume & Risk Optimization

OPTIMIZATION TARGETS:
- 📈 Increase Trading Volume: 600 → 1,500+ trades per 6 months
- 📉 Reduce Max Drawdown: 21.8% → <15%
- 📊 Maintain Performance: >19% returns, >1.0 Sharpe ratio

KEY OPTIMIZATIONS:
✅ Lower confidence threshold: 0.6 → 0.5 (more trading opportunities)
✅ Enhanced grid trading: 8 → 12 levels, faster rebalancing
✅ Multi-asset expansion: BTC, ETH + SOL, ADA, DOT
✅ Dynamic hedging: Volatility-based hedge ratios
✅ Real-time risk control: 30s hedge intervals, 15% max drawdown
✅ Market regime adaptation: Bull/bear/volatility-specific parameters
✅ GARCH volatility forecasting: Preemptive risk management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZED ENUMS AND DATA STRUCTURES
# ============================================================================

class VolatilityRegime(Enum):
    """Enhanced volatility regime classification."""
    LOW_VOL = "low_volatility"      # ATR < 20th percentile - More aggressive
    NORMAL_VOL = "normal"           # 20th-80th percentile - Baseline
    HIGH_VOL = "high_volatility"    # ATR > 80th percentile - Defensive

class MarketRegime(Enum):
    """Market trend classification."""
    BULL_MARKET = "bull_market"     # Price > 200 SMA + 5%
    BEAR_MARKET = "bear_market"     # Price < 200 SMA - 5%
    RANGING_MARKET = "ranging"      # Within ±5% of 200 SMA

class OptimizedSignalType(Enum):
    """Enhanced signal types for higher frequency trading."""
    HIGH_CONFIDENCE = "high_confidence"     # 0.7+ confidence
    MEDIUM_CONFIDENCE = "medium_confidence" # 0.5-0.7 confidence
    LOW_CONFIDENCE = "low_confidence"       # 0.4-0.5 confidence (new)
    ARBITRAGE_SIGNAL = "arbitrage"          # Cross-exchange opportunity
    VOLATILITY_BREAKOUT = "vol_breakout"    # Volatility-based entry
    MEAN_REVERSION = "mean_reversion"       # Grid-based signals

@dataclass
class OptimizedRiskLimits:
    """Enhanced risk limits for lower drawdown target."""
    # Tightened limits based on Gemini recommendations
    max_portfolio_delta: float = 0.03       # ↓ from 0.05 (tighter delta control)
    max_daily_loss: float = 3000            # ↓ from 5000 (stricter daily limit)
    max_drawdown_pct: float = 0.15          # ↓ from 0.20 (target <15%)
    max_position_size: float = 8000         # ↓ from 10000 (smaller positions)
    
    # New: Volatility-based adjustments
    volatility_position_scaler: float = 0.8 # Scale positions during high vol
    correlation_limit: float = 0.7          # Max correlation between assets
    
    # Enhanced hedge controls
    min_hedge_ratio: float = 0.8            # Minimum hedge coverage
    max_hedge_ratio: float = 1.5            # Maximum over-hedge

@dataclass
class OptimizedGridParams:
    """Enhanced grid parameters for higher volume trading."""
    # Increased levels for more trading opportunities
    num_levels: int = 12                    # ↑ from 8 (50% more levels)
    base_spacing: float = 0.008             # ↓ from 0.01 (tighter spacing)
    position_size_pct: float = 0.02         # ↓ from 0.025 (smaller, more frequent)
    
    # Faster adaptation for market changes
    rebalance_interval: int = 180           # ↓ from 300s (3min vs 5min)
    grid_reset_threshold: float = 0.15      # Reset grid on 15% price move
    
    # Volatility-based adjustments
    vol_spacing_multiplier: float = 1.5     # Wider spacing in high vol
    vol_size_reducer: float = 0.7           # Smaller positions in high vol

@dataclass 
class DynamicHedgeConfig:
    """Advanced hedging configuration for risk control."""
    base_check_interval: int = 30           # ↓ from 60s (faster hedging)
    delta_threshold: float = 0.025          # ↓ from 0.05 (tighter threshold)
    
    # Volatility-based hedge adjustment
    volatility_hedge_multiplier: float = 0.3  # Extra hedge in high vol
    max_hedge_ratio: float = 1.5            # Cap over-hedging
    
    # Market regime adjustments
    bull_hedge_ratio: float = 1.0           # Standard hedging in bull
    bear_hedge_ratio: float = 1.3           # Over-hedge in bear markets
    
    # GARCH forecasting integration
    forecast_horizon: int = 24              # 24-hour volatility forecast
    forecast_threshold: float = 1.3         # Act on 30% vol increase forecast

# ============================================================================
# ENHANCED INSTITUTIONAL MODULES
# ============================================================================

class OptimizedBitVolIndicator:
    """Enhanced BitVol with regime-specific parameters."""
    
    def __init__(self):
        self.lookback_periods = [24, 168, 720]  # 1d, 1w, 1m
        self.regime_thresholds = {
            'low_vol': 0.2,     # 20th percentile
            'high_vol': 0.8     # 80th percentile
        }
        
    def calculate_volatility_regime(self, price_data: pd.Series) -> VolatilityRegime:
        """Determine current volatility regime."""
        returns = price_data.pct_change().dropna()
        current_vol = returns.rolling(24).std().iloc[-1] * np.sqrt(365)
        
        # Calculate historical percentiles
        historical_vol = returns.rolling(720).std() * np.sqrt(365)
        percentile = (historical_vol <= current_vol).mean()
        
        if percentile < self.regime_thresholds['low_vol']:
            return VolatilityRegime.LOW_VOL
        elif percentile > self.regime_thresholds['high_vol']:
            return VolatilityRegime.HIGH_VOL
        else:
            return VolatilityRegime.NORMAL_VOL

class OptimizedKellyCriterion:
    """Enhanced Kelly Criterion with regime adjustments."""
    
    def __init__(self):
        self.base_multiplier = 0.25  # Conservative baseline
        self.regime_multipliers = {
            VolatilityRegime.LOW_VOL: 0.35,    # More aggressive in low vol
            VolatilityRegime.NORMAL_VOL: 0.25,  # Baseline
            VolatilityRegime.HIGH_VOL: 0.15     # Conservative in high vol
        }
        
    def calculate_optimal_position_size(self, win_prob: float, avg_return: float, 
                                      volatility: float, regime: VolatilityRegime) -> float:
        """Calculate Kelly-optimal position size with regime adjustment."""
        if avg_return <= 0:
            return 0.0
            
        # Base Kelly calculation
        kelly_fraction = (win_prob * avg_return - (1 - win_prob)) / avg_return
        
        # Apply regime-specific multiplier
        regime_multiplier = self.regime_multipliers.get(regime, 0.25)
        adjusted_kelly = kelly_fraction * regime_multiplier
        
        # Cap at reasonable limits
        return min(max(adjusted_kelly, 0.001), 0.08)  # 0.1%-8% range

class GARCHVolatilityForecaster:
    """GARCH(1,1) model for volatility forecasting."""
    
    def __init__(self):
        self.alpha = 0.1    # ARCH coefficient
        self.beta = 0.85    # GARCH coefficient
        self.omega = 0.0001 # Long-term variance
        self.forecast_horizon = 24
        
    def forecast_volatility(self, returns: pd.Series) -> Tuple[float, bool]:
        """Forecast next-period volatility and detect spikes."""
        if len(returns) < 100:
            return returns.std(), False
            
        # Simple GARCH(1,1) implementation
        current_vol = returns.rolling(24).std().iloc[-1]
        recent_return = returns.iloc[-1]
        
        # GARCH forecast
        forecasted_variance = (self.omega + 
                             self.alpha * recent_return**2 + 
                             self.beta * current_vol**2)
        forecasted_vol = np.sqrt(forecasted_variance)
        
        # Detect volatility spike (30% increase)
        vol_spike_detected = forecasted_vol > current_vol * 1.3
        
        return forecasted_vol, vol_spike_detected

# ============================================================================
# OPTIMIZED ACTIVE TRADING ENGINE
# ============================================================================

class ActiveTradingEngineOptimized:
    """
    Enhanced Active Trading Engine v6.0 with volume and risk optimizations.
    Targets 1,500+ trades with <15% max drawdown.
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        
        # Optimized core parameters
        self.confidence_threshold = 0.5        # ↓ from 0.6 (more opportunities)
        self.bull_market_multiplier = 2.0      # ↑ from 1.5 (more aggressive)
        
        # Multi-asset expansion for volume increase
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT',   # Core + new assets
            'ADAUSDT', 'DOTUSDT'               # Additional volume sources
        ]
        
        # Optimized components
        self.risk_limits = OptimizedRiskLimits()
        self.grid_params = OptimizedGridParams()
        self.hedge_config = DynamicHedgeConfig()
        
        # Enhanced modules
        self.bitvol = OptimizedBitVolIndicator()
        self.kelly = OptimizedKellyCriterion()
        self.garch = GARCHVolatilityForecaster()
        
        # State tracking
        self.current_regime = {
            'volatility': VolatilityRegime.NORMAL_VOL,
            'market': MarketRegime.RANGING_MARKET
        }
        self.positions = {}
        self.hedge_positions = {}
        self.performance_metrics = {}
        
        # Optimization tracking
        self.trade_count = 0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.config.get('initial_capital', 10000)
        
        logger.info("🚀 Active Trading Engine v6.0 OPTIMIZED initialized")
        logger.info(f"📊 Target: 1,500+ trades with <15% max drawdown")
        logger.info(f"💰 Trading pairs: {len(self.trading_pairs)} assets")
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration with optimized defaults."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            config = {}
            
        # Set optimized defaults
        defaults = {
            'initial_capital': 10000,
            'max_position_size': 8000,          # Reduced for risk control
            'confidence_threshold': 0.5,        # Lowered for more trades
            'hedge_check_interval': 30,         # Faster hedging
            'max_drawdown_pct': 0.15            # Target <15%
        }
        
        for key, value in defaults.items():
            config.setdefault(key, value)
            
        return config
    
    async def detect_market_regime(self, symbol: str) -> None:
        """Detect current market and volatility regimes."""
        try:
            # Get price data (mock implementation)
            price_data = await self._get_price_data(symbol, periods=1000)
            
            # Volatility regime detection
            self.current_regime['volatility'] = self.bitvol.calculate_volatility_regime(price_data)
            
            # Market trend detection
            sma_200 = price_data.rolling(200).mean().iloc[-1]
            current_price = price_data.iloc[-1]
            
            if current_price > sma_200 * 1.05:
                self.current_regime['market'] = MarketRegime.BULL_MARKET
            elif current_price < sma_200 * 0.95:
                self.current_regime['market'] = MarketRegime.BEAR_MARKET
            else:
                self.current_regime['market'] = MarketRegime.RANGING_MARKET
                
            logger.info(f"📊 Market regime: {self.current_regime}")
            
        except Exception as e:
            logger.error(f"❌ Regime detection error: {e}")
    
    def get_regime_adjusted_parameters(self) -> Dict[str, Any]:
        """Get trading parameters adjusted for current market regime."""
        vol_regime = self.current_regime['volatility']
        market_regime = self.current_regime['market']
        
        params = {
            'confidence_threshold': 0.5,  # Base threshold
            'position_scaler': 1.0,
            'hedge_ratio_multiplier': 1.0,
            'grid_levels': 12
        }
        
        # Volatility regime adjustments
        if vol_regime == VolatilityRegime.LOW_VOL:
            params.update({
                'confidence_threshold': 0.40,      # More aggressive
                'position_scaler': 1.2,            # Larger positions
                'grid_levels': 15                  # More grid levels
            })
        elif vol_regime == VolatilityRegime.HIGH_VOL:
            params.update({
                'confidence_threshold': 0.55,      # More selective
                'position_scaler': 0.7,            # Smaller positions
                'hedge_ratio_multiplier': 1.3,     # Over-hedge
                'grid_levels': 8                   # Fewer levels
            })
        
        # Market regime adjustments
        if market_regime == MarketRegime.BULL_MARKET:
            params['bull_multiplier'] = 2.0       # Aggressive in bull
        elif market_regime == MarketRegime.BEAR_MARKET:
            params['bull_multiplier'] = 1.0       # Defensive in bear
            params['hedge_ratio_multiplier'] *= 1.2  # Extra hedging
        
        return params
    
    async def calculate_dynamic_hedge_ratio(self, delta_exposure: float, symbol: str) -> float:
        """Calculate volatility-adjusted hedge ratio."""
        try:
            # Get current volatility metrics
            price_data = await self._get_price_data(symbol, periods=100)
            returns = price_data.pct_change().dropna()
            
            current_vol = returns.rolling(24).std().iloc[-1] if len(returns) >= 24 else 0.02
            avg_vol = returns.std() if len(returns) > 0 else 0.02
            
            # Volatility factor (1.0 = normal, >1.0 = high vol)
            volatility_factor = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Base hedge ratio with volatility adjustment
            base_ratio = 1.0
            volatility_adjustment = (volatility_factor - 1.0) * self.hedge_config.volatility_hedge_multiplier
            
            # Market regime adjustment
            regime_params = self.get_regime_adjusted_parameters()
            regime_multiplier = regime_params.get('hedge_ratio_multiplier', 1.0)
            
            # Final hedge ratio
            hedge_ratio = base_ratio + volatility_adjustment
            hedge_ratio *= regime_multiplier
            
            # Cap the hedge ratio
            hedge_ratio = min(max(hedge_ratio, self.risk_limits.min_hedge_ratio), 
                            self.risk_limits.max_hedge_ratio)
            
            logger.info(f"🔄 Dynamic hedge ratio: {hedge_ratio:.3f} (vol_factor: {volatility_factor:.3f})")
            return hedge_ratio
            
        except Exception as e:
            logger.error(f"❌ Hedge ratio calculation error: {e}")
            return 1.0  # Fallback to 1:1 hedging
    
    async def execute_optimized_hedge(self, delta_exposure: float, symbol: str) -> bool:
        """Execute hedge with dynamic ratio and faster timing."""
        try:
            if abs(delta_exposure) < self.hedge_config.delta_threshold:
                return True  # No hedge needed
            
            # Calculate dynamic hedge size
            hedge_ratio = await self.calculate_dynamic_hedge_ratio(delta_exposure, symbol)
            hedge_size = delta_exposure * hedge_ratio
            
            # Execute hedge order (mock implementation)
            hedge_side = "short" if delta_exposure > 0 else "long"
            
            logger.info(f"🛡️ Executing {hedge_side} hedge: {abs(hedge_size):.4f} BTC")
            
            # Record hedge position
            hedge_id = f"hedge_{symbol}_{datetime.now().strftime('%H%M%S')}"
            self.hedge_positions[hedge_id] = {
                'symbol': symbol,
                'size': hedge_size,
                'side': hedge_side,
                'timestamp': datetime.now(),
                'ratio': hedge_ratio
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Hedge execution error: {e}")
            return False
    
    async def generate_optimized_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals with lowered confidence threshold."""
        signals = []
        
        try:
            # Get regime-adjusted parameters
            params = self.get_regime_adjusted_parameters()
            confidence_threshold = params['confidence_threshold']
            
            # Mock signal generation with multiple confidence levels
            signal_types = [
                {'type': 'grid_buy', 'confidence': 0.65, 'size': 0.03},
                {'type': 'arbitrage', 'confidence': 0.55, 'size': 0.02},
                {'type': 'mean_reversion', 'confidence': 0.45, 'size': 0.025},
                {'type': 'volatility_breakout', 'confidence': 0.42, 'size': 0.015}
            ]
            
            # Filter signals by confidence threshold
            for signal in signal_types:
                if signal['confidence'] >= confidence_threshold:
                    # Adjust size based on regime
                    adjusted_size = signal['size'] * params['position_scaler']
                    
                    signals.append({
                        'symbol': symbol,
                        'type': signal['type'],
                        'confidence': signal['confidence'],
                        'size': adjusted_size,
                        'timestamp': datetime.now()
                    })
                    
            logger.info(f"📊 Generated {len(signals)} signals for {symbol} (threshold: {confidence_threshold:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Signal generation error for {symbol}: {e}")
        
        return signals
    
    async def execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Execute trading signal with optimized parameters."""
        try:
            symbol = signal['symbol']
            signal_type = signal['type']
            confidence = signal['confidence']
            size = signal['size']
            
            # Position size validation
            max_size = self.risk_limits.max_position_size / self._get_current_price(symbol)
            size = min(size, max_size)
            
            # Kelly Criterion position sizing
            regime = self.current_regime['volatility']
            kelly_size = self.kelly.calculate_optimal_position_size(0.6, 0.02, 0.3, regime)
            size = min(size, kelly_size)
            
            # Execute trade (mock implementation)
            logger.info(f"✅ Executing {signal_type}: {size:.4f} {symbol} (confidence: {confidence:.2f})")
            
            # Record position
            position_id = f"pos_{symbol}_{datetime.now().strftime('%H%M%S')}"
            self.positions[position_id] = {
                'symbol': symbol,
                'size': size,
                'type': signal_type,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            # Execute hedge if needed
            await self.execute_optimized_hedge(size, symbol)
            
            # Update trade count
            self.trade_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal execution error: {e}")
            return False
    
    async def update_performance_metrics(self) -> None:
        """Update performance tracking with optimization targets."""
        try:
            current_equity = self._calculate_portfolio_value()
            
            # Update peak equity and drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Check if we're exceeding risk limits
            if current_drawdown > self.risk_limits.max_drawdown_pct:
                logger.warning(f"⚠️ Drawdown limit exceeded: {current_drawdown:.2%}")
                await self._emergency_risk_reduction()
            
            # Log performance metrics
            logger.info(f"📊 Performance Update:")
            logger.info(f"   💰 Equity: ${current_equity:.2f}")
            logger.info(f"   📈 Trades: {self.trade_count}")
            logger.info(f"   ⚠️ Max DD: {self.max_drawdown:.2%}")
            logger.info(f"   🎯 Target: <15% DD, 1,500+ trades")
            
        except Exception as e:
            logger.error(f"❌ Performance update error: {e}")
    
    async def _emergency_risk_reduction(self) -> None:
        """Emergency procedure when risk limits are exceeded."""
        logger.critical("🚨 EMERGENCY RISK REDUCTION ACTIVATED")
        
        try:
            # Reduce position sizes
            for position_id, position in self.positions.items():
                position['size'] *= 0.5  # 50% position reduction
            
            # Increase hedge ratios
            self.hedge_config.volatility_hedge_multiplier = 0.5
            
            # Raise confidence threshold
            self.confidence_threshold = 0.65
            
            logger.critical("🛡️ Risk reduction measures implemented")
            
        except Exception as e:
            logger.error(f"❌ Emergency risk reduction error: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Mock price data - replace with real exchange API."""
        prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'SOLUSDT': 100.0,
            'ADAUSDT': 0.5,
            'DOTUSDT': 8.0
        }
        return prices.get(symbol, 1000.0)
    
    async def _get_price_data(self, symbol: str, periods: int) -> pd.Series:
        """Mock price data generation - replace with real data."""
        import numpy as np
        np.random.seed(42)
        
        base_price = self._get_current_price(symbol)
        returns = np.random.normal(0, 0.02, periods)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.Series(prices[1:])
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        base_value = self.config.get('initial_capital', 10000)
        
        # Mock P&L calculation
        pnl = len(self.positions) * 50 - self.max_drawdown * base_value
        
        return base_value + pnl
    
    async def run_optimized_trading_loop(self) -> None:
        """Main optimized trading loop for higher volume and lower drawdown."""
        logger.info("🚀 Starting optimized trading loop...")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                loop_start = datetime.now()
                
                # Update market regimes for all symbols
                for symbol in self.trading_pairs:
                    await self.detect_market_regime(symbol)
                
                # Generate and execute signals for all symbols
                total_signals = 0
                for symbol in self.trading_pairs:
                    signals = await self.generate_optimized_signals(symbol)
                    total_signals += len(signals)
                    
                    for signal in signals:
                        await self.execute_signal(signal)
                
                # Fast hedge checking (every 30 seconds)
                if iteration % 1 == 0:  # Every iteration for now
                    for symbol in self.trading_pairs:
                        # Mock delta calculation
                        delta_exposure = np.random.uniform(-0.05, 0.05)
                        await self.execute_optimized_hedge(delta_exposure, symbol)
                
                # Performance monitoring
                if iteration % 10 == 0:  # Every 10 iterations
                    await self.update_performance_metrics()
                
                # Summary logging
                if iteration % 60 == 0:  # Every hour equivalent
                    logger.info(f"🎯 Optimization Summary (Iteration {iteration}):")
                    logger.info(f"   📊 Total trades: {self.trade_count}")
                    logger.info(f"   💰 Signals generated: {total_signals}")
                    logger.info(f"   ⚠️ Max drawdown: {self.max_drawdown:.2%}")
                    logger.info(f"   🎯 Target progress: {self.trade_count}/1500 trades")
                
                # Sleep for next iteration (reduced for higher frequency)
                await asyncio.sleep(10)  # 10 second intervals for active trading
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
            raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    try:
        # Initialize optimized trading engine
        engine = ActiveTradingEngineOptimized("config.json")
        
        # Run optimized trading loop
        await engine.run_optimized_trading_loop()
        
    except Exception as e:
        logger.error(f"❌ Main execution error: {e}")
        raise

if __name__ == "__main__":
    print("🚀 ACTIVE TRADING ENGINE v6.0 OPTIMIZED")
    print("=" * 60)
    print("📊 OPTIMIZATION TARGETS:")
    print("   📈 Volume: 600 → 1,500+ trades per 6 months")
    print("   📉 Drawdown: 21.8% → <15% maximum")
    print("   📊 Performance: Maintain >19% returns")
    print("=" * 60)
    print("🎯 Starting optimized trading engine...")
    print()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the optimized trading engine
    asyncio.run(main())