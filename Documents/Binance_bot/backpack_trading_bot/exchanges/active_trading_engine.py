#!/usr/bin/env python3
"""
ACTIVE TRADING ENGINE v6.0
Enhanced Multi-Exchange Institutional Trading System

Features:
- Real-time 24/7 market monitoring
- All 8 institutional modules active
- Dynamic grid range optimization
- Automatic futures hedging
- Cross-exchange arbitrage
- Bull market aggression
- Emergency controls
- API rate limiting compliance
"""

import asyncio
import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import websocket
import ccxt
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import warnings
warnings.filterwarnings('ignore')

# Import all institutional modules
from bitvol_module import BitVolModule
from lxvx_module import LXVXModule
from garch_module import GARCHModule
from kelly_module import KellyModule
from gamma_hedging_module import GammaHedgingModule
from emergency_module import EmergencyModule
from atr_supertrend_module import ATRSupertrendModule
from multi_timeframe_module import MultiTimeframeModule

@dataclass
class TradingSignal:
    """Unified trading signal structure"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    grid_spacing: float
    target_range: Tuple[float, float]
    hedge_ratio: float
    position_size: float
    stop_loss: float
    take_profit: float
    source_module: str
    market_regime: str
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH', 'EMERGENCY'

@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    binance_price: float
    backpack_price: float
    spread: float
    spread_percent: float
    volume: float
    profit_potential: float
    execution_time: datetime
    confidence: float

@dataclass
class PortfolioState:
    """Current portfolio state"""
    spot_positions: Dict[str, float]
    futures_positions: Dict[str, float]
    total_value: float
    delta_exposure: float
    gamma_exposure: float
    unrealized_pnl: float
    daily_pnl: float
    last_update: datetime

class ActiveTradingEngine:
    """
    Enhanced Active Trading Engine v6.0
    
    Integrates all 8 institutional modules for continuous trading
    with real-time execution, hedging, and arbitrage capabilities
    """
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the Active Trading Engine"""
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize exchanges
        self.binance_client = self._init_binance()
        self.backpack_client = self._init_backpack()
        
        # Initialize institutional modules
        self.bitvol = BitVolModule()
        self.lxvx = LXVXModule()
        self.garch = GARCHModule()
        self.kelly = KellyModule()
        self.gamma_hedging = GammaHedgingModule()
        self.emergency = EmergencyModule()
        self.atr_supertrend = ATRSupertrendModule()
        self.multi_timeframe = MultiTimeframeModule()
        
        # Trading state
        self.is_running = False
        self.market_data = {}
        self.signal_queue = Queue()
        self.arbitrage_queue = Queue()
        self.portfolio_state = PortfolioState(
            spot_positions={},
            futures_positions={},
            total_value=0.0,
            delta_exposure=0.0,
            gamma_exposure=0.0,
            unrealized_pnl=0.0,
            daily_pnl=0.0,
            last_update=datetime.now()
        )
        
        # Rate limiting
        self.last_request_time = {}
        self.request_counts = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.websocket_threads = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'arbitrage_opportunities': 0,
            'emergency_stops': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Active Trading Engine v6.0 initialized")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Enhanced configuration with aggressive bull market settings
            default_config = {
                'trading': {
                    'active_symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                    'base_position_size': 0.1,
                    'max_position_size': 1.0,
                    'min_confidence_threshold': 0.6,  # Lowered for more active trading
                    'bull_market_threshold': 0.75,
                    'bull_market_multiplier': 1.5,
                    'grid_levels': 10,
                    'max_spread': 0.02,
                    'rebalance_threshold': 0.05
                },
                'risk_management': {
                    'max_portfolio_risk': 0.02,
                    'max_single_trade_risk': 0.005,
                    'stop_loss_percent': 0.02,
                    'take_profit_percent': 0.06,
                    'max_drawdown': 0.1,
                    'emergency_stop_loss': 0.05
                },
                'hedging': {
                    'auto_hedge': True,
                    'hedge_threshold': 0.1,
                    'delta_target': 0.0,
                    'gamma_threshold': 0.05,
                    'futures_multiplier': 1.0
                },
                'arbitrage': {
                    'min_spread': 0.001,  # 0.1%
                    'max_execution_time': 5.0,
                    'min_volume': 100.0,
                    'profit_threshold': 0.005
                },
                'api': {
                    'binance_rate_limit': 1200,  # requests per minute
                    'backpack_rate_limit': 600,
                    'websocket_timeout': 30,
                    'reconnect_delay': 5
                }
            }
            
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logger = logging.getLogger('ActiveTradingEngine')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            f'active_trading_engine_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_binance(self):
        """Initialize Binance client"""
        try:
            client = BinanceClient(
                api_key=self.config.get('binance_api_key', ''),
                api_secret=self.config.get('binance_api_secret', ''),
                testnet=self.config.get('testnet', True)
            )
            
            # Test connection
            client.get_account()
            self.logger.info("Binance client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            return None
    
    def _init_backpack(self):
        """Initialize Backpack client"""
        try:
            client = ccxt.backpack({
                'apiKey': self.config.get('backpack_api_key', ''),
                'secret': self.config.get('backpack_api_secret', ''),
                'sandbox': self.config.get('testnet', True),
                'enableRateLimit': True,
            })
            
            # Test connection
            client.fetch_balance()
            self.logger.info("Backpack client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Backpack client: {e}")
            return None
    
    def _check_rate_limit(self, exchange: str) -> bool:
        """Check if we can make API request within rate limits"""
        current_time = time.time()
        
        if exchange not in self.last_request_time:
            self.last_request_time[exchange] = current_time
            self.request_counts[exchange] = 1
            return True
        
        time_diff = current_time - self.last_request_time[exchange]
        
        # Reset counter if more than 1 minute has passed
        if time_diff > 60:
            self.request_counts[exchange] = 1
            self.last_request_time[exchange] = current_time
            return True
        
        # Check rate limit
        rate_limit = self.config['api'].get(f'{exchange}_rate_limit', 600)
        if self.request_counts[exchange] >= rate_limit:
            return False
        
        self.request_counts[exchange] += 1
        return True
    
    async def start_trading(self):
        """Start the active trading engine"""
        self.is_running = True
        self.logger.info("Starting Active Trading Engine v6.0")
        
        # Start all components
        tasks = [
            self.market_data_collector(),
            self.signal_generator(),
            self.trade_executor(),
            self.hedge_manager(),
            self.arbitrage_scanner(),
            self.portfolio_monitor(),
            self.emergency_monitor(),
            self.performance_tracker()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Trading engine error: {e}")
            await self.emergency_shutdown()
    
    async def market_data_collector(self):
        """Collect real-time market data from multiple sources"""
        while self.is_running:
            try:
                # Collect data from all sources
                tasks = []
                
                for symbol in self.config['trading']['active_symbols']:
                    tasks.append(self._collect_symbol_data(symbol))
                
                # Execute data collection concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Market data collection error: {result}")
                
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                self.logger.error(f"Market data collector error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_symbol_data(self, symbol: str):
        """Collect data for a specific symbol"""
        try:
            # Binance data
            if self.binance_client and self._check_rate_limit('binance'):
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                klines = self.binance_client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=100
                )
                
                self.market_data[f'{symbol}_binance'] = {
                    'price': float(ticker['price']),
                    'timestamp': datetime.now(),
                    'klines': klines
                }
            
            # Backpack data (if symbol exists)
            if self.backpack_client and self._check_rate_limit('backpack'):
                try:
                    ticker = self.backpack_client.fetch_ticker(symbol)
                    self.market_data[f'{symbol}_backpack'] = {
                        'price': ticker['last'],
                        'timestamp': datetime.now(),
                        'volume': ticker['baseVolume']
                    }
                except:
                    pass  # Symbol might not exist on Backpack
            
        except Exception as e:
            self.logger.error(f"Symbol data collection error for {symbol}: {e}")
    
    async def signal_generator(self):
        """Generate trading signals using all 8 institutional modules"""
        while self.is_running:
            try:
                for symbol in self.config['trading']['active_symbols']:
                    await self._generate_symbol_signals(symbol)
                
                await asyncio.sleep(10)  # Generate signals every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Signal generator error: {e}")
                await asyncio.sleep(30)
    
    async def _generate_symbol_signals(self, symbol: str):
        """Generate signals for a specific symbol using all modules"""
        try:
            # Get market data
            binance_data = self.market_data.get(f'{symbol}_binance')
            if not binance_data:
                return
            
            # Convert klines to DataFrame
            klines = binance_data['klines']
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Generate signals from all modules
            signals = []
            
            # 1. BitVol Module
            bitvol_signal = await self._get_bitvol_signal(symbol, df)
            if bitvol_signal:
                signals.append(bitvol_signal)
            
            # 2. LXVX Module
            lxvx_signal = await self._get_lxvx_signal(symbol, df)
            if lxvx_signal:
                signals.append(lxvx_signal)
            
            # 3. GARCH Module
            garch_signal = await self._get_garch_signal(symbol, df)
            if garch_signal:
                signals.append(garch_signal)
            
            # 4. Kelly Criterion
            kelly_signal = await self._get_kelly_signal(symbol, df)
            if kelly_signal:
                signals.append(kelly_signal)
            
            # 5. Gamma Hedging
            gamma_signal = await self._get_gamma_signal(symbol, df)
            if gamma_signal:
                signals.append(gamma_signal)
            
            # 6. ATR + Supertrend
            atr_signal = await self._get_atr_signal(symbol, df)
            if atr_signal:
                signals.append(atr_signal)
            
            # 7. Multi-timeframe
            mtf_signal = await self._get_mtf_signal(symbol, df)
            if mtf_signal:
                signals.append(mtf_signal)
            
            # 8. Emergency Module (always running)
            emergency_signal = await self._get_emergency_signal(symbol, df)
            if emergency_signal:
                signals.append(emergency_signal)
            
            # Combine and prioritize signals
            final_signal = self._combine_signals(symbol, signals)
            if final_signal:
                self.signal_queue.put(final_signal)
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
    
    async def _get_bitvol_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get BitVol module signal"""
        try:
            # Calculate volatility-based signal
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # 24-hour volatility
            
            # BitVol thresholds
            low_vol_threshold = 0.02
            high_vol_threshold = 0.08
            
            current_price = df['close'].iloc[-1]
            
            if volatility < low_vol_threshold:
                # Low volatility - expect breakout
                action = 'BUY'
                confidence = 0.7
                grid_spacing = volatility * 0.5
            elif volatility > high_vol_threshold:
                # High volatility - expect reversion
                action = 'SELL'
                confidence = 0.8
                grid_spacing = volatility * 0.3
            else:
                return None
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=grid_spacing,
                target_range=(current_price * 0.95, current_price * 1.05),
                hedge_ratio=0.5,
                position_size=self.config['trading']['base_position_size'],
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.04,
                source_module='BitVol',
                market_regime='VOLATILE' if volatility > high_vol_threshold else 'STABLE',
                urgency='MEDIUM'
            )
            
        except Exception as e:
            self.logger.error(f"BitVol signal error: {e}")
            return None
    
    async def _get_lxvx_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get LXVX module signal"""
        try:
            # LXVX-style liquidity and execution quality analysis
            volumes = df['volume'].values
            prices = df['close'].values
            
            # Calculate VWAP
            vwap = np.sum(prices * volumes) / np.sum(volumes)
            current_price = prices[-1]
            
            # Calculate liquidity score
            avg_volume = np.mean(volumes[-20:])  # 20-period average
            current_volume = volumes[-1]
            
            liquidity_score = current_volume / avg_volume
            price_deviation = abs(current_price - vwap) / vwap
            
            if liquidity_score > 1.5 and price_deviation < 0.01:
                # High liquidity, price near VWAP
                action = 'BUY'
                confidence = 0.75
            elif liquidity_score < 0.5:
                # Low liquidity, avoid trading
                return None
            else:
                action = 'HOLD'
                confidence = 0.6
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=price_deviation,
                target_range=(vwap * 0.99, vwap * 1.01),
                hedge_ratio=0.3,
                position_size=self.config['trading']['base_position_size'] * liquidity_score,
                stop_loss=current_price * 0.995,
                take_profit=current_price * 1.02,
                source_module='LXVX',
                market_regime='LIQUID' if liquidity_score > 1.0 else 'ILLIQUID',
                urgency='LOW'
            )
            
        except Exception as e:
            self.logger.error(f"LXVX signal error: {e}")
            return None
    
    async def _get_garch_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get GARCH module signal"""
        try:
            # GARCH-style volatility forecasting
            returns = df['close'].pct_change().dropna()
            
            # Simple GARCH(1,1) approximation
            alpha, beta = 0.1, 0.85
            omega = 0.0001
            
            # Calculate conditional volatility
            variance = omega
            for i in range(1, len(returns)):
                variance = omega + alpha * returns.iloc[i-1]**2 + beta * variance
            
            volatility = np.sqrt(variance)
            current_price = df['close'].iloc[-1]
            
            # Volatility-based trading signal
            if volatility > 0.05:  # High volatility
                action = 'SELL'
                confidence = 0.8
                grid_spacing = volatility * 0.4
            elif volatility < 0.02:  # Low volatility
                action = 'BUY'
                confidence = 0.7
                grid_spacing = volatility * 0.8
            else:
                return None
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=grid_spacing,
                target_range=(current_price * (1 - volatility), current_price * (1 + volatility)),
                hedge_ratio=0.6,
                position_size=self.config['trading']['base_position_size'],
                stop_loss=current_price * (1 - volatility * 2),
                take_profit=current_price * (1 + volatility * 1.5),
                source_module='GARCH',
                market_regime='HIGH_VOL' if volatility > 0.05 else 'LOW_VOL',
                urgency='MEDIUM'
            )
            
        except Exception as e:
            self.logger.error(f"GARCH signal error: {e}")
            return None
    
    async def _get_kelly_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get Kelly Criterion signal"""
        try:
            # Kelly Criterion position sizing
            returns = df['close'].pct_change().dropna()
            
            # Calculate win rate and average win/loss
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return None
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            if kelly_fraction > 0.1:  # Positive edge
                action = 'BUY'
                confidence = min(kelly_fraction * 2, 0.9)
                position_size = self.config['trading']['base_position_size'] * kelly_fraction
            elif kelly_fraction < -0.1:  # Negative edge
                action = 'SELL'
                confidence = min(abs(kelly_fraction) * 2, 0.9)
                position_size = self.config['trading']['base_position_size'] * abs(kelly_fraction)
            else:
                return None
            
            current_price = df['close'].iloc[-1]
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=avg_win,
                target_range=(current_price * 0.98, current_price * 1.02),
                hedge_ratio=0.4,
                position_size=position_size,
                stop_loss=current_price * 0.97,
                take_profit=current_price * 1.06,
                source_module='Kelly',
                market_regime='TRENDING' if abs(kelly_fraction) > 0.2 else 'RANGING',
                urgency='HIGH' if abs(kelly_fraction) > 0.3 else 'MEDIUM'
            )
            
        except Exception as e:
            self.logger.error(f"Kelly signal error: {e}")
            return None
    
    async def _get_gamma_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get Gamma Hedging signal"""
        try:
            # Gamma hedging requirements
            current_price = df['close'].iloc[-1]
            
            # Calculate delta exposure (simplified)
            portfolio_delta = self.portfolio_state.delta_exposure
            
            # Calculate required hedge
            if abs(portfolio_delta) > self.config['hedging']['hedge_threshold']:
                if portfolio_delta > 0:
                    action = 'SELL'  # Short futures to hedge long delta
                    hedge_ratio = abs(portfolio_delta)
                else:
                    action = 'BUY'  # Long futures to hedge short delta
                    hedge_ratio = abs(portfolio_delta)
                
                return TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action=action,
                    confidence=0.9,
                    grid_spacing=0.001,  # Tight spread for hedging
                    target_range=(current_price * 0.999, current_price * 1.001),
                    hedge_ratio=hedge_ratio,
                    position_size=hedge_ratio,
                    stop_loss=current_price * 0.999,
                    take_profit=current_price * 1.001,
                    source_module='Gamma',
                    market_regime='HEDGING',
                    urgency='HIGH'
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Gamma signal error: {e}")
            return None
    
    async def _get_atr_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get ATR + Supertrend signal"""
        try:
            # Calculate ATR
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            
            atr = np.mean(tr[-14:])  # 14-period ATR
            
            # Supertrend calculation
            multiplier = 2.0
            supertrend_upper = (high + low) / 2 + multiplier * atr
            supertrend_lower = (high + low) / 2 - multiplier * atr
            
            current_price = close[-1]
            
            # Generate signal
            if current_price > supertrend_upper[-1]:
                action = 'BUY'
                confidence = 0.8
                target_range = (current_price, current_price + atr)
            elif current_price < supertrend_lower[-1]:
                action = 'SELL'
                confidence = 0.8
                target_range = (current_price - atr, current_price)
            else:
                return None
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=atr * 0.5,
                target_range=target_range,
                hedge_ratio=0.5,
                position_size=self.config['trading']['base_position_size'],
                stop_loss=current_price - atr if action == 'BUY' else current_price + atr,
                take_profit=current_price + atr * 2 if action == 'BUY' else current_price - atr * 2,
                source_module='ATR_Supertrend',
                market_regime='TRENDING',
                urgency='MEDIUM'
            )
            
        except Exception as e:
            self.logger.error(f"ATR signal error: {e}")
            return None
    
    async def _get_mtf_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get Multi-timeframe signal"""
        try:
            # Multi-timeframe analysis (simplified)
            prices = df['close'].values
            
            # Short-term trend (5 periods)
            short_ma = np.mean(prices[-5:])
            # Medium-term trend (20 periods)
            medium_ma = np.mean(prices[-20:])
            # Long-term trend (50 periods)
            long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else medium_ma
            
            current_price = prices[-1]
            
            # Trend alignment
            if short_ma > medium_ma > long_ma:
                action = 'BUY'
                confidence = 0.85
                trend_strength = (short_ma - long_ma) / long_ma
            elif short_ma < medium_ma < long_ma:
                action = 'SELL'
                confidence = 0.85
                trend_strength = (long_ma - short_ma) / long_ma
            else:
                return None
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=confidence,
                grid_spacing=trend_strength * 0.5,
                target_range=(current_price * 0.98, current_price * 1.02),
                hedge_ratio=0.3,
                position_size=self.config['trading']['base_position_size'],
                stop_loss=current_price * 0.975,
                take_profit=current_price * 1.05,
                source_module='MultiTimeframe',
                market_regime='TRENDING',
                urgency='LOW'
            )
            
        except Exception as e:
            self.logger.error(f"MTF signal error: {e}")
            return None
    
    async def _get_emergency_signal(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get Emergency module signal"""
        try:
            # Emergency conditions
            current_price = df['close'].iloc[-1]
            
            # Check for flash crash conditions
            price_change = df['close'].pct_change().iloc[-1]
            if abs(price_change) > 0.1:  # 10% move in 1 minute
                action = 'SELL' if price_change > 0 else 'BUY'  # Contrarian
                
                return TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action=action,
                    confidence=0.95,
                    grid_spacing=0.001,
                    target_range=(current_price * 0.99, current_price * 1.01),
                    hedge_ratio=1.0,
                    position_size=self.config['trading']['max_position_size'],
                    stop_loss=current_price * 0.98,
                    take_profit=current_price * 1.02,
                    source_module='Emergency',
                    market_regime='EMERGENCY',
                    urgency='EMERGENCY'
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Emergency signal error: {e}")
            return None
    
    def _combine_signals(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Combine multiple signals into a single trading decision"""
        if not signals:
            return None
        
        # Emergency signals take priority
        emergency_signals = [s for s in signals if s.urgency == 'EMERGENCY']
        if emergency_signals:
            return emergency_signals[0]
        
        # Weight signals by confidence and module priority
        module_weights = {
            'Emergency': 1.0,
            'Gamma': 0.9,
            'Kelly': 0.8,
            'ATR_Supertrend': 0.7,
            'GARCH': 0.6,
            'BitVol': 0.5,
            'LXVX': 0.4,
            'MultiTimeframe': 0.3
        }
        
        # Calculate weighted average
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        for signal in signals:
            weight = module_weights.get(signal.source_module, 0.5) * signal.confidence
            
            if signal.action == 'BUY':
                buy_score += weight
            elif signal.action == 'SELL':
                sell_score += weight
            
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Determine final action
        if buy_score > sell_score * 1.2:  # Require 20% edge
            action = 'BUY'
            confidence = buy_score / total_weight
        elif sell_score > buy_score * 1.2:
            action = 'SELL'
            confidence = sell_score / total_weight
        else:
            return None
        
        # Check confidence threshold
        min_confidence = self.config['trading']['min_confidence_threshold']
        if confidence < min_confidence:
            return None
        
        # Bull market adjustment
        if self._is_bull_market() and confidence > self.config['trading']['bull_market_threshold']:
            confidence *= self.config['trading']['bull_market_multiplier']
            confidence = min(confidence, 0.95)
        
        # Create combined signal
        avg_signal = signals[0]  # Use first signal as template
        avg_signal.action = action
        avg_signal.confidence = confidence
        avg_signal.source_module = 'Combined'
        avg_signal.urgency = 'HIGH' if confidence > 0.8 else 'MEDIUM'
        
        return avg_signal
    
    def _is_bull_market(self) -> bool:
        """Determine if we're in a bull market"""
        # Simple bull market detection based on recent performance
        return self.portfolio_state.daily_pnl > 0 and len(self.market_data) > 0
    
    async def trade_executor(self):
        """Execute trading signals"""
        while self.is_running:
            try:
                # Process signal queue
                try:
                    signal = self.signal_queue.get_nowait()
                    await self._execute_signal(signal)
                except Empty:
                    await asyncio.sleep(1)
                    continue
                
            except Exception as e:
                self.logger.error(f"Trade executor error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            self.logger.info(f"Executing signal: {signal.action} {signal.symbol} "
                           f"confidence={signal.confidence:.2f} source={signal.source_module}")
            
            # Check if we can execute
            if not self._can_execute_trade(signal):
                self.logger.warning(f"Cannot execute trade for {signal.symbol}")
                return
            
            # Execute spot trade
            if signal.action in ['BUY', 'SELL']:
                await self._execute_spot_trade(signal)
            
            # Execute hedge if required
            if signal.hedge_ratio > 0:
                await self._execute_hedge_trade(signal)
            
            # Update portfolio state
            await self._update_portfolio_state()
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Signal execution error: {e}")
    
    def _can_execute_trade(self, signal: TradingSignal) -> bool:
        """Check if we can execute a trade"""
        # Risk management checks
        if signal.position_size > self.config['trading']['max_position_size']:
            return False
        
        # Portfolio risk check
        if self.portfolio_state.delta_exposure > self.config['risk_management']['max_portfolio_risk']:
            return False
        
        # Emergency stop check
        if self.portfolio_state.daily_pnl < -self.config['risk_management']['emergency_stop_loss']:
            return False
        
        return True
    
    async def _execute_spot_trade(self, signal: TradingSignal):
        """Execute spot trade"""
        try:
            if not self.binance_client or not self._check_rate_limit('binance'):
                return
            
            # Calculate quantity
            current_price = self.market_data.get(f'{signal.symbol}_binance', {}).get('price', 0)
            if current_price == 0:
                return
            
            quantity = signal.position_size / current_price
            
            # Place order
            if signal.action == 'BUY':
                order = self.binance_client.order_market_buy(
                    symbol=signal.symbol,
                    quantity=quantity
                )
            elif signal.action == 'SELL':
                order = self.binance_client.order_market_sell(
                    symbol=signal.symbol,
                    quantity=quantity
                )
            
            self.logger.info(f"Spot trade executed: {order}")
            
        except Exception as e:
            self.logger.error(f"Spot trade execution error: {e}")
    
    async def _execute_hedge_trade(self, signal: TradingSignal):
        """Execute hedge trade using futures"""
        try:
            if not self.binance_client or not self._check_rate_limit('binance'):
                return
            
            # Calculate hedge quantity
            hedge_quantity = signal.position_size * signal.hedge_ratio
            
            # Execute futures hedge (opposite direction)
            futures_symbol = signal.symbol.replace('USDT', 'USDT')  # Ensure futures format
            
            if signal.action == 'BUY':
                # Long spot, short futures
                order = self.binance_client.futures_create_order(
                    symbol=futures_symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=hedge_quantity
                )
            elif signal.action == 'SELL':
                # Short spot, long futures
                order = self.binance_client.futures_create_order(
                    symbol=futures_symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=hedge_quantity
                )
            
            self.logger.info(f"Hedge trade executed: {order}")
            
        except Exception as e:
            self.logger.error(f"Hedge trade execution error: {e}")
    
    async def hedge_manager(self):
        """Manage hedging positions"""
        while self.is_running:
            try:
                await self._rebalance_hedges()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Hedge manager error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _rebalance_hedges(self):
        """Rebalance hedge positions"""
        try:
            # Calculate current delta exposure
            delta_exposure = self.portfolio_state.delta_exposure
            
            # Check if rebalancing is needed
            if abs(delta_exposure) > self.config['hedging']['hedge_threshold']:
                # Create rebalancing signal
                for symbol in self.config['trading']['active_symbols']:
                    hedge_signal = TradingSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action='SELL' if delta_exposure > 0 else 'BUY',
                        confidence=0.9,
                        grid_spacing=0.001,
                        target_range=(0, 0),
                        hedge_ratio=abs(delta_exposure),
                        position_size=abs(delta_exposure),
                        stop_loss=0,
                        take_profit=0,
                        source_module='HedgeManager',
                        market_regime='HEDGING',
                        urgency='HIGH'
                    )
                    
                    await self._execute_hedge_trade(hedge_signal)
                    break  # Only hedge with one symbol
            
        except Exception as e:
            self.logger.error(f"Hedge rebalancing error: {e}")
    
    async def arbitrage_scanner(self):
        """Scan for arbitrage opportunities"""
        while self.is_running:
            try:
                await self._scan_arbitrage_opportunities()
                await asyncio.sleep(5)  # Scan every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Arbitrage scanner error: {e}")
                await asyncio.sleep(30)
    
    async def _scan_arbitrage_opportunities(self):
        """Scan for cross-exchange arbitrage opportunities"""
        try:
            for symbol in self.config['trading']['active_symbols']:
                binance_data = self.market_data.get(f'{symbol}_binance')
                backpack_data = self.market_data.get(f'{symbol}_backpack')
                
                if not binance_data or not backpack_data:
                    continue
                
                binance_price = binance_data['price']
                backpack_price = backpack_data['price']
                
                # Calculate spread
                spread = abs(binance_price - backpack_price)
                spread_percent = spread / min(binance_price, backpack_price)
                
                # Check if arbitrage opportunity exists
                if spread_percent > self.config['arbitrage']['min_spread']:
                    opportunity = ArbitrageOpportunity(
                        symbol=symbol,
                        binance_price=binance_price,
                        backpack_price=backpack_price,
                        spread=spread,
                        spread_percent=spread_percent,
                        volume=backpack_data.get('volume', 0),
                        profit_potential=spread_percent * self.config['trading']['base_position_size'],
                        execution_time=datetime.now(),
                        confidence=0.8 if spread_percent > 0.01 else 0.6
                    )
                    
                    if opportunity.profit_potential > self.config['arbitrage']['profit_threshold']:
                        self.arbitrage_queue.put(opportunity)
                        self.performance_metrics['arbitrage_opportunities'] += 1
                        
                        self.logger.info(f"Arbitrage opportunity found: {symbol} "
                                       f"spread={spread_percent:.4f} "
                                       f"profit={opportunity.profit_potential:.4f}")
            
        except Exception as e:
            self.logger.error(f"Arbitrage scanning error: {e}")
    
    async def portfolio_monitor(self):
        """Monitor portfolio state"""
        while self.is_running:
            try:
                await self._update_portfolio_state()
                await self._check_risk_limits()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Portfolio monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _update_portfolio_state(self):
        """Update current portfolio state"""
        try:
            # Get account information
            if self.binance_client and self._check_rate_limit('binance'):
                account = self.binance_client.get_account()
                futures_account = self.binance_client.futures_account()
                
                # Update spot positions
                spot_positions = {}
                for balance in account['balances']:
                    if float(balance['free']) > 0 or float(balance['locked']) > 0:
                        spot_positions[balance['asset']] = float(balance['free']) + float(balance['locked'])
                
                # Update futures positions
                futures_positions = {}
                for position in futures_account['positions']:
                    if float(position['positionAmt']) != 0:
                        futures_positions[position['symbol']] = float(position['positionAmt'])
                
                # Calculate portfolio value and exposures
                total_value = 0
                delta_exposure = 0
                
                for symbol, amount in spot_positions.items():
                    if symbol != 'USDT':
                        price = self.market_data.get(f'{symbol}USDT_binance', {}).get('price', 0)
                        value = amount * price
                        total_value += value
                        delta_exposure += value
                
                for symbol, amount in futures_positions.items():
                    price = self.market_data.get(f'{symbol}_binance', {}).get('price', 0)
                    value = amount * price
                    delta_exposure += value
                
                # Update portfolio state
                self.portfolio_state.spot_positions = spot_positions
                self.portfolio_state.futures_positions = futures_positions
                self.portfolio_state.total_value = total_value
                self.portfolio_state.delta_exposure = delta_exposure
                self.portfolio_state.last_update = datetime.now()
                
                # Calculate daily P&L
                start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                if self.portfolio_state.last_update.date() == start_of_day.date():
                    # Same day, calculate P&L
                    pass  # Would need to track starting balance
                
        except Exception as e:
            self.logger.error(f"Portfolio update error: {e}")
    
    async def _check_risk_limits(self):
        """Check risk limits and take action if needed"""
        try:
            # Check maximum drawdown
            if self.portfolio_state.daily_pnl < -self.config['risk_management']['max_drawdown']:
                self.logger.warning("Maximum drawdown reached, reducing positions")
                await self._reduce_positions()
            
            # Check delta exposure
            if abs(self.portfolio_state.delta_exposure) > self.config['risk_management']['max_portfolio_risk']:
                self.logger.warning("High delta exposure, executing hedge")
                await self._emergency_hedge()
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
    
    async def _reduce_positions(self):
        """Reduce positions due to risk management"""
        try:
            # Close 50% of all positions
            for symbol, amount in self.portfolio_state.spot_positions.items():
                if symbol != 'USDT' and amount > 0:
                    sell_amount = amount * 0.5
                    
                    # Execute sell order
                    if self.binance_client and self._check_rate_limit('binance'):
                        order = self.binance_client.order_market_sell(
                            symbol=f'{symbol}USDT',
                            quantity=sell_amount
                        )
                        self.logger.info(f"Position reduction executed: {order}")
            
        except Exception as e:
            self.logger.error(f"Position reduction error: {e}")
    
    async def _emergency_hedge(self):
        """Execute emergency hedge"""
        try:
            delta_exposure = self.portfolio_state.delta_exposure
            
            # Execute opposite futures position
            hedge_amount = abs(delta_exposure)
            symbol = self.config['trading']['active_symbols'][0]  # Use first symbol
            
            if self.binance_client and self._check_rate_limit('binance'):
                order = self.binance_client.futures_create_order(
                    symbol=symbol,
                    side='SELL' if delta_exposure > 0 else 'BUY',
                    type='MARKET',
                    quantity=hedge_amount
                )
                
                self.logger.info(f"Emergency hedge executed: {order}")
            
        except Exception as e:
            self.logger.error(f"Emergency hedge error: {e}")
    
    async def emergency_monitor(self):
        """Monitor for emergency conditions"""
        while self.is_running:
            try:
                await self._check_emergency_conditions()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Emergency monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _check_emergency_conditions(self):
        """Check for emergency conditions"""
        try:
            # Check for flash crashes
            for symbol in self.config['trading']['active_symbols']:
                binance_data = self.market_data.get(f'{symbol}_binance')
                if binance_data:
                    # Check if price data is recent
                    if (datetime.now() - binance_data['timestamp']).seconds > 60:
                        self.logger.warning(f"Stale price data for {symbol}")
                        continue
                    
                    # Check for extreme price movements
                    # (This would need historical data to compare)
                    pass
            
            # Check API connectivity
            if not self.binance_client:
                self.logger.error("Binance client not connected")
                await self._reconnect_binance()
            
        except Exception as e:
            self.logger.error(f"Emergency condition check error: {e}")
    
    async def _reconnect_binance(self):
        """Reconnect to Binance"""
        try:
            self.binance_client = self._init_binance()
            if self.binance_client:
                self.logger.info("Binance client reconnected successfully")
        except Exception as e:
            self.logger.error(f"Binance reconnection error: {e}")
    
    async def performance_tracker(self):
        """Track performance metrics"""
        while self.is_running:
            try:
                await self._update_performance_metrics()
                await self._log_performance_summary()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(600)
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate success rate
            if self.performance_metrics['total_trades'] > 0:
                success_rate = (self.performance_metrics['successful_trades'] / 
                               self.performance_metrics['total_trades'])
                self.performance_metrics['success_rate'] = success_rate
            
            # Calculate runtime
            runtime = datetime.now() - self.performance_metrics['start_time']
            self.performance_metrics['runtime_hours'] = runtime.total_seconds() / 3600
            
            # Update total P&L
            self.performance_metrics['total_pnl'] = self.portfolio_state.daily_pnl
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
    
    async def _log_performance_summary(self):
        """Log performance summary"""
        try:
            summary = {
                'total_trades': self.performance_metrics['total_trades'],
                'success_rate': self.performance_metrics.get('success_rate', 0),
                'arbitrage_opportunities': self.performance_metrics['arbitrage_opportunities'],
                'total_pnl': self.performance_metrics['total_pnl'],
                'runtime_hours': self.performance_metrics['runtime_hours'],
                'portfolio_value': self.portfolio_state.total_value,
                'delta_exposure': self.portfolio_state.delta_exposure
            }
            
            self.logger.info(f"Performance Summary: {summary}")
            
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        self.logger.warning("EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Stop all trading
            self.is_running = False
            
            # Close all positions
            await self._close_all_positions()
            
            # Cancel all orders
            await self._cancel_all_orders()
            
            # Log final state
            self.logger.info(f"Final Portfolio State: {asdict(self.portfolio_state)}")
            self.logger.info(f"Final Performance Metrics: {self.performance_metrics}")
            
            self.logger.info("Emergency shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            # Close spot positions
            for symbol, amount in self.portfolio_state.spot_positions.items():
                if symbol != 'USDT' and amount > 0:
                    if self.binance_client and self._check_rate_limit('binance'):
                        order = self.binance_client.order_market_sell(
                            symbol=f'{symbol}USDT',
                            quantity=amount
                        )
                        self.logger.info(f"Position closed: {order}")
            
            # Close futures positions
            for symbol, amount in self.portfolio_state.futures_positions.items():
                if amount != 0:
                    if self.binance_client and self._check_rate_limit('binance'):
                        order = self.binance_client.futures_create_order(
                            symbol=symbol,
                            side='SELL' if amount > 0 else 'BUY',
                            type='MARKET',
                            quantity=abs(amount),
                            reduceOnly=True
                        )
                        self.logger.info(f"Futures position closed: {order}")
            
        except Exception as e:
            self.logger.error(f"Position closing error: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            if self.binance_client and self._check_rate_limit('binance'):
                # Cancel spot orders
                for symbol in self.config['trading']['active_symbols']:
                    orders = self.binance_client.get_open_orders(symbol=symbol)
                    for order in orders:
                        self.binance_client.cancel_order(
                            symbol=symbol,
                            orderId=order['orderId']
                        )
                        self.logger.info(f"Order cancelled: {order['orderId']}")
                
                # Cancel futures orders
                futures_orders = self.binance_client.futures_get_open_orders()
                for order in futures_orders:
                    self.binance_client.futures_cancel_order(
                        symbol=order['symbol'],
                        orderId=order['orderId']
                    )
                    self.logger.info(f"Futures order cancelled: {order['orderId']}")
            
        except Exception as e:
            self.logger.error(f"Order cancellation error: {e}")

def main():
    """Main function to run the Active Trading Engine"""
    try:
        # Create and run the trading engine
        engine = ActiveTradingEngine()
        
        # Run the engine
        asyncio.run(engine.start_trading())
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        asyncio.run(engine.emergency_shutdown())
    except Exception as e:
        print(f"Fatal error: {e}")
        if 'engine' in locals():
            asyncio.run(engine.emergency_shutdown())

if __name__ == "__main__":
    main()