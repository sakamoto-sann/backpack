#!/usr/bin/env python3
"""
🎯 PAPER TRADING ENGINE v1.0.0
Live simulation environment with real market data and virtual execution

Features:
- 📡 Real-time Market Data Integration
- 💰 Virtual Portfolio Management
- 📊 Live Performance Tracking
- 🎭 Exchange API Simulation
- 🚀 Strategy Validation Environment
- 📈 Risk Management Testing
- 🔄 OMS/PMS Integration
- 📱 Real-time Monitoring Dashboard
- 🚨 Alerts and Notifications
- 📋 Comprehensive Analytics
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor

# Import core components
from ..data.market_data_feeder import MarketDataFeeder, TickerData, OrderBookData, DataType
from ..core.order_management_system import (
    OrderManagementSystem, OrderRequest, OrderExecution, OrderStatus, 
    OrderType, OrderSide, ArbitrageExecution, OMSMetrics
)
from ..core.position_management_system import PositionManagementSystem
from ..risk_management.integrated_risk_manager import IntegratedRiskManager
from ..strategies.arbitrage_detector import ArbitrageDetector, ArbitrageOpportunity

logger = logging.getLogger(__name__)

class PaperTradingStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class SimulationMode(Enum):
    CONSERVATIVE = "conservative"  # Realistic slippage and delays
    OPTIMISTIC = "optimistic"     # Minimal slippage and delays
    AGGRESSIVE = "aggressive"     # Maximum slippage and delays

@dataclass
class VirtualBalance:
    """Virtual balance for paper trading."""
    asset: str
    total: float
    available: float
    locked: float = 0.0
    
    def lock_funds(self, amount: float) -> bool:
        """Lock funds for an order."""
        if self.available >= amount:
            self.available -= amount
            self.locked += amount
            return True
        return False
    
    def unlock_funds(self, amount: float):
        """Unlock funds after order cancellation."""
        unlock_amount = min(amount, self.locked)
        self.locked -= unlock_amount
        self.available += unlock_amount
    
    def execute_trade(self, amount: float, is_buy: bool = True):
        """Execute a trade - add or remove funds."""
        if is_buy:
            self.available += amount
            self.total += amount
        else:
            # Selling - remove from locked funds
            remove_amount = min(amount, self.locked)
            self.locked -= remove_amount
            self.total -= remove_amount

@dataclass
class VirtualOrder:
    """Virtual order in paper trading system."""
    id: str
    exchange: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    fill_time: Optional[datetime] = None
    fees_paid: float = 0.0
    
    # Simulation parameters
    expected_slippage: float = 0.0
    expected_delay_ms: float = 0.0
    fill_probability: float = 1.0

@dataclass
class SimulationSettings:
    """Paper trading simulation settings."""
    # Execution simulation
    base_latency_ms: float = 50.0
    max_latency_ms: float = 500.0
    base_slippage_bps: float = 5.0  # 0.05%
    max_slippage_bps: float = 50.0  # 0.5%
    
    # Fill simulation
    partial_fill_probability: float = 0.1
    rejection_probability: float = 0.02
    
    # Fees (per exchange)
    binance_maker_fee: float = 0.001  # 0.1%
    binance_taker_fee: float = 0.001  # 0.1%
    backpack_maker_fee: float = 0.0005  # 0.05%
    backpack_taker_fee: float = 0.0007  # 0.07%
    
    # Market impact
    enable_market_impact: bool = True
    market_impact_factor: float = 0.1
    
    # Realistic constraints
    enable_realistic_constraints: bool = True
    max_order_size_ratio: float = 0.1  # 10% of 24h volume

@dataclass
class PaperTradingMetrics:
    """Paper trading performance metrics."""
    session_start: datetime = field(default_factory=datetime.now)
    
    # Trading metrics
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    
    # Portfolio metrics
    starting_portfolio_value: float = 0.0
    current_portfolio_value: float = 0.0
    total_pnl: float = 0.0
    total_fees_paid: float = 0.0
    
    # Arbitrage metrics
    arbitrage_opportunities: int = 0
    successful_arbitrages: int = 0
    arbitrage_pnl: float = 0.0
    
    # Performance metrics
    avg_execution_time_ms: float = 0.0
    avg_slippage_bps: float = 0.0
    success_rate: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0
    portfolio_volatility: float = 0.0
    
    def calculate_success_rate(self):
        """Calculate order success rate."""
        if self.total_orders > 0:
            self.success_rate = self.filled_orders / self.total_orders * 100
        return self.success_rate
    
    def calculate_pnl_pct(self):
        """Calculate PnL percentage."""
        if self.starting_portfolio_value > 0:
            return (self.total_pnl / self.starting_portfolio_value) * 100
        return 0.0

@dataclass
class AlertConfig:
    """Alert configuration for paper trading."""
    enable_alerts: bool = True
    
    # PnL alerts
    pnl_threshold_pct: float = 5.0  # Alert on 5% PnL change
    
    # Risk alerts
    drawdown_threshold_pct: float = 10.0  # Alert on 10% drawdown
    
    # Performance alerts
    low_success_rate_threshold: float = 80.0  # Alert if success rate < 80%
    
    # System alerts
    high_latency_threshold_ms: float = 1000.0  # Alert on high latency
    connection_alerts: bool = True

class VirtualExchange:
    """Virtual exchange simulator for paper trading."""
    
    def __init__(self, exchange_name: str, settings: SimulationSettings):
        self.exchange_name = exchange_name
        self.settings = settings
        
        # Order simulation
        self.pending_orders: Dict[str, VirtualOrder] = {}
        self.order_queue = asyncio.Queue()
        
        # Market data references
        self.latest_tickers: Dict[str, TickerData] = {}
        self.latest_orderbooks: Dict[str, OrderBookData] = {}
        
        # Simulation state
        self.is_running = False
        
    async def start(self):
        """Start the virtual exchange."""
        self.is_running = True
        asyncio.create_task(self._order_processor())
        logger.info(f"🎭 Virtual {self.exchange_name} exchange started")
    
    async def stop(self):
        """Stop the virtual exchange."""
        self.is_running = False
        logger.info(f"🎭 Virtual {self.exchange_name} exchange stopped")
    
    def update_market_data(self, ticker: TickerData = None, orderbook: OrderBookData = None):
        """Update market data for simulation."""
        if ticker:
            self.latest_tickers[ticker.symbol] = ticker
        if orderbook:
            self.latest_orderbooks[orderbook.symbol] = orderbook
    
    async def place_order(self, order_request: OrderRequest) -> VirtualOrder:
        """Place a virtual order."""
        try:
            # Create virtual order
            virtual_order = VirtualOrder(
                id=str(uuid.uuid4()),
                exchange=self.exchange_name,
                symbol=order_request.symbol,
                side=order_request.side,
                type=order_request.type,
                quantity=order_request.quantity,
                price=order_request.price
            )
            
            # Calculate simulation parameters
            await self._calculate_simulation_params(virtual_order)
            
            # Add to pending orders
            self.pending_orders[virtual_order.id] = virtual_order
            
            # Queue for processing
            await self.order_queue.put(virtual_order.id)
            
            logger.info(f"📝 Virtual order placed: {virtual_order.id} on {self.exchange_name}")
            
            return virtual_order
            
        except Exception as e:
            logger.error(f"❌ Virtual order placement error: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a virtual order."""
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    order.status = OrderStatus.CANCELLED
                    logger.info(f"❌ Virtual order cancelled: {order_id}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Virtual order cancellation error: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[VirtualOrder]:
        """Get virtual order status."""
        return self.pending_orders.get(order_id)
    
    async def _order_processor(self):
        """Process virtual orders with realistic simulation."""
        while self.is_running:
            try:
                # Get next order from queue
                order_id = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                
                if order_id in self.pending_orders:
                    order = self.pending_orders[order_id]
                    await self._simulate_order_execution(order)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Order processor error: {e}")
    
    async def _simulate_order_execution(self, order: VirtualOrder):
        """Simulate realistic order execution."""
        try:
            # Check if order was cancelled
            if order.status == OrderStatus.CANCELLED:
                return
            
            # Update status to submitted
            order.status = OrderStatus.SUBMITTED
            
            # Simulate network latency
            await asyncio.sleep(order.expected_delay_ms / 1000.0)
            
            # Check for rejection
            if np.random.random() < self.settings.rejection_probability:
                order.status = OrderStatus.REJECTED
                logger.warning(f"🚫 Virtual order rejected: {order.id}")
                return
            
            # Get current market data
            ticker = self.latest_tickers.get(order.symbol)
            if not ticker:
                order.status = OrderStatus.REJECTED
                logger.warning(f"🚫 No market data for {order.symbol}")
                return
            
            # Determine fill price and quantity
            fill_price, fill_quantity = await self._calculate_fill(order, ticker)
            
            if fill_quantity > 0:
                # Calculate fees
                fees = self._calculate_fees(fill_quantity, fill_price)
                
                # Update order
                order.filled_quantity = fill_quantity
                order.average_price = fill_price
                order.fees_paid = fees
                order.fill_time = datetime.now()
                
                # Determine final status
                if fill_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                logger.info(f"✅ Virtual order filled: {order.id} "
                          f"({fill_quantity} @ ${fill_price:.2f})")
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"🚫 Virtual order could not be filled: {order.id}")
                
        except Exception as e:
            logger.error(f"❌ Order simulation error: {e}")
            order.status = OrderStatus.FAILED
    
    async def _calculate_simulation_params(self, order: VirtualOrder):
        """Calculate realistic simulation parameters."""
        try:
            # Base latency with random variation
            order.expected_delay_ms = self.settings.base_latency_ms + (
                np.random.exponential(self.settings.max_latency_ms - self.settings.base_latency_ms)
            )
            
            # Slippage based on market conditions and order size
            base_slippage = self.settings.base_slippage_bps / 10000.0
            max_slippage = self.settings.max_slippage_bps / 10000.0
            
            # Market impact based on order size
            ticker = self.latest_tickers.get(order.symbol)
            if ticker and self.settings.enable_market_impact:
                order_value = order.quantity * (order.price or ticker.price)
                volume_ratio = order_value / (ticker.volume_24h * ticker.price)
                impact_factor = min(volume_ratio * self.settings.market_impact_factor, max_slippage)
                order.expected_slippage = base_slippage + impact_factor
            else:
                order.expected_slippage = base_slippage + np.random.uniform(0, max_slippage - base_slippage)
            
            # Fill probability based on order type and market conditions
            if order.type == OrderType.MARKET:
                order.fill_probability = 0.98  # Market orders almost always fill
            else:
                # Limit orders depend on price relative to market
                if ticker:
                    if order.side == OrderSide.BUY:
                        if order.price >= ticker.ask:
                            order.fill_probability = 0.95
                        elif order.price >= ticker.bid:
                            order.fill_probability = 0.7
                        else:
                            order.fill_probability = 0.3
                    else:  # SELL
                        if order.price <= ticker.bid:
                            order.fill_probability = 0.95
                        elif order.price <= ticker.ask:
                            order.fill_probability = 0.7
                        else:
                            order.fill_probability = 0.3
                else:
                    order.fill_probability = 0.5
                    
        except Exception as e:
            logger.error(f"❌ Simulation parameters calculation error: {e}")
    
    async def _calculate_fill(self, order: VirtualOrder, ticker: TickerData) -> Tuple[float, float]:
        """Calculate fill price and quantity."""
        try:
            # Check fill probability
            if np.random.random() > order.fill_probability:
                return 0.0, 0.0
            
            # Determine base fill price
            if order.type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    base_price = ticker.ask
                else:
                    base_price = ticker.bid
            else:  # LIMIT
                base_price = order.price
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + order.expected_slippage)
            else:
                fill_price = base_price * (1 - order.expected_slippage)
            
            # Determine fill quantity
            if np.random.random() < self.settings.partial_fill_probability:
                # Partial fill
                fill_quantity = order.quantity * np.random.uniform(0.5, 0.95)
            else:
                # Full fill
                fill_quantity = order.quantity
            
            return fill_price, fill_quantity
            
        except Exception as e:
            logger.error(f"❌ Fill calculation error: {e}")
            return 0.0, 0.0
    
    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees."""
        try:
            order_value = quantity * price
            
            if self.exchange_name == "binance":
                # Use taker fee as default for simulation
                fee_rate = self.settings.binance_taker_fee
            elif self.exchange_name == "backpack":
                fee_rate = self.settings.backpack_taker_fee
            else:
                fee_rate = 0.001  # Default 0.1%
            
            return order_value * fee_rate
            
        except Exception as e:
            logger.error(f"❌ Fee calculation error: {e}")
            return 0.0

class VirtualPortfolio:
    """Virtual portfolio manager for paper trading."""
    
    def __init__(self, initial_balances: Dict[str, float]):
        self.balances: Dict[str, VirtualBalance] = {}
        self.initial_balances = initial_balances.copy()
        
        # Initialize balances
        for asset, amount in initial_balances.items():
            self.balances[asset] = VirtualBalance(asset, amount, amount)
        
        # Portfolio tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.pnl_history = deque(maxlen=1000)
        self.last_portfolio_value = 0.0
        
        logger.info(f"💰 Virtual portfolio initialized with: {initial_balances}")
    
    def get_balance(self, asset: str) -> VirtualBalance:
        """Get balance for an asset."""
        if asset not in self.balances:
            self.balances[asset] = VirtualBalance(asset, 0.0, 0.0)
        return self.balances[asset]
    
    def can_place_order(self, order: VirtualOrder, prices: Dict[str, float]) -> bool:
        """Check if order can be placed with current balances."""
        try:
            if order.side == OrderSide.BUY:
                # Need quote currency
                quote_asset = self._get_quote_asset(order.symbol)
                required_amount = order.quantity * (order.price or prices.get(order.symbol, 0))
                return self.get_balance(quote_asset).available >= required_amount
            else:
                # Need base currency
                base_asset = self._get_base_asset(order.symbol)
                return self.get_balance(base_asset).available >= order.quantity
                
        except Exception as e:
            logger.error(f"❌ Order validation error: {e}")
            return False
    
    def reserve_funds(self, order: VirtualOrder, prices: Dict[str, float]) -> bool:
        """Reserve funds for an order."""
        try:
            if order.side == OrderSide.BUY:
                quote_asset = self._get_quote_asset(order.symbol)
                required_amount = order.quantity * (order.price or prices.get(order.symbol, 0))
                return self.get_balance(quote_asset).lock_funds(required_amount)
            else:
                base_asset = self._get_base_asset(order.symbol)
                return self.get_balance(base_asset).lock_funds(order.quantity)
                
        except Exception as e:
            logger.error(f"❌ Fund reservation error: {e}")
            return False
    
    def execute_trade(self, order: VirtualOrder):
        """Execute a trade and update balances."""
        try:
            base_asset = self._get_base_asset(order.symbol)
            quote_asset = self._get_quote_asset(order.symbol)
            
            if order.side == OrderSide.BUY:
                # Add base asset
                self.get_balance(base_asset).execute_trade(order.filled_quantity, True)
                
                # Remove quote asset (from locked funds)
                quote_amount = order.filled_quantity * order.average_price + order.fees_paid
                self.get_balance(quote_asset).execute_trade(quote_amount, False)
            else:
                # Remove base asset (from locked funds)
                self.get_balance(base_asset).execute_trade(order.filled_quantity, False)
                
                # Add quote asset
                quote_amount = order.filled_quantity * order.average_price - order.fees_paid
                self.get_balance(quote_asset).execute_trade(quote_amount, True)
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'order_id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.filled_quantity,
                'price': order.average_price,
                'fees': order.fees_paid,
                'pnl': self._calculate_trade_pnl(order)
            })
            
            logger.info(f"💰 Trade executed: {order.side.value} {order.filled_quantity} {order.symbol} @ ${order.average_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
    
    def cancel_order_funds(self, order: VirtualOrder, prices: Dict[str, float]):
        """Release reserved funds when order is cancelled."""
        try:
            if order.side == OrderSide.BUY:
                quote_asset = self._get_quote_asset(order.symbol)
                locked_amount = (order.quantity - order.filled_quantity) * (order.price or prices.get(order.symbol, 0))
                self.get_balance(quote_asset).unlock_funds(locked_amount)
            else:
                base_asset = self._get_base_asset(order.symbol)
                self.get_balance(base_asset).unlock_funds(order.quantity - order.filled_quantity)
                
        except Exception as e:
            logger.error(f"❌ Fund release error: {e}")
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in USD."""
        try:
            total_value = 0.0
            
            for asset, balance in self.balances.items():
                if asset == 'USDT' or asset == 'USDC' or asset == 'USD':
                    total_value += balance.total
                else:
                    # Get price in USDT
                    symbol = f"{asset}USDT"
                    price = prices.get(symbol, 0.0)
                    total_value += balance.total * price
            
            return total_value
            
        except Exception as e:
            logger.error(f"❌ Portfolio value calculation error: {e}")
            return 0.0
    
    def update_portfolio_history(self, prices: Dict[str, float]):
        """Update portfolio history for performance tracking."""
        try:
            current_value = self.calculate_portfolio_value(prices)
            
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'total_value': current_value,
                'balances': {asset: balance.total for asset, balance in self.balances.items()},
                'pnl': current_value - sum(self.initial_balances.values())
            })
            
            # Update PnL history
            if self.last_portfolio_value > 0:
                pnl_change = current_value - self.last_portfolio_value
                self.pnl_history.append(pnl_change)
            
            self.last_portfolio_value = current_value
            
        except Exception as e:
            logger.error(f"❌ Portfolio history update error: {e}")
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol."""
        if symbol.endswith('USDT'):
            return symbol[:-4]
        elif symbol.endswith('USDC'):
            return symbol[:-4]
        elif symbol.endswith('BTC'):
            return symbol[:-3]
        else:
            return symbol.split('/')[0] if '/' in symbol else symbol[:3]
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extract quote asset from symbol."""
        if symbol.endswith('USDT'):
            return 'USDT'
        elif symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('BTC'):
            return 'BTC'
        else:
            return symbol.split('/')[1] if '/' in symbol else 'USDT'
    
    def _calculate_trade_pnl(self, order: VirtualOrder) -> float:
        """Calculate PnL for a trade (simplified)."""
        # This is a simplified calculation
        # In practice, you'd need to track cost basis properly
        return 0.0

class AlertManager:
    """Alert and notification manager for paper trading."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        
    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    async def check_pnl_alerts(self, current_pnl_pct: float, previous_pnl_pct: float):
        """Check for PnL-based alerts."""
        if not self.config.enable_alerts:
            return
        
        pnl_change = abs(current_pnl_pct - previous_pnl_pct)
        if pnl_change >= self.config.pnl_threshold_pct:
            await self._send_alert(
                "PnL Alert",
                f"Portfolio PnL changed by {pnl_change:.2f}% (now {current_pnl_pct:.2f}%)",
                "warning" if current_pnl_pct < 0 else "info"
            )
    
    async def check_drawdown_alert(self, drawdown_pct: float):
        """Check for drawdown alerts."""
        if not self.config.enable_alerts:
            return
        
        if drawdown_pct >= self.config.drawdown_threshold_pct:
            await self._send_alert(
                "Drawdown Alert",
                f"Portfolio drawdown reached {drawdown_pct:.2f}%",
                "error"
            )
    
    async def check_performance_alerts(self, success_rate: float, avg_latency: float):
        """Check for performance alerts."""
        if not self.config.enable_alerts:
            return
        
        if success_rate < self.config.low_success_rate_threshold:
            await self._send_alert(
                "Performance Alert",
                f"Order success rate dropped to {success_rate:.1f}%",
                "warning"
            )
        
        if avg_latency > self.config.high_latency_threshold_ms:
            await self._send_alert(
                "Latency Alert",
                f"Average execution latency: {avg_latency:.0f}ms",
                "warning"
            )
    
    async def _send_alert(self, title: str, message: str, level: str = "info"):
        """Send an alert."""
        try:
            alert = {
                'timestamp': datetime.now(),
                'title': title,
                'message': message,
                'level': level
            }
            
            self.alert_history.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"❌ Alert callback error: {e}")
            
            # Log alert
            if level == "error":
                logger.error(f"🚨 {title}: {message}")
            elif level == "warning":
                logger.warning(f"⚠️ {title}: {message}")
            else:
                logger.info(f"ℹ️ {title}: {message}")
                
        except Exception as e:
            logger.error(f"❌ Alert sending error: {e}")

class PaperTradingEngine:
    """
    Comprehensive Paper Trading Engine for live simulation testing.
    Provides a realistic trading environment with virtual execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.market_data_feeder = None
        self.virtual_exchanges: Dict[str, VirtualExchange] = {}
        self.virtual_portfolio = None
        self.alert_manager = AlertManager(AlertConfig(**config.get('alerts', {})))
        
        # Strategy components (will be injected)
        self.arbitrage_detector = None
        self.risk_manager = None
        
        # Simulation settings
        self.simulation_settings = SimulationSettings(**config.get('simulation', {}))
        self.simulation_mode = SimulationMode(config.get('mode', 'conservative'))
        
        # State management
        self.status = PaperTradingStatus.STOPPED
        self.session_id = str(uuid.uuid4())
        self.start_time = None
        
        # Performance tracking
        self.metrics = PaperTradingMetrics()
        self.metrics_history = deque(maxlen=1000)
        
        # Data storage
        self.db_path = config.get('db_path', 'paper_trading.db')
        self._init_database()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        logger.info("🎯 Paper Trading Engine v1.0.0 initialized")
        logger.info(f"   Session ID: {self.session_id}")
        logger.info(f"   Simulation Mode: {self.simulation_mode.value}")
    
    async def initialize(self, market_data_feeder: MarketDataFeeder, 
                        arbitrage_detector: ArbitrageDetector = None,
                        risk_manager: IntegratedRiskManager = None,
                        initial_balances: Dict[str, float] = None):
        """Initialize the paper trading engine."""
        try:
            self.status = PaperTradingStatus.STARTING
            
            # Store components
            self.market_data_feeder = market_data_feeder
            self.arbitrage_detector = arbitrage_detector
            self.risk_manager = risk_manager
            
            # Initialize virtual portfolio
            if initial_balances is None:
                initial_balances = {'USDT': 10000.0}  # Default $10k
            
            self.virtual_portfolio = VirtualPortfolio(initial_balances)
            self.metrics.starting_portfolio_value = sum(initial_balances.values())
            
            # Initialize virtual exchanges
            exchanges = self.config.get('exchanges', ['binance', 'backpack'])
            for exchange in exchanges:
                self.virtual_exchanges[exchange] = VirtualExchange(exchange, self.simulation_settings)
                await self.virtual_exchanges[exchange].start()
            
            # Set up market data callbacks
            self.market_data_feeder.add_callback(DataType.TICKER, self._on_ticker_update)
            self.market_data_feeder.add_callback(DataType.ORDERBOOK, self._on_orderbook_update)
            
            # Set up arbitrage detector callbacks if available
            if self.arbitrage_detector:
                self.arbitrage_detector.add_opportunity_callback(self._on_arbitrage_opportunity)
            
            # Start background tasks
            self._tasks.append(asyncio.create_task(self._metrics_updater()))
            self._tasks.append(asyncio.create_task(self._portfolio_monitor()))
            self._tasks.append(asyncio.create_task(self._alert_monitor()))
            
            self.status = PaperTradingStatus.RUNNING
            self.start_time = datetime.now()
            
            logger.info("✅ Paper Trading Engine initialized successfully")
            logger.info(f"   Initial Portfolio: {initial_balances}")
            logger.info(f"   Virtual Exchanges: {list(self.virtual_exchanges.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Paper Trading Engine initialization failed: {e}")
            self.status = PaperTradingStatus.ERROR
            raise
    
    async def shutdown(self):
        """Shutdown the paper trading engine."""
        try:
            self.status = PaperTradingStatus.STOPPING
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Stop virtual exchanges
            for exchange in self.virtual_exchanges.values():
                await exchange.stop()
            
            # Save final metrics
            await self._save_session_summary()
            
            self.status = PaperTradingStatus.STOPPED
            
            logger.info("🔒 Paper Trading Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    async def place_order(self, exchange: str, symbol: str, side: OrderSide, 
                         order_type: OrderType, quantity: float, price: float = None) -> str:
        """Place a virtual order."""
        try:
            if self.status != PaperTradingStatus.RUNNING:
                raise ValueError("Paper trading engine not running")
            
            if exchange not in self.virtual_exchanges:
                raise ValueError(f"Exchange {exchange} not available")
            
            # Create order request
            order_request = OrderRequest(
                id=str(uuid.uuid4()),
                exchange=exchange,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price
            )
            
            # Get current prices for validation
            current_prices = self._get_current_prices()
            
            # Check if order can be placed
            if not self.virtual_portfolio.can_place_order(
                VirtualOrder(
                    id=order_request.id,
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price
                ), current_prices):
                raise ValueError("Insufficient balance for order")
            
            # Place virtual order
            virtual_order = await self.virtual_exchanges[exchange].place_order(order_request)
            
            # Reserve funds
            self.virtual_portfolio.reserve_funds(virtual_order, current_prices)
            
            # Update metrics
            self.metrics.total_orders += 1
            
            logger.info(f"📝 Paper order placed: {virtual_order.id}")
            
            return virtual_order.id
            
        except Exception as e:
            logger.error(f"❌ Order placement error: {e}")
            raise
    
    async def cancel_order(self, exchange: str, order_id: str) -> bool:
        """Cancel a virtual order."""
        try:
            if exchange not in self.virtual_exchanges:
                return False
            
            # Cancel virtual order
            success = await self.virtual_exchanges[exchange].cancel_order(order_id)
            
            if success:
                # Release reserved funds
                order = await self.virtual_exchanges[exchange].get_order_status(order_id)
                if order:
                    current_prices = self._get_current_prices()
                    self.virtual_portfolio.cancel_order_funds(order, current_prices)
                
                self.metrics.cancelled_orders += 1
                logger.info(f"❌ Paper order cancelled: {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Order cancellation error: {e}")
            return False
    
    async def get_order_status(self, exchange: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get virtual order status."""
        try:
            if exchange not in self.virtual_exchanges:
                return None
            
            order = await self.virtual_exchanges[exchange].get_order_status(order_id)
            if order:
                return {
                    'id': order.id,
                    'exchange': order.exchange,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'type': order.type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status.value,
                    'filled_quantity': order.filled_quantity,
                    'average_price': order.average_price,
                    'fees_paid': order.fees_paid,
                    'creation_time': order.creation_time,
                    'fill_time': order.fill_time
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Order status error: {e}")
            return None
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        try:
            if not self.virtual_portfolio:
                return {}
            
            current_prices = self._get_current_prices()
            current_value = self.virtual_portfolio.calculate_portfolio_value(current_prices)
            
            return {
                'balances': {
                    asset: {
                        'total': balance.total,
                        'available': balance.available,
                        'locked': balance.locked
                    }
                    for asset, balance in self.virtual_portfolio.balances.items()
                },
                'total_value_usd': current_value,
                'initial_value_usd': self.metrics.starting_portfolio_value,
                'total_pnl_usd': current_value - self.metrics.starting_portfolio_value,
                'total_pnl_pct': ((current_value - self.metrics.starting_portfolio_value) / 
                                self.metrics.starting_portfolio_value * 100) if self.metrics.starting_portfolio_value > 0 else 0.0,
                'trade_count': len(self.virtual_portfolio.trade_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio status error: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            # Update calculated metrics
            self.metrics.calculate_success_rate()
            
            if self.virtual_portfolio:
                current_prices = self._get_current_prices()
                current_value = self.virtual_portfolio.calculate_portfolio_value(current_prices)
                self.metrics.current_portfolio_value = current_value
                self.metrics.total_pnl = current_value - self.metrics.starting_portfolio_value
            
            return {
                'session_id': self.session_id,
                'status': self.status.value,
                'session_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
                
                # Trading metrics
                'total_orders': self.metrics.total_orders,
                'filled_orders': self.metrics.filled_orders,
                'cancelled_orders': self.metrics.cancelled_orders,
                'rejected_orders': self.metrics.rejected_orders,
                'success_rate_pct': self.metrics.success_rate,
                
                # Portfolio metrics
                'starting_value_usd': self.metrics.starting_portfolio_value,
                'current_value_usd': self.metrics.current_portfolio_value,
                'total_pnl_usd': self.metrics.total_pnl,
                'total_pnl_pct': self.metrics.calculate_pnl_pct(),
                'total_fees_paid': self.metrics.total_fees_paid,
                
                # Arbitrage metrics
                'arbitrage_opportunities': self.metrics.arbitrage_opportunities,
                'successful_arbitrages': self.metrics.successful_arbitrages,
                'arbitrage_pnl': self.metrics.arbitrage_pnl,
                
                # Performance metrics
                'avg_execution_time_ms': self.metrics.avg_execution_time_ms,
                'avg_slippage_bps': self.metrics.avg_slippage_bps,
                'max_drawdown_pct': self.metrics.max_drawdown,
                'sharpe_ratio': self.metrics.sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation error: {e}")
            return {}
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        try:
            if not self.virtual_portfolio:
                return []
            
            return self.virtual_portfolio.trade_history[-limit:]
            
        except Exception as e:
            logger.error(f"❌ Trade history error: {e}")
            return []
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        try:
            return self.alert_manager.alert_history[-limit:]
            
        except Exception as e:
            logger.error(f"❌ Alert history error: {e}")
            return []
    
    async def _on_ticker_update(self, ticker: TickerData):
        """Handle ticker updates."""
        try:
            # Update virtual exchanges with market data
            if ticker.exchange in self.virtual_exchanges:
                self.virtual_exchanges[ticker.exchange].update_market_data(ticker=ticker)
            
        except Exception as e:
            logger.error(f"❌ Ticker update error: {e}")
    
    async def _on_orderbook_update(self, orderbook: OrderBookData):
        """Handle orderbook updates."""
        try:
            # Update virtual exchanges with market data
            if orderbook.exchange in self.virtual_exchanges:
                self.virtual_exchanges[orderbook.exchange].update_market_data(orderbook=orderbook)
            
        except Exception as e:
            logger.error(f"❌ Orderbook update error: {e}")
    
    async def _on_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """Handle arbitrage opportunities."""
        try:
            self.metrics.arbitrage_opportunities += 1
            
            # Log opportunity
            logger.info(f"🎯 Arbitrage opportunity detected: {opportunity.symbol} "
                       f"({opportunity.profit_potential:.4f})")
            
            # Here you could implement automatic execution
            # For now, just log the opportunity
            
        except Exception as e:
            logger.error(f"❌ Arbitrage opportunity handling error: {e}")
    
    async def _metrics_updater(self):
        """Background task to update metrics."""
        try:
            while self.status == PaperTradingStatus.RUNNING:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Update metrics from virtual exchanges
                await self._update_execution_metrics()
                
                # Update portfolio metrics
                if self.virtual_portfolio:
                    current_prices = self._get_current_prices()
                    self.virtual_portfolio.update_portfolio_history(current_prices)
                
                # Store metrics history
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': self.get_metrics()
                })
                
        except Exception as e:
            logger.error(f"❌ Metrics updater error: {e}")
    
    async def _portfolio_monitor(self):
        """Background task to monitor portfolio."""
        try:
            last_pnl_pct = 0.0
            
            while self.status == PaperTradingStatus.RUNNING:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self.virtual_portfolio:
                    current_prices = self._get_current_prices()
                    current_value = self.virtual_portfolio.calculate_portfolio_value(current_prices)
                    current_pnl_pct = ((current_value - self.metrics.starting_portfolio_value) / 
                                     self.metrics.starting_portfolio_value * 100) if self.metrics.starting_portfolio_value > 0 else 0.0
                    
                    # Check for PnL alerts
                    await self.alert_manager.check_pnl_alerts(current_pnl_pct, last_pnl_pct)
                    last_pnl_pct = current_pnl_pct
                
        except Exception as e:
            logger.error(f"❌ Portfolio monitor error: {e}")
    
    async def _alert_monitor(self):
        """Background task to monitor alerts."""
        try:
            while self.status == PaperTradingStatus.RUNNING:
                await asyncio.sleep(60)  # Check every minute
                
                # Check performance alerts
                await self.alert_manager.check_performance_alerts(
                    self.metrics.success_rate,
                    self.metrics.avg_execution_time_ms
                )
                
        except Exception as e:
            logger.error(f"❌ Alert monitor error: {e}")
    
    async def _update_execution_metrics(self):
        """Update execution metrics from virtual exchanges."""
        try:
            total_filled = 0
            total_cancelled = 0
            total_rejected = 0
            execution_times = []
            slippage_values = []
            
            for exchange in self.virtual_exchanges.values():
                for order in exchange.pending_orders.values():
                    if order.status == OrderStatus.FILLED:
                        total_filled += 1
                        if order.fill_time and order.creation_time:
                            exec_time = (order.fill_time - order.creation_time).total_seconds() * 1000
                            execution_times.append(exec_time)
                        slippage_values.append(order.expected_slippage * 10000)  # Convert to bps
                        
                        # Execute trade in portfolio if not already done
                        if order.id not in [trade['order_id'] for trade in self.virtual_portfolio.trade_history]:
                            self.virtual_portfolio.execute_trade(order)
                            self.metrics.total_fees_paid += order.fees_paid
                    
                    elif order.status == OrderStatus.CANCELLED:
                        total_cancelled += 1
                    elif order.status == OrderStatus.REJECTED:
                        total_rejected += 1
            
            # Update metrics
            self.metrics.filled_orders = total_filled
            self.metrics.cancelled_orders = total_cancelled
            self.metrics.rejected_orders = total_rejected
            
            if execution_times:
                self.metrics.avg_execution_time_ms = sum(execution_times) / len(execution_times)
            
            if slippage_values:
                self.metrics.avg_slippage_bps = sum(slippage_values) / len(slippage_values)
            
        except Exception as e:
            logger.error(f"❌ Execution metrics update error: {e}")
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices."""
        try:
            prices = {}
            
            for exchange in self.virtual_exchanges.values():
                for symbol, ticker in exchange.latest_tickers.items():
                    prices[symbol] = ticker.price
            
            return prices
            
        except Exception as e:
            logger.error(f"❌ Current prices error: {e}")
            return {}
    
    def _init_database(self):
        """Initialize SQLite database for paper trading data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trading_sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    initial_portfolio_value REAL NOT NULL,
                    final_portfolio_value REAL,
                    total_pnl REAL,
                    total_orders INTEGER,
                    successful_orders INTEGER,
                    config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fees REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metrics snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("📊 Paper trading database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    async def _save_session_summary(self):
        """Save session summary to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save session summary
            cursor.execute('''
                INSERT OR REPLACE INTO paper_trading_sessions 
                (id, start_time, end_time, initial_portfolio_value, final_portfolio_value, 
                 total_pnl, total_orders, successful_orders, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                self.start_time.isoformat() if self.start_time else None,
                datetime.now().isoformat(),
                self.metrics.starting_portfolio_value,
                self.metrics.current_portfolio_value,
                self.metrics.total_pnl,
                self.metrics.total_orders,
                self.metrics.filled_orders,
                json.dumps(self.config)
            ))
            
            # Save final trades
            if self.virtual_portfolio:
                for trade in self.virtual_portfolio.trade_history:
                    cursor.execute('''
                        INSERT OR IGNORE INTO paper_trades 
                        (session_id, order_id, exchange, symbol, side, quantity, price, fees, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.session_id,
                        trade['order_id'],
                        'virtual',  # We don't track exchange in trade history
                        trade['symbol'],
                        trade['side'],
                        trade['quantity'],
                        trade['price'],
                        trade['fees'],
                        trade['timestamp'].isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Session summary saved: {self.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Session summary save error: {e}")

# Example usage and testing
async def example_usage():
    """Example of how to use the Paper Trading Engine."""
    
    # Configuration
    config = {
        'mode': 'conservative',
        'exchanges': ['binance', 'backpack'],
        'simulation': {
            'base_latency_ms': 100.0,
            'max_latency_ms': 1000.0,
            'base_slippage_bps': 10.0,
            'max_slippage_bps': 100.0,
            'enable_realistic_constraints': True
        },
        'alerts': {
            'enable_alerts': True,
            'pnl_threshold_pct': 5.0,
            'drawdown_threshold_pct': 10.0
        },
        'db_path': 'paper_trading_test.db'
    }
    
    # Initial portfolio
    initial_balances = {
        'USDT': 10000.0,
        'BTC': 0.1,
        'ETH': 2.0
    }
    
    try:
        # Initialize paper trading engine
        engine = PaperTradingEngine(config)
        
        # Mock market data feeder (in real usage, this would be properly initialized)
        class MockMarketDataFeeder:
            def __init__(self):
                self.callbacks = defaultdict(list)
            
            def add_callback(self, data_type, callback):
                self.callbacks[data_type].append(callback)
        
        mock_feeder = MockMarketDataFeeder()
        
        # Initialize engine
        await engine.initialize(
            market_data_feeder=mock_feeder,
            initial_balances=initial_balances
        )
        
        # Add alert callback
        async def on_alert(alert):
            print(f"🚨 ALERT: {alert['title']} - {alert['message']}")
        
        engine.alert_manager.add_alert_callback(on_alert)
        
        print("🚀 Paper Trading Engine started!")
        print("📊 Initial Status:")
        print(json.dumps(engine.get_portfolio_status(), indent=2))
        
        # Simulate some trading
        try:
            # Place a buy order
            order_id = await engine.place_order(
                exchange='binance',
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.0
            )
            print(f"📝 Order placed: {order_id}")
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check order status
            status = await engine.get_order_status('binance', order_id)
            print(f"📊 Order status: {status}")
            
        except Exception as e:
            print(f"❌ Trading error: {e}")
        
        # Run for a short time
        await asyncio.sleep(10)
        
        # Get final metrics
        print("\n📈 Final Metrics:")
        print(json.dumps(engine.get_metrics(), indent=2))
        
        print("\n💰 Final Portfolio:")
        print(json.dumps(engine.get_portfolio_status(), indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'engine' in locals():
            await engine.shutdown()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())