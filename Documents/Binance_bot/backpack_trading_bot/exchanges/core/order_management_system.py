#!/usr/bin/env python3
"""
🎯 MULTI-EXCHANGE ORDER MANAGEMENT SYSTEM (OMS) v1.0.0
Advanced order coordination and execution for cross-exchange arbitrage

Features:
- 🔄 Multi-Exchange Order Coordination
- 📊 Order State Management & Tracking
- 🛡️ Risk Integration & Enforcement
- ⚡ Arbitrage Execution Logic
- 🚨 Error Handling & Partial Fills
- 🎯 Order Matching & Delta Neutrality
- 🛑 Emergency Controls Integration
- 📈 Performance Monitoring
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque

# Import exchange adapters
from ..exchanges.binance_adapter import BinanceAdapter, BinanceOrderSide, BinanceOrderType
from ..exchanges.backpack_adapter import BackpackAdapter, BackpackOrderSide, BackpackOrderType
from ..risk_management.integrated_risk_manager import IntegratedRiskManager, RiskLevel
from ..strategies.arbitrage_detector import ArbitrageOpportunity, MarketDirection

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class ExecutionStrategy(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"
    SMART_ROUTING = "smart_routing"

class PositionType(Enum):
    ARBITRAGE = "arbitrage"
    HEDGING = "hedging"
    STANDALONE = "standalone"

@dataclass
class OrderRequest:
    """Order request structure for OMS."""
    id: str
    exchange: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    position_type: PositionType = PositionType.STANDALONE
    parent_strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Risk parameters
    max_slippage: float = 0.005  # 0.5% max slippage
    timeout_seconds: int = 30
    retry_count: int = 3

@dataclass
class OrderExecution:
    """Order execution tracking."""
    order_id: str
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    remaining_quantity: float = 0.0
    fees_paid: float = 0.0
    error_message: Optional[str] = None
    creation_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    fill_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_fill(self, fill_qty: float, fill_price: float, fee: float = 0.0):
        """Update order with new fill information."""
        self.fill_history.append({
            'quantity': fill_qty,
            'price': fill_price,
            'fee': fee,
            'timestamp': datetime.now()
        })
        
        self.filled_quantity += fill_qty
        self.remaining_quantity -= fill_qty
        self.fees_paid += fee
        
        # Update average price
        if self.filled_quantity > 0:
            total_value = sum(fill['quantity'] * fill['price'] for fill in self.fill_history)
            self.average_price = total_value / self.filled_quantity
        
        self.last_update = datetime.now()

@dataclass
class ArbitrageExecution:
    """Arbitrage execution group tracking."""
    id: str
    strategy_id: str
    symbol: str
    direction: MarketDirection
    target_quantity: float
    buy_order: OrderRequest
    sell_order: OrderRequest
    buy_execution: OrderExecution
    sell_execution: OrderExecution
    expected_profit: float
    actual_profit: float = 0.0
    status: str = "active"
    creation_time: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    
    def calculate_actual_profit(self) -> float:
        """Calculate actual profit from executed orders."""
        if (self.buy_execution.status == OrderStatus.FILLED and 
            self.sell_execution.status == OrderStatus.FILLED):
            
            buy_cost = self.buy_execution.filled_quantity * self.buy_execution.average_price
            sell_proceeds = self.sell_execution.filled_quantity * self.sell_execution.average_price
            total_fees = self.buy_execution.fees_paid + self.sell_execution.fees_paid
            
            self.actual_profit = sell_proceeds - buy_cost - total_fees
        
        return self.actual_profit

@dataclass
class OMSMetrics:
    """OMS performance metrics."""
    total_orders_submitted: int = 0
    total_orders_filled: int = 0
    total_orders_cancelled: int = 0
    total_orders_failed: int = 0
    
    successful_arbitrages: int = 0
    failed_arbitrages: int = 0
    total_arbitrage_profit: float = 0.0
    
    average_execution_time: float = 0.0
    average_slippage: float = 0.0
    
    last_reset: datetime = field(default_factory=datetime.now)
    
    def success_rate(self) -> float:
        """Calculate order success rate."""
        total = self.total_orders_submitted
        return (self.total_orders_filled / total * 100) if total > 0 else 0.0

class OrderManagementSystem:
    """
    Comprehensive Order Management System for multi-exchange coordination.
    Handles order lifecycle, risk integration, and arbitrage execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Exchange adapters
        self.exchanges = {}
        self.risk_manager = None
        
        # Order tracking
        self.orders: Dict[str, OrderRequest] = {}
        self.executions: Dict[str, OrderExecution] = {}
        self.arbitrage_executions: Dict[str, ArbitrageExecution] = {}
        
        # Order queues by priority
        self.order_queues = {
            'high': deque(),
            'medium': deque(),
            'low': deque()
        }
        
        # Performance tracking
        self.metrics = OMSMetrics()
        self.execution_times = deque(maxlen=100)
        self.slippage_history = deque(maxlen=100)
        
        # State management
        self.is_running = False
        self.emergency_stop = False
        self.processing_orders = False
        
        # Configuration
        self.max_concurrent_orders = config.get('max_concurrent_orders', 10)
        self.order_timeout = config.get('order_timeout_seconds', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.position_matching_tolerance = config.get('position_matching_tolerance', 0.01)
        
        # Callbacks
        self.order_update_callbacks: List[Callable] = []
        self.arbitrage_completion_callbacks: List[Callable] = []
        
        logger.info("🎯 Order Management System v1.0.0 initialized")
        logger.info(f"   Max Concurrent Orders: {self.max_concurrent_orders}")
        logger.info(f"   Order Timeout: {self.order_timeout}s")
        logger.info(f"   Retry Attempts: {self.retry_attempts}")
    
    async def initialize(self, binance_adapter: BinanceAdapter, backpack_adapter: BackpackAdapter, 
                        risk_manager: IntegratedRiskManager):
        """Initialize OMS with exchange adapters and risk manager."""
        try:
            self.exchanges['binance'] = binance_adapter
            self.exchanges['backpack'] = backpack_adapter
            self.risk_manager = risk_manager
            
            # Start background tasks
            self.is_running = True
            
            # Start order processing loop
            asyncio.create_task(self._order_processing_loop())
            
            # Start monitoring tasks
            asyncio.create_task(self._order_monitoring_loop())
            asyncio.create_task(self._arbitrage_monitoring_loop())
            
            logger.info("✅ Order Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ OMS initialization failed: {e}")
            raise
    
    async def submit_order(self, order_request: OrderRequest, priority: str = "medium") -> str:
        """Submit a single order to the OMS."""
        try:
            # Validate order
            if not await self._validate_order(order_request):
                raise ValueError("Order validation failed")
            
            # Check risk limits
            if not await self._check_risk_limits(order_request):
                raise ValueError("Risk limits exceeded")
            
            # Generate execution tracking
            execution = OrderExecution(
                order_id=order_request.id,
                remaining_quantity=order_request.quantity
            )
            
            # Store order and execution
            self.orders[order_request.id] = order_request
            self.executions[order_request.id] = execution
            
            # Queue for execution
            self.order_queues[priority].append(order_request.id)
            
            logger.info(f"📝 Order submitted: {order_request.id} ({order_request.symbol} {order_request.side.value} {order_request.quantity})")
            
            return order_request.id
            
        except Exception as e:
            logger.error(f"❌ Order submission failed: {e}")
            raise
    
    async def submit_arbitrage_orders(self, opportunity: ArbitrageOpportunity, 
                                    quantity: float, execution_strategy: ExecutionStrategy = ExecutionStrategy.SIMULTANEOUS) -> str:
        """Submit coordinated arbitrage orders."""
        try:
            arbitrage_id = str(uuid.uuid4())
            
            # Create buy and sell orders based on opportunity direction
            if opportunity.direction == MarketDirection.BUY_BINANCE_SELL_BACKPACK:
                buy_exchange = "binance"
                sell_exchange = "backpack"
                buy_price = opportunity.binance_price
                sell_price = opportunity.backpack_price
            else:
                buy_exchange = "backpack"
                sell_exchange = "binance"
                buy_price = opportunity.backpack_price
                sell_price = opportunity.binance_price
            
            # Create order requests
            buy_order = OrderRequest(
                id=str(uuid.uuid4()),
                exchange=buy_exchange,
                symbol=opportunity.symbol,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=quantity,
                price=buy_price,
                position_type=PositionType.ARBITRAGE,
                parent_strategy_id=arbitrage_id,
                timeout_seconds=int(opportunity.estimated_execution_time) + 10
            )
            
            sell_order = OrderRequest(
                id=str(uuid.uuid4()),
                exchange=sell_exchange,
                symbol=opportunity.symbol,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                quantity=quantity,
                price=sell_price,
                position_type=PositionType.ARBITRAGE,
                parent_strategy_id=arbitrage_id,
                timeout_seconds=int(opportunity.estimated_execution_time) + 10
            )
            
            # Create arbitrage execution tracker
            arbitrage_exec = ArbitrageExecution(
                id=arbitrage_id,
                strategy_id=f"arbitrage_{opportunity.type.value}",
                symbol=opportunity.symbol,
                direction=opportunity.direction,
                target_quantity=quantity,
                buy_order=buy_order,
                sell_order=sell_order,
                buy_execution=OrderExecution(order_id=buy_order.id, remaining_quantity=quantity),
                sell_execution=OrderExecution(order_id=sell_order.id, remaining_quantity=quantity),
                expected_profit=opportunity.profit_potential * quantity
            )
            
            self.arbitrage_executions[arbitrage_id] = arbitrage_exec
            
            # Submit orders based on execution strategy
            if execution_strategy == ExecutionStrategy.SIMULTANEOUS:
                await self._submit_simultaneous_orders(buy_order, sell_order)
            elif execution_strategy == ExecutionStrategy.SEQUENTIAL:
                await self._submit_sequential_orders(buy_order, sell_order)
            else:  # Smart routing
                await self._submit_smart_routed_orders(buy_order, sell_order, opportunity)
            
            logger.info(f"🎯 Arbitrage orders submitted: {arbitrage_id}")
            logger.info(f"   Symbol: {opportunity.symbol}")
            logger.info(f"   Direction: {opportunity.direction.value}")
            logger.info(f"   Quantity: {quantity}")
            logger.info(f"   Expected Profit: ${opportunity.profit_potential * quantity:.2f}")
            
            return arbitrage_id
            
        except Exception as e:
            logger.error(f"❌ Arbitrage order submission failed: {e}")
            raise
    
    async def _submit_simultaneous_orders(self, buy_order: OrderRequest, sell_order: OrderRequest):
        """Submit both orders simultaneously."""
        try:
            # Submit both orders at the same time
            buy_task = asyncio.create_task(self.submit_order(buy_order, "high"))
            sell_task = asyncio.create_task(self.submit_order(sell_order, "high"))
            
            await asyncio.gather(buy_task, sell_task)
            
        except Exception as e:
            logger.error(f"❌ Simultaneous order submission failed: {e}")
            raise
    
    async def _submit_sequential_orders(self, buy_order: OrderRequest, sell_order: OrderRequest):
        """Submit orders sequentially (buy first, then sell)."""
        try:
            # Submit buy order first
            await self.submit_order(buy_order, "high")
            
            # Wait a moment for the buy order to be processed
            await asyncio.sleep(0.1)
            
            # Submit sell order
            await self.submit_order(sell_order, "high")
            
        except Exception as e:
            logger.error(f"❌ Sequential order submission failed: {e}")
            raise
    
    async def _submit_smart_routed_orders(self, buy_order: OrderRequest, sell_order: OrderRequest, 
                                        opportunity: ArbitrageOpportunity):
        """Submit orders with smart routing based on market conditions."""
        try:
            # Determine which order to submit first based on liquidity and volatility
            if opportunity.liquidity_score > 0.8 and opportunity.volatility_risk < 0.3:
                # High liquidity, low volatility - submit simultaneously
                await self._submit_simultaneous_orders(buy_order, sell_order)
            else:
                # Submit the safer order first
                if opportunity.direction == MarketDirection.BUY_BINANCE_SELL_BACKPACK:
                    # Submit buy order first (typically safer)
                    await self._submit_sequential_orders(buy_order, sell_order)
                else:
                    # Submit sell order first
                    await self._submit_sequential_orders(sell_order, buy_order)
                    
        except Exception as e:
            logger.error(f"❌ Smart routed order submission failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel a specific order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            execution = self.executions[order_id]
            
            if execution.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Order {order_id} already {execution.status.value}")
                return False
            
            # Cancel on exchange
            success = await self._cancel_exchange_order(order, execution)
            
            if success:
                execution.status = OrderStatus.CANCELLED
                execution.error_message = reason
                execution.last_update = datetime.now()
                
                self.metrics.total_orders_cancelled += 1
                
                logger.info(f"✅ Order cancelled: {order_id} - {reason}")
                
                # Notify callbacks
                await self._notify_order_update(order, execution)
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            return False
    
    async def cancel_arbitrage(self, arbitrage_id: str, reason: str = "User request") -> bool:
        """Cancel all orders in an arbitrage execution."""
        try:
            if arbitrage_id not in self.arbitrage_executions:
                logger.warning(f"Arbitrage {arbitrage_id} not found")
                return False
            
            arbitrage = self.arbitrage_executions[arbitrage_id]
            
            # Cancel both orders
            buy_cancelled = await self.cancel_order(arbitrage.buy_order.id, f"Arbitrage cancelled: {reason}")
            sell_cancelled = await self.cancel_order(arbitrage.sell_order.id, f"Arbitrage cancelled: {reason}")
            
            if buy_cancelled or sell_cancelled:
                arbitrage.status = "cancelled"
                arbitrage.completion_time = datetime.now()
                
                logger.info(f"✅ Arbitrage cancelled: {arbitrage_id} - {reason}")
                
                # Notify callbacks
                await self._notify_arbitrage_completion(arbitrage)
                
            return buy_cancelled and sell_cancelled
            
        except Exception as e:
            logger.error(f"❌ Arbitrage cancellation failed: {e}")
            return False
    
    async def emergency_stop_all(self, reason: str = "Emergency stop"):
        """Emergency stop all trading activities."""
        try:
            logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
            
            self.emergency_stop = True
            
            # Cancel all pending orders
            cancel_tasks = []
            for order_id in list(self.orders.keys()):
                execution = self.executions.get(order_id)
                if execution and execution.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                    cancel_tasks.append(self.cancel_order(order_id, f"Emergency stop: {reason}"))
            
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
            
            # Cancel all active arbitrages
            arbitrage_tasks = []
            for arbitrage_id in list(self.arbitrage_executions.keys()):
                arbitrage = self.arbitrage_executions[arbitrage_id]
                if arbitrage.status == "active":
                    arbitrage_tasks.append(self.cancel_arbitrage(arbitrage_id, f"Emergency stop: {reason}"))
            
            if arbitrage_tasks:
                await asyncio.gather(*arbitrage_tasks, return_exceptions=True)
            
            logger.critical("🛑 Emergency stop completed - all orders cancelled")
            
        except Exception as e:
            logger.error(f"❌ Emergency stop failed: {e}")
    
    async def _order_processing_loop(self):
        """Main order processing loop."""
        try:
            while self.is_running:
                if self.emergency_stop:
                    await asyncio.sleep(1)
                    continue
                
                # Process orders from high priority queue first
                for priority in ['high', 'medium', 'low']:
                    if self.order_queues[priority] and len(self._get_active_orders()) < self.max_concurrent_orders:
                        order_id = self.order_queues[priority].popleft()
                        asyncio.create_task(self._process_order(order_id))
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        except Exception as e:
            logger.error(f"❌ Order processing loop error: {e}")
    
    async def _process_order(self, order_id: str):
        """Process a single order."""
        try:
            order = self.orders[order_id]
            execution = self.executions[order_id]
            
            start_time = time.time()
            
            # Update status
            execution.status = OrderStatus.SUBMITTED
            execution.last_update = datetime.now()
            
            # Submit to exchange
            success = await self._submit_to_exchange(order, execution)
            
            if success:
                # Monitor order execution
                await self._monitor_order_execution(order, execution)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                # Update metrics
                if execution.status == OrderStatus.FILLED:
                    self.metrics.total_orders_filled += 1
                    
                    # Calculate slippage
                    if order.price and execution.average_price:
                        slippage = abs(execution.average_price - order.price) / order.price
                        self.slippage_history.append(slippage)
                        
            else:
                execution.status = OrderStatus.FAILED
                self.metrics.total_orders_failed += 1
            
            # Notify callbacks
            await self._notify_order_update(order, execution)
            
        except Exception as e:
            logger.error(f"❌ Order processing error for {order_id}: {e}")
            execution = self.executions.get(order_id)
            if execution:
                execution.status = OrderStatus.FAILED
                execution.error_message = str(e)
                execution.last_update = datetime.now()
    
    async def _submit_to_exchange(self, order: OrderRequest, execution: OrderExecution) -> bool:
        """Submit order to the appropriate exchange."""
        try:
            exchange_adapter = self.exchanges.get(order.exchange)
            if not exchange_adapter:
                raise ValueError(f"Exchange {order.exchange} not available")
            
            # Convert to exchange-specific format
            if order.exchange == "binance":
                binance_side = BinanceOrderSide.BUY if order.side == OrderSide.BUY else BinanceOrderSide.SELL
                binance_type = BinanceOrderType.LIMIT if order.type == OrderType.LIMIT else BinanceOrderType.MARKET
                
                exchange_order = await exchange_adapter.place_order(
                    symbol=order.symbol,
                    side=binance_side,
                    order_type=binance_type,
                    quantity=order.quantity,
                    price=order.price,
                    time_in_force=order.time_in_force
                )
                
                execution.exchange_order_id = exchange_order.id
                
            elif order.exchange == "backpack":
                backpack_side = BackpackOrderSide.BUY if order.side == OrderSide.BUY else BackpackOrderSide.SELL
                backpack_type = BackpackOrderType.LIMIT if order.type == OrderType.LIMIT else BackpackOrderType.MARKET
                
                exchange_order = await exchange_adapter.place_order(
                    symbol=order.symbol,
                    side=backpack_side,
                    order_type=backpack_type,
                    quantity=order.quantity,
                    price=order.price,
                    time_in_force=order.time_in_force
                )
                
                execution.exchange_order_id = exchange_order.id
            
            self.metrics.total_orders_submitted += 1
            
            logger.info(f"📤 Order submitted to {order.exchange}: {order.id} -> {execution.exchange_order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Exchange submission failed: {e}")
            execution.error_message = str(e)
            return False
    
    async def _monitor_order_execution(self, order: OrderRequest, execution: OrderExecution):
        """Monitor order execution until completion."""
        try:
            start_time = time.time()
            
            while (execution.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED] and
                   time.time() - start_time < order.timeout_seconds):
                
                # Check order status on exchange
                await self._update_order_status(order, execution)
                
                if execution.status == OrderStatus.FILLED:
                    break
                
                await asyncio.sleep(1)  # Check every second
            
            # Handle timeout
            if execution.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                logger.warning(f"⏰ Order {order.id} timed out, attempting cancellation")
                await self.cancel_order(order.id, "Order timeout")
                
        except Exception as e:
            logger.error(f"❌ Order monitoring error: {e}")
    
    async def _update_order_status(self, order: OrderRequest, execution: OrderExecution):
        """Update order status from exchange."""
        try:
            exchange_adapter = self.exchanges.get(order.exchange)
            if not exchange_adapter or not execution.exchange_order_id:
                return
            
            # Get order status from exchange
            exchange_order = await exchange_adapter.get_order_status(
                symbol=order.symbol,
                order_id=execution.exchange_order_id
            )
            
            # Update execution status
            old_filled = execution.filled_quantity
            execution.filled_quantity = exchange_order.filled
            execution.remaining_quantity = order.quantity - execution.filled_quantity
            
            # Check for new fills
            if execution.filled_quantity > old_filled:
                new_fill = execution.filled_quantity - old_filled
                execution.update_fill(new_fill, exchange_order.price)
                
                logger.info(f"📊 Order {order.id} fill: {new_fill} @ ${exchange_order.price:.2f}")
            
            # Update status
            if exchange_order.status == "FILLED":
                execution.status = OrderStatus.FILLED
            elif exchange_order.status == "PARTIALLY_FILLED":
                execution.status = OrderStatus.PARTIALLY_FILLED
            elif exchange_order.status == "CANCELLED":
                execution.status = OrderStatus.CANCELLED
            elif exchange_order.status == "REJECTED":
                execution.status = OrderStatus.REJECTED
            
            execution.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Order status update error: {e}")
    
    async def _cancel_exchange_order(self, order: OrderRequest, execution: OrderExecution) -> bool:
        """Cancel order on exchange."""
        try:
            exchange_adapter = self.exchanges.get(order.exchange)
            if not exchange_adapter or not execution.exchange_order_id:
                return False
            
            success = await exchange_adapter.cancel_order(
                symbol=order.symbol,
                order_id=execution.exchange_order_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Exchange cancellation error: {e}")
            return False
    
    async def _arbitrage_monitoring_loop(self):
        """Monitor arbitrage executions for completion."""
        try:
            while self.is_running:
                for arbitrage_id, arbitrage in list(self.arbitrage_executions.items()):
                    if arbitrage.status == "active":
                        await self._check_arbitrage_completion(arbitrage)
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Arbitrage monitoring error: {e}")
    
    async def _check_arbitrage_completion(self, arbitrage: ArbitrageExecution):
        """Check if an arbitrage execution is complete."""
        try:
            buy_exec = self.executions.get(arbitrage.buy_order.id)
            sell_exec = self.executions.get(arbitrage.sell_order.id)
            
            if not buy_exec or not sell_exec:
                return
            
            # Update arbitrage execution references
            arbitrage.buy_execution = buy_exec
            arbitrage.sell_execution = sell_exec
            
            # Check completion conditions
            both_filled = (buy_exec.status == OrderStatus.FILLED and 
                          sell_exec.status == OrderStatus.FILLED)
            
            either_failed = (buy_exec.status in [OrderStatus.FAILED, OrderStatus.REJECTED, OrderStatus.CANCELLED] or
                            sell_exec.status in [OrderStatus.FAILED, OrderStatus.REJECTED, OrderStatus.CANCELLED])
            
            if both_filled:
                arbitrage.status = "completed"
                arbitrage.completion_time = datetime.now()
                arbitrage.calculate_actual_profit()
                
                self.metrics.successful_arbitrages += 1
                self.metrics.total_arbitrage_profit += arbitrage.actual_profit
                
                logger.info(f"✅ Arbitrage completed: {arbitrage.id}")
                logger.info(f"   Expected Profit: ${arbitrage.expected_profit:.2f}")
                logger.info(f"   Actual Profit: ${arbitrage.actual_profit:.2f}")
                
                await self._notify_arbitrage_completion(arbitrage)
                
            elif either_failed:
                arbitrage.status = "failed"
                arbitrage.completion_time = datetime.now()
                
                self.metrics.failed_arbitrages += 1
                
                logger.warning(f"❌ Arbitrage failed: {arbitrage.id}")
                
                # Handle partial fills - may need hedging
                await self._handle_partial_arbitrage(arbitrage)
                
                await self._notify_arbitrage_completion(arbitrage)
                
        except Exception as e:
            logger.error(f"❌ Arbitrage completion check error: {e}")
    
    async def _handle_partial_arbitrage(self, arbitrage: ArbitrageExecution):
        """Handle partial arbitrage fills - may require hedging."""
        try:
            buy_exec = arbitrage.buy_execution
            sell_exec = arbitrage.sell_execution
            
            # Calculate position imbalance
            net_position = buy_exec.filled_quantity - sell_exec.filled_quantity
            
            if abs(net_position) > self.position_matching_tolerance:
                logger.warning(f"⚠️ Position imbalance detected: {net_position} {arbitrage.symbol}")
                
                # TODO: Implement hedging logic
                # This could involve:
                # 1. Placing offsetting orders
                # 2. Notifying risk management
                # 3. Manual intervention alerts
                
                # For now, just log the issue
                logger.warning(f"   Buy filled: {buy_exec.filled_quantity}")
                logger.warning(f"   Sell filled: {sell_exec.filled_quantity}")
                logger.warning(f"   Net position: {net_position}")
                
        except Exception as e:
            logger.error(f"❌ Partial arbitrage handling error: {e}")
    
    async def _order_monitoring_loop(self):
        """Monitor order health and timeouts."""
        try:
            while self.is_running:
                current_time = datetime.now()
                
                for order_id, execution in list(self.executions.items()):
                    if execution.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                        # Check for timeouts
                        time_elapsed = (current_time - execution.creation_time).total_seconds()
                        order = self.orders.get(order_id)
                        
                        if order and time_elapsed > order.timeout_seconds:
                            logger.warning(f"⏰ Order {order_id} timed out after {time_elapsed:.1f}s")
                            await self.cancel_order(order_id, "Order timeout")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"❌ Order monitoring error: {e}")
    
    async def _validate_order(self, order: OrderRequest) -> bool:
        """Validate order parameters."""
        try:
            # Basic validation
            if order.quantity <= 0:
                logger.error(f"Invalid quantity: {order.quantity}")
                return False
            
            if order.exchange not in self.exchanges:
                logger.error(f"Invalid exchange: {order.exchange}")
                return False
            
            if order.type == OrderType.LIMIT and (not order.price or order.price <= 0):
                logger.error(f"Invalid price for limit order: {order.price}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order validation error: {e}")
            return False
    
    async def _check_risk_limits(self, order: OrderRequest) -> bool:
        """Check risk limits before order submission."""
        try:
            if not self.risk_manager:
                return True
            
            # Check if trading is allowed
            if not self.risk_manager.is_trading_allowed():
                logger.warning("Trading is not allowed by risk manager")
                return False
            
            # Check position limits
            position_size = order.quantity * (order.price or 0)
            allowed, reason = self.risk_manager.can_open_position(
                order.exchange, position_size, "spot"
            )
            
            if not allowed:
                logger.warning(f"Position not allowed: {reason}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk limit check error: {e}")
            return False
    
    def _get_active_orders(self) -> List[OrderExecution]:
        """Get list of active orders."""
        return [exec for exec in self.executions.values() 
                if exec.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
    
    async def _notify_order_update(self, order: OrderRequest, execution: OrderExecution):
        """Notify callbacks of order updates."""
        try:
            for callback in self.order_update_callbacks:
                await callback(order, execution)
        except Exception as e:
            logger.error(f"❌ Order update notification error: {e}")
    
    async def _notify_arbitrage_completion(self, arbitrage: ArbitrageExecution):
        """Notify callbacks of arbitrage completion."""
        try:
            for callback in self.arbitrage_completion_callbacks:
                await callback(arbitrage)
        except Exception as e:
            logger.error(f"❌ Arbitrage completion notification error: {e}")
    
    def add_order_update_callback(self, callback: Callable):
        """Add callback for order updates."""
        self.order_update_callbacks.append(callback)
    
    def add_arbitrage_completion_callback(self, callback: Callable):
        """Add callback for arbitrage completion."""
        self.arbitrage_completion_callbacks.append(callback)
    
    def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order execution status."""
        return self.executions.get(order_id)
    
    def get_arbitrage_status(self, arbitrage_id: str) -> Optional[ArbitrageExecution]:
        """Get arbitrage execution status."""
        return self.arbitrage_executions.get(arbitrage_id)
    
    def get_metrics(self) -> OMSMetrics:
        """Get OMS performance metrics."""
        # Update calculated metrics
        if self.execution_times:
            self.metrics.average_execution_time = sum(self.execution_times) / len(self.execution_times)
        
        if self.slippage_history:
            self.metrics.average_slippage = sum(self.slippage_history) / len(self.slippage_history)
        
        return self.metrics
    
    def get_portfolio_delta(self) -> Dict[str, float]:
        """Calculate current portfolio delta from active orders."""
        try:
            delta_by_symbol = defaultdict(float)
            
            for execution in self.executions.values():
                if execution.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]:
                    order = self.orders.get(execution.order_id)
                    if order:
                        delta = execution.filled_quantity
                        if order.side == OrderSide.SELL:
                            delta *= -1
                        delta_by_symbol[order.symbol] += delta
            
            return dict(delta_by_symbol)
            
        except Exception as e:
            logger.error(f"❌ Portfolio delta calculation error: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the OMS gracefully."""
        try:
            logger.info("🔄 Shutting down Order Management System...")
            
            self.is_running = False
            
            # Cancel all active orders
            cancel_tasks = []
            for order_id, execution in self.executions.items():
                if execution.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                    cancel_tasks.append(self.cancel_order(order_id, "System shutdown"))
            
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
            
            logger.info("✅ Order Management System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ OMS shutdown error: {e}")

# Example usage and testing
async def example_usage():
    """Example of how to use the Order Management System."""
    
    # Configuration
    config = {
        'max_concurrent_orders': 10,
        'order_timeout_seconds': 30,
        'retry_attempts': 3,
        'position_matching_tolerance': 0.01
    }
    
    # Initialize OMS
    oms = OrderManagementSystem(config)
    
    # Mock exchange adapters and risk manager for testing
    # In real usage, these would be properly initialized
    print("🎯 Order Management System example initialized")
    print(f"   Max Concurrent Orders: {config['max_concurrent_orders']}")
    print(f"   Order Timeout: {config['order_timeout_seconds']}s")
    
    # Example order request
    order_request = OrderRequest(
        id=str(uuid.uuid4()),
        exchange="binance",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=0.001,
        price=50000.0,
        position_type=PositionType.STANDALONE
    )
    
    print(f"\n📝 Example order created:")
    print(f"   ID: {order_request.id}")
    print(f"   Exchange: {order_request.exchange}")
    print(f"   Symbol: {order_request.symbol}")
    print(f"   Side: {order_request.side.value}")
    print(f"   Quantity: {order_request.quantity}")
    print(f"   Price: ${order_request.price}")
    
    # Example metrics
    metrics = oms.get_metrics()
    print(f"\n📊 OMS Metrics:")
    print(f"   Orders Submitted: {metrics.total_orders_submitted}")
    print(f"   Orders Filled: {metrics.total_orders_filled}")
    print(f"   Success Rate: {metrics.success_rate():.1f}%")
    print(f"   Arbitrages: {metrics.successful_arbitrages}")
    print(f"   Total Profit: ${metrics.total_arbitrage_profit:.2f}")

if __name__ == "__main__":
    asyncio.run(example_usage())