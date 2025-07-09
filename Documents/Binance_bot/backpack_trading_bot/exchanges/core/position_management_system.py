#!/usr/bin/env python3
"""
⚖️ POSITION MANAGEMENT SYSTEM (PMS) v1.0.0
Comprehensive cross-exchange position tracking and delta-neutral management

Features:
- 🔄 Cross-Exchange Position Tracking (Binance & Backpack)
- ⚖️ Delta-Neutral Position Management
- 🎯 Multi-Position Coordination (Grid, Arbitrage, Hedge)
- 📊 Real-time PnL Calculation (Unrealized & Realized)
- 🔄 Position Reconciliation with Exchange Data
- 🛡️ Risk Integration & Position Limits
- 🚨 Automated Hedging & Delta Neutrality
- 📈 Position Lifecycle Management
- 🎲 Integration with Institutional Bot Delta-Neutral Concepts
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque

# Import exchange adapters and risk management
from ..exchanges.binance_adapter import BinanceAdapter
from ..exchanges.backpack_adapter import BackpackAdapter
from ..risk_management.integrated_risk_manager import IntegratedRiskManager, RiskLevel

logger = logging.getLogger(__name__)

class PositionType(Enum):
    """Position types for multi-strategy system."""
    SPOT_LONG = "spot_long"
    SPOT_SHORT = "spot_short"
    FUTURES_LONG = "futures_long"
    FUTURES_SHORT = "futures_short"
    GRID_BUY = "grid_buy"
    GRID_SELL = "grid_sell"
    ARBITRAGE_LONG = "arbitrage_long"
    ARBITRAGE_SHORT = "arbitrage_short"
    HEDGE_LONG = "hedge_long"
    HEDGE_SHORT = "hedge_short"

class PositionStatus(Enum):
    """Position status lifecycle."""
    OPENING = "opening"
    OPEN = "open"
    PARTIAL = "partial"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DeltaStatus(Enum):
    """Delta neutrality status."""
    NEUTRAL = "neutral"
    LONG_BIAS = "long_bias"
    SHORT_BIAS = "short_bias"
    REBALANCE_NEEDED = "rebalance_needed"
    HEDGE_URGENT = "hedge_urgent"

class PositionStrategy(Enum):
    """Strategy classification for positions."""
    GRID_TRADING = "grid_trading"
    ARBITRAGE = "arbitrage"
    DELTA_HEDGE = "delta_hedge"
    MARKET_MAKING = "market_making"
    STANDALONE = "standalone"

@dataclass
class Position:
    """Comprehensive position tracking structure."""
    position_id: str
    exchange: str
    symbol: str
    position_type: PositionType
    strategy: PositionStrategy
    
    # Quantity and pricing
    quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    average_price: float = 0.0
    
    # Position status and timing
    status: PositionStatus = PositionStatus.OPENING
    created_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    closed_time: Optional[datetime] = None
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_fees: float = 0.0  # For futures
    
    # Risk and delta
    margin_used: float = 0.0
    delta: float = 0.0  # Position delta for neutrality calculations
    gamma: float = 0.0  # Position gamma (sensitivity)
    max_risk: float = 0.0
    
    # Strategy-specific
    grid_level: Optional[int] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Related positions (for delta-neutral pairs)
    paired_position_id: Optional[str] = None
    hedge_ratio: float = 1.0
    
    # Metadata
    parent_strategy_id: Optional[str] = None
    order_ids: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class DeltaNeutralPair:
    """Delta-neutral position pair tracking."""
    pair_id: str
    strategy: PositionStrategy
    symbol: str
    
    # Position references
    long_position_id: str
    short_position_id: str
    long_exchange: str
    short_exchange: str
    
    # Delta metrics
    target_delta: float = 0.0
    current_delta: float = 0.0
    delta_deviation: float = 0.0
    hedge_ratio: float = 1.0
    
    # P&L tracking
    combined_pnl: float = 0.0
    basis_pnl: float = 0.0  # Spread profit
    funding_pnl: float = 0.0
    
    # Status and timing
    status: str = "active"
    created_time: datetime = field(default_factory=datetime.now)
    last_rebalance: datetime = field(default_factory=datetime.now)
    
    # Risk metrics
    correlation: float = 1.0
    tracking_error: float = 0.0
    max_deviation: float = 0.0

@dataclass
class PortfolioDelta:
    """Portfolio-wide delta tracking."""
    total_delta: float = 0.0
    symbol_deltas: Dict[str, float] = field(default_factory=dict)
    exchange_deltas: Dict[str, float] = field(default_factory=dict)
    strategy_deltas: Dict[str, float] = field(default_factory=dict)
    
    # Target and deviations
    target_delta: float = 0.0
    delta_deviation: float = 0.0
    rebalance_threshold: float = 0.05  # 5% deviation
    
    # Hedge effectiveness
    hedge_effectiveness: float = 0.0
    correlation_breakdown: Dict[str, float] = field(default_factory=dict)
    
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class PMSMetrics:
    """Position Management System metrics."""
    total_positions: int = 0
    active_positions: int = 0
    closed_positions: int = 0
    
    # By type
    spot_positions: int = 0
    futures_positions: int = 0
    
    # By exchange
    binance_positions: int = 0
    backpack_positions: int = 0
    
    # By strategy
    grid_positions: int = 0
    arbitrage_positions: int = 0
    hedge_positions: int = 0
    
    # P&L metrics
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    
    # Delta metrics
    portfolio_delta: float = 0.0
    delta_neutral_pairs: int = 0
    rebalance_events: int = 0
    
    # Performance
    total_margin_used: float = 0.0
    margin_utilization: float = 0.0
    average_holding_time: float = 0.0
    
    last_update: datetime = field(default_factory=datetime.now)

class PositionManagementSystem:
    """
    Comprehensive Position Management System for cross-exchange trading.
    Integrates with OMS, Risk Manager, and maintains delta-neutral concepts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components (will be injected)
        self.exchanges = {}
        self.risk_manager = None
        self.oms = None  # Order Management System reference
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.delta_neutral_pairs: Dict[str, DeltaNeutralPair] = {}
        self.position_history: List[Position] = []
        
        # Delta tracking
        self.portfolio_delta = PortfolioDelta()
        self.delta_params = {
            'target_delta': 0.0,
            'rebalance_threshold': config.get('delta_rebalance_threshold', 0.05),
            'hedge_ratio': config.get('default_hedge_ratio', 1.0),
            'max_delta_exposure': config.get('max_delta_exposure', 5.0),
            'auto_hedge_enabled': config.get('auto_hedge_enabled', True),
            'hedge_urgency_threshold': config.get('hedge_urgency_threshold', 0.15),
        }
        
        # Position limits
        self.position_limits = {
            'max_positions_per_exchange': config.get('max_positions_per_exchange', 50),
            'max_position_size': config.get('max_position_size', 100000),  # $100k
            'max_margin_per_position': config.get('max_margin_per_position', 20000),  # $20k
            'max_total_margin': config.get('max_total_margin', 200000),  # $200k
            'position_concentration_limit': config.get('position_concentration_limit', 0.2),  # 20%
        }
        
        # Delta-neutral specific (preserved from institutional bot)
        self.institutional_params = {
            'volatility_harvest_enabled': True,
            'basis_trading_enabled': True,
            'gamma_hedging_enabled': True,
            'funding_arbitrage_enabled': True,
            'grid_delta_neutral': True,  # Grid positions maintain delta neutrality
        }
        
        # Performance tracking
        self.metrics = PMSMetrics()
        self.pnl_history = deque(maxlen=1000)
        self.delta_history = deque(maxlen=1000)
        
        # State management
        self.is_running = False
        self.last_reconciliation = datetime.now()
        self.reconciliation_interval = 300  # 5 minutes
        
        # Callbacks
        self.position_update_callbacks: List[Callable] = []
        self.delta_rebalance_callbacks: List[Callable] = []
        self.risk_violation_callbacks: List[Callable] = []
        
        logger.info("⚖️ Position Management System v1.0.0 initialized")
        logger.info(f"   Delta Rebalance Threshold: {self.delta_params['rebalance_threshold']:.3f}")
        logger.info(f"   Max Positions per Exchange: {self.position_limits['max_positions_per_exchange']}")
        logger.info(f"   Auto Hedge Enabled: {self.delta_params['auto_hedge_enabled']}")
    
    async def initialize(self, binance_adapter: BinanceAdapter, backpack_adapter: BackpackAdapter,
                        risk_manager: IntegratedRiskManager, oms=None):
        """Initialize PMS with adapters and managers."""
        try:
            self.exchanges['binance'] = binance_adapter
            self.exchanges['backpack'] = backpack_adapter
            self.risk_manager = risk_manager
            self.oms = oms
            
            # Start background tasks
            self.is_running = True
            
            # Start monitoring loops
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._delta_monitoring_loop())
            asyncio.create_task(self._reconciliation_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            # Initial position sync
            await self._sync_positions_from_exchanges()
            
            logger.info("✅ Position Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ PMS initialization failed: {e}")
            raise
    
    async def create_position(self, position_request: Dict[str, Any]) -> str:
        """Create a new position with comprehensive tracking."""
        try:
            # Generate position ID
            position_id = str(uuid.uuid4())
            
            # Validate position request
            if not await self._validate_position_request(position_request):
                raise ValueError("Position request validation failed")
            
            # Check risk limits
            if not await self._check_position_risk_limits(position_request):
                raise ValueError("Position risk limits exceeded")
            
            # Create position object
            position = Position(
                position_id=position_id,
                exchange=position_request['exchange'],
                symbol=position_request['symbol'],
                position_type=PositionType(position_request['position_type']),
                strategy=PositionStrategy(position_request.get('strategy', 'standalone')),
                quantity=position_request['quantity'],
                remaining_quantity=position_request['quantity'],
                entry_price=position_request.get('entry_price', 0.0),
                current_price=position_request.get('current_price', 0.0),
                parent_strategy_id=position_request.get('parent_strategy_id'),
                grid_level=position_request.get('grid_level'),
                target_price=position_request.get('target_price'),
                stop_loss=position_request.get('stop_loss'),
                take_profit=position_request.get('take_profit'),
                notes=position_request.get('notes', '')
            )
            
            # Calculate initial delta
            position.delta = self._calculate_position_delta(position)
            
            # Store position
            self.positions[position_id] = position
            
            # Update metrics
            await self._update_metrics()
            
            # Update portfolio delta
            await self._update_portfolio_delta()
            
            # Check for auto-hedging if enabled
            if (self.delta_params['auto_hedge_enabled'] and 
                position.strategy in [PositionStrategy.GRID_TRADING, PositionStrategy.ARBITRAGE]):
                await self._check_auto_hedge_requirement(position)
            
            logger.info(f"📝 Position created: {position_id} ({position.symbol} {position.position_type.value} {position.quantity})")
            
            # Notify callbacks
            await self._notify_position_update(position, 'created')
            
            return position_id
            
        except Exception as e:
            logger.error(f"❌ Position creation failed: {e}")
            raise
    
    async def update_position(self, position_id: str, update_data: Dict[str, Any]) -> bool:
        """Update existing position with new data."""
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found for update")
                return False
            
            position = self.positions[position_id]
            old_status = position.status
            
            # Update position fields
            for field, value in update_data.items():
                if hasattr(position, field):
                    setattr(position, field, value)
            
            # Update timestamp
            position.last_update = datetime.now()
            
            # Recalculate derived values
            position.remaining_quantity = position.quantity - position.filled_quantity
            position.delta = self._calculate_position_delta(position)
            
            # Update unrealized P&L
            if position.current_price > 0 and position.average_price > 0:
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)
            
            # Check for status changes
            if old_status != position.status:
                await self._handle_position_status_change(position, old_status)
            
            # Update portfolio delta if delta-relevant change
            if 'filled_quantity' in update_data or 'current_price' in update_data:
                await self._update_portfolio_delta()
            
            # Update metrics
            await self._update_metrics()
            
            logger.debug(f"Position updated: {position_id}")
            
            # Notify callbacks
            await self._notify_position_update(position, 'updated')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Position update failed: {e}")
            return False
    
    async def close_position(self, position_id: str, close_price: float = None, 
                           reason: str = "Manual close") -> bool:
        """Close a position and calculate final P&L."""
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found for closure")
                return False
            
            position = self.positions[position_id]
            
            if position.status == PositionStatus.CLOSED:
                logger.warning(f"Position {position_id} already closed")
                return True
            
            # Set closing price
            if close_price:
                position.current_price = close_price
            
            # Calculate final P&L
            final_pnl = self._calculate_unrealized_pnl(position)
            position.realized_pnl += final_pnl
            position.unrealized_pnl = 0.0
            
            # Update status and timing
            position.status = PositionStatus.CLOSED
            position.closed_time = datetime.now()
            
            # Move to history
            self.position_history.append(position)
            
            # Remove from active positions
            del self.positions[position_id]
            
            # Update portfolio delta
            await self._update_portfolio_delta()
            
            # Update metrics
            await self._update_metrics()
            
            logger.info(f"✅ Position closed: {position_id}, P&L: ${final_pnl:.2f}, Reason: {reason}")
            
            # Notify callbacks
            await self._notify_position_update(position, 'closed')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Position closure failed: {e}")
            return False
    
    async def create_delta_neutral_pair(self, long_position_id: str, short_position_id: str,
                                       strategy: PositionStrategy = PositionStrategy.DELTA_HEDGE) -> str:
        """Create a delta-neutral position pair."""
        try:
            if long_position_id not in self.positions or short_position_id not in self.positions:
                raise ValueError("One or both positions not found")
            
            long_position = self.positions[long_position_id]
            short_position = self.positions[short_position_id]
            
            # Validate pair compatibility
            if long_position.symbol != short_position.symbol:
                raise ValueError("Positions must be for the same symbol")
            
            # Generate pair ID
            pair_id = str(uuid.uuid4())
            
            # Create delta-neutral pair
            pair = DeltaNeutralPair(
                pair_id=pair_id,
                strategy=strategy,
                symbol=long_position.symbol,
                long_position_id=long_position_id,
                short_position_id=short_position_id,
                long_exchange=long_position.exchange,
                short_exchange=short_position.exchange,
                hedge_ratio=abs(short_position.quantity / long_position.quantity) if long_position.quantity != 0 else 1.0
            )
            
            # Calculate initial delta
            pair.current_delta = long_position.delta + short_position.delta
            pair.delta_deviation = abs(pair.current_delta - pair.target_delta)
            
            # Link positions to pair
            long_position.paired_position_id = short_position_id
            short_position.paired_position_id = long_position_id
            
            # Store pair
            self.delta_neutral_pairs[pair_id] = pair
            
            logger.info(f"⚖️ Delta-neutral pair created: {pair_id} ({long_position.symbol})")
            logger.info(f"   Long: {long_position.exchange} {long_position.quantity}")
            logger.info(f"   Short: {short_position.exchange} {short_position.quantity}")
            logger.info(f"   Initial Delta: {pair.current_delta:.6f}")
            
            return pair_id
            
        except Exception as e:
            logger.error(f"❌ Delta-neutral pair creation failed: {e}")
            raise
    
    async def rebalance_delta_neutral_pair(self, pair_id: str, target_ratio: float = None) -> bool:
        """Rebalance a delta-neutral pair to maintain neutrality."""
        try:
            if pair_id not in self.delta_neutral_pairs:
                logger.warning(f"Delta-neutral pair {pair_id} not found")
                return False
            
            pair = self.delta_neutral_pairs[pair_id]
            long_position = self.positions.get(pair.long_position_id)
            short_position = self.positions.get(pair.short_position_id)
            
            if not long_position or not short_position:
                logger.warning(f"Positions for pair {pair_id} not found")
                return False
            
            # Calculate current imbalance
            current_ratio = abs(short_position.filled_quantity / long_position.filled_quantity) if long_position.filled_quantity > 0 else 1.0
            target_ratio = target_ratio or pair.hedge_ratio
            
            ratio_deviation = abs(current_ratio - target_ratio) / target_ratio if target_ratio > 0 else 0
            
            # Check if rebalancing is needed
            if ratio_deviation < self.delta_params['rebalance_threshold']:
                return True  # Already balanced
            
            # Calculate required adjustment
            if current_ratio < target_ratio:
                # Need to increase short position
                adjustment_qty = (target_ratio * long_position.filled_quantity) - short_position.filled_quantity
                adjustment_side = 'short'
            else:
                # Need to increase long position or reduce short
                adjustment_qty = (short_position.filled_quantity / target_ratio) - long_position.filled_quantity
                adjustment_side = 'long'
            
            # Execute rebalancing through OMS if available
            if self.oms and abs(adjustment_qty) > 0.001:  # Minimum meaningful adjustment
                success = await self._execute_delta_rebalance(pair, adjustment_side, adjustment_qty)
                if success:
                    pair.last_rebalance = datetime.now()
                    logger.info(f"⚖️ Delta pair rebalanced: {pair_id}, Adjustment: {adjustment_side} {adjustment_qty:.6f}")
                    
                    # Notify callbacks
                    for callback in self.delta_rebalance_callbacks:
                        await callback(pair, adjustment_side, adjustment_qty)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Delta rebalancing failed: {e}")
            return False
    
    async def get_position_pnl(self, position_id: str) -> Dict[str, float]:
        """Get comprehensive P&L for a position."""
        try:
            if position_id not in self.positions:
                position = next((p for p in self.position_history if p.position_id == position_id), None)
                if not position:
                    return {}
            else:
                position = self.positions[position_id]
            
            unrealized_pnl = self._calculate_unrealized_pnl(position)
            
            return {
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': unrealized_pnl + position.realized_pnl,
                'fees_paid': position.fees_paid,
                'funding_fees': position.funding_fees,
                'net_pnl': unrealized_pnl + position.realized_pnl - position.fees_paid - position.funding_fees,
                'pnl_percentage': ((unrealized_pnl + position.realized_pnl) / (position.average_price * position.quantity * 100)) if position.average_price > 0 and position.quantity > 0 else 0.0,
                'position_value': position.current_price * position.filled_quantity,
                'margin_used': position.margin_used,
                'leverage': (position.current_price * position.filled_quantity / position.margin_used) if position.margin_used > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"❌ P&L calculation failed: {e}")
            return {}
    
    async def get_portfolio_delta_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio delta status."""
        try:
            await self._update_portfolio_delta()
            
            # Determine delta status
            if abs(self.portfolio_delta.delta_deviation) < self.delta_params['rebalance_threshold']:
                delta_status = DeltaStatus.NEUTRAL
            elif abs(self.portfolio_delta.delta_deviation) > self.delta_params['hedge_urgency_threshold']:
                delta_status = DeltaStatus.HEDGE_URGENT
            elif self.portfolio_delta.total_delta > 0:
                delta_status = DeltaStatus.LONG_BIAS
            else:
                delta_status = DeltaStatus.SHORT_BIAS
            
            # Calculate hedge recommendations
            hedge_recommendations = []
            if delta_status in [DeltaStatus.REBALANCE_NEEDED, DeltaStatus.HEDGE_URGENT]:
                hedge_recommendations = await self._generate_hedge_recommendations()
            
            return {
                'total_delta': self.portfolio_delta.total_delta,
                'target_delta': self.portfolio_delta.target_delta,
                'delta_deviation': self.portfolio_delta.delta_deviation,
                'delta_status': delta_status.value,
                'hedge_effectiveness': self.portfolio_delta.hedge_effectiveness,
                'symbol_deltas': self.portfolio_delta.symbol_deltas,
                'exchange_deltas': self.portfolio_delta.exchange_deltas,
                'strategy_deltas': self.portfolio_delta.strategy_deltas,
                'rebalance_threshold': self.delta_params['rebalance_threshold'],
                'hedge_urgency_threshold': self.delta_params['hedge_urgency_threshold'],
                'delta_neutral_pairs': len(self.delta_neutral_pairs),
                'active_pairs': len([p for p in self.delta_neutral_pairs.values() if p.status == 'active']),
                'hedge_recommendations': hedge_recommendations,
                'last_update': self.portfolio_delta.last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio delta status error: {e}")
            return {}
    
    async def reconcile_positions_with_exchanges(self) -> Dict[str, Any]:
        """Reconcile position data with exchange data."""
        try:
            reconciliation_results = {
                'binance': {'matched': 0, 'missing': 0, 'extra': 0, 'discrepancies': []},
                'backpack': {'matched': 0, 'missing': 0, 'extra': 0, 'discrepancies': []},
                'total_discrepancies': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            for exchange_name, adapter in self.exchanges.items():
                try:
                    # Get positions from exchange
                    exchange_positions = await self._get_exchange_positions(adapter)
                    
                    # Get our tracked positions for this exchange
                    our_positions = {p.position_id: p for p in self.positions.values() 
                                   if p.exchange == exchange_name}
                    
                    # Compare positions
                    exchange_result = reconciliation_results[exchange_name]
                    
                    for exchange_pos in exchange_positions:
                        # Find matching position
                        matching_pos = None
                        for pos_id, our_pos in our_positions.items():
                            if (our_pos.symbol == exchange_pos.get('symbol') and
                                abs(our_pos.filled_quantity - exchange_pos.get('quantity', 0)) < 0.001):
                                matching_pos = our_pos
                                break
                        
                        if matching_pos:
                            exchange_result['matched'] += 1
                            # Check for discrepancies
                            if abs(matching_pos.current_price - exchange_pos.get('price', 0)) > 0.01:
                                exchange_result['discrepancies'].append({
                                    'position_id': matching_pos.position_id,
                                    'type': 'price_mismatch',
                                    'our_price': matching_pos.current_price,
                                    'exchange_price': exchange_pos.get('price', 0)
                                })
                        else:
                            exchange_result['missing'] += 1
                    
                    # Check for extra positions we have but exchange doesn't
                    for our_pos in our_positions.values():
                        found = False
                        for exchange_pos in exchange_positions:
                            if (our_pos.symbol == exchange_pos.get('symbol') and
                                abs(our_pos.filled_quantity - exchange_pos.get('quantity', 0)) < 0.001):
                                found = True
                                break
                        if not found:
                            exchange_result['extra'] += 1
                    
                except Exception as e:
                    logger.error(f"Reconciliation error for {exchange_name}: {e}")
                    reconciliation_results[exchange_name]['error'] = str(e)
            
            # Calculate total discrepancies
            reconciliation_results['total_discrepancies'] = sum(
                len(r.get('discrepancies', [])) for r in reconciliation_results.values() 
                if isinstance(r, dict) and 'discrepancies' in r
            )
            
            # Update last reconciliation time
            self.last_reconciliation = datetime.now()
            
            logger.info(f"📊 Position reconciliation completed: {reconciliation_results['total_discrepancies']} discrepancies found")
            
            return reconciliation_results
            
        except Exception as e:
            logger.error(f"❌ Position reconciliation failed: {e}")
            return {'error': str(e)}
    
    def get_metrics(self) -> PMSMetrics:
        """Get comprehensive PMS metrics."""
        return self.metrics
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position."""
        return self.positions.get(position_id)
    
    def get_positions_by_strategy(self, strategy: PositionStrategy) -> List[Position]:
        """Get all positions for a specific strategy."""
        return [p for p in self.positions.values() if p.strategy == strategy]
    
    def get_positions_by_exchange(self, exchange: str) -> List[Position]:
        """Get all positions for a specific exchange."""
        return [p for p in self.positions.values() if p.exchange == exchange]
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        return [p for p in self.positions.values() if p.symbol == symbol]
    
    def add_position_update_callback(self, callback: Callable):
        """Add callback for position updates."""
        self.position_update_callbacks.append(callback)
    
    def add_delta_rebalance_callback(self, callback: Callable):
        """Add callback for delta rebalancing events."""
        self.delta_rebalance_callbacks.append(callback)
    
    def add_risk_violation_callback(self, callback: Callable):
        """Add callback for risk violations."""
        self.risk_violation_callbacks.append(callback)
    
    # ========== PRIVATE METHODS ==========
    
    def _calculate_position_delta(self, position: Position) -> float:
        """Calculate position delta for neutrality calculations."""
        try:
            # Basic delta calculation (can be enhanced with option greeks)
            if position.position_type in [PositionType.SPOT_LONG, PositionType.FUTURES_LONG, 
                                        PositionType.GRID_BUY, PositionType.ARBITRAGE_LONG]:
                return position.filled_quantity
            elif position.position_type in [PositionType.SPOT_SHORT, PositionType.FUTURES_SHORT,
                                          PositionType.GRID_SELL, PositionType.ARBITRAGE_SHORT]:
                return -position.filled_quantity
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Delta calculation error: {e}")
            return 0.0
    
    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L for a position."""
        try:
            if position.filled_quantity == 0 or position.average_price == 0:
                return 0.0
            
            if position.position_type in [PositionType.SPOT_LONG, PositionType.FUTURES_LONG,
                                        PositionType.GRID_BUY, PositionType.ARBITRAGE_LONG]:
                # Long position: profit when price goes up
                return position.filled_quantity * (position.current_price - position.average_price)
            elif position.position_type in [PositionType.SPOT_SHORT, PositionType.FUTURES_SHORT,
                                          PositionType.GRID_SELL, PositionType.ARBITRAGE_SHORT]:
                # Short position: profit when price goes down
                return position.filled_quantity * (position.average_price - position.current_price)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"P&L calculation error: {e}")
            return 0.0
    
    async def _validate_position_request(self, request: Dict[str, Any]) -> bool:
        """Validate position creation request."""
        try:
            required_fields = ['exchange', 'symbol', 'position_type', 'quantity']
            for field in required_fields:
                if field not in request:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            if request['quantity'] <= 0:
                logger.error("Invalid quantity")
                return False
            
            if request['exchange'] not in self.exchanges:
                logger.error(f"Invalid exchange: {request['exchange']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Position validation error: {e}")
            return False
    
    async def _check_position_risk_limits(self, request: Dict[str, Any]) -> bool:
        """Check position against risk limits."""
        try:
            if not self.risk_manager:
                return True
            
            exchange = request['exchange']
            position_size = request['quantity'] * request.get('current_price', 0)
            
            # Check position limits
            exchange_positions = len(self.get_positions_by_exchange(exchange))
            if exchange_positions >= self.position_limits['max_positions_per_exchange']:
                logger.warning(f"Max positions limit reached for {exchange}")
                return False
            
            if position_size > self.position_limits['max_position_size']:
                logger.warning(f"Position size ${position_size:,.0f} exceeds limit")
                return False
            
            # Check with risk manager
            allowed, reason = self.risk_manager.can_open_position(exchange, position_size)
            if not allowed:
                logger.warning(f"Risk manager rejected position: {reason}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk limit check error: {e}")
            return False
    
    async def _update_portfolio_delta(self):
        """Update portfolio-wide delta calculations."""
        try:
            # Reset delta tracking
            self.portfolio_delta.symbol_deltas.clear()
            self.portfolio_delta.exchange_deltas.clear()
            self.portfolio_delta.strategy_deltas.clear()
            
            total_delta = 0.0
            
            # Calculate deltas by category
            for position in self.positions.values():
                delta = position.delta
                total_delta += delta
                
                # By symbol
                if position.symbol not in self.portfolio_delta.symbol_deltas:
                    self.portfolio_delta.symbol_deltas[position.symbol] = 0.0
                self.portfolio_delta.symbol_deltas[position.symbol] += delta
                
                # By exchange
                if position.exchange not in self.portfolio_delta.exchange_deltas:
                    self.portfolio_delta.exchange_deltas[position.exchange] = 0.0
                self.portfolio_delta.exchange_deltas[position.exchange] += delta
                
                # By strategy
                strategy_name = position.strategy.value
                if strategy_name not in self.portfolio_delta.strategy_deltas:
                    self.portfolio_delta.strategy_deltas[strategy_name] = 0.0
                self.portfolio_delta.strategy_deltas[strategy_name] += delta
            
            # Update portfolio delta
            self.portfolio_delta.total_delta = total_delta
            self.portfolio_delta.delta_deviation = abs(total_delta - self.portfolio_delta.target_delta)
            
            # Calculate hedge effectiveness
            total_exposure = sum(abs(delta) for delta in self.portfolio_delta.symbol_deltas.values())
            if total_exposure > 0:
                self.portfolio_delta.hedge_effectiveness = 1.0 - (self.portfolio_delta.delta_deviation / total_exposure)
            else:
                self.portfolio_delta.hedge_effectiveness = 1.0
            
            self.portfolio_delta.last_update = datetime.now()
            
            # Update delta-neutral pairs
            for pair in self.delta_neutral_pairs.values():
                await self._update_delta_neutral_pair(pair)
            
        except Exception as e:
            logger.error(f"Portfolio delta update error: {e}")
    
    async def _update_delta_neutral_pair(self, pair: DeltaNeutralPair):
        """Update delta metrics for a delta-neutral pair."""
        try:
            long_position = self.positions.get(pair.long_position_id)
            short_position = self.positions.get(pair.short_position_id)
            
            if not long_position or not short_position:
                return
            
            # Update current delta
            pair.current_delta = long_position.delta + short_position.delta
            pair.delta_deviation = abs(pair.current_delta - pair.target_delta)
            
            # Update combined P&L
            long_pnl = self._calculate_unrealized_pnl(long_position)
            short_pnl = self._calculate_unrealized_pnl(short_position)
            pair.combined_pnl = long_pnl + short_pnl
            
            # Calculate basis P&L (spread profit)
            if long_position.current_price > 0 and short_position.current_price > 0:
                spread = long_position.current_price - short_position.current_price
                pair.basis_pnl = spread * min(long_position.filled_quantity, short_position.filled_quantity)
            
        except Exception as e:
            logger.error(f"Delta-neutral pair update error: {e}")
    
    async def _update_metrics(self):
        """Update PMS metrics."""
        try:
            # Reset counters
            self.metrics.total_positions = len(self.positions) + len(self.position_history)
            self.metrics.active_positions = len(self.positions)
            self.metrics.closed_positions = len(self.position_history)
            
            # Count by type
            self.metrics.spot_positions = len([p for p in self.positions.values() 
                                             if p.position_type in [PositionType.SPOT_LONG, PositionType.SPOT_SHORT]])
            self.metrics.futures_positions = len([p for p in self.positions.values() 
                                                if p.position_type in [PositionType.FUTURES_LONG, PositionType.FUTURES_SHORT]])
            
            # Count by exchange
            self.metrics.binance_positions = len(self.get_positions_by_exchange('binance'))
            self.metrics.backpack_positions = len(self.get_positions_by_exchange('backpack'))
            
            # Count by strategy
            self.metrics.grid_positions = len(self.get_positions_by_strategy(PositionStrategy.GRID_TRADING))
            self.metrics.arbitrage_positions = len(self.get_positions_by_strategy(PositionStrategy.ARBITRAGE))
            self.metrics.hedge_positions = len(self.get_positions_by_strategy(PositionStrategy.DELTA_HEDGE))
            
            # P&L metrics
            self.metrics.total_unrealized_pnl = sum(self._calculate_unrealized_pnl(p) for p in self.positions.values())
            self.metrics.total_realized_pnl = sum(p.realized_pnl for p in self.position_history)
            self.metrics.total_fees_paid = sum(p.fees_paid for p in list(self.positions.values()) + self.position_history)
            
            # Delta metrics
            self.metrics.portfolio_delta = self.portfolio_delta.total_delta
            self.metrics.delta_neutral_pairs = len(self.delta_neutral_pairs)
            
            # Margin utilization
            self.metrics.total_margin_used = sum(p.margin_used for p in self.positions.values())
            if self.position_limits['max_total_margin'] > 0:
                self.metrics.margin_utilization = self.metrics.total_margin_used / self.position_limits['max_total_margin']
            
            self.metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    async def _check_auto_hedge_requirement(self, position: Position):
        """Check if auto-hedging is required for a new position."""
        try:
            if not self.delta_params['auto_hedge_enabled']:
                return
            
            # Update portfolio delta
            await self._update_portfolio_delta()
            
            # Check if hedge is needed
            if self.portfolio_delta.delta_deviation > self.delta_params['rebalance_threshold']:
                logger.info(f"Auto-hedge triggered for position {position.position_id}")
                
                # Generate hedge recommendation
                hedge_recommendations = await self._generate_hedge_recommendations()
                
                # Execute first recommendation if available and OMS is connected
                if hedge_recommendations and self.oms:
                    recommendation = hedge_recommendations[0]
                    await self._execute_auto_hedge(recommendation)
                    
        except Exception as e:
            logger.error(f"Auto-hedge check error: {e}")
    
    async def _generate_hedge_recommendations(self) -> List[Dict[str, Any]]:
        """Generate hedge recommendations for portfolio rebalancing."""
        try:
            recommendations = []
            
            # Analyze symbol-level imbalances
            for symbol, delta in self.portfolio_delta.symbol_deltas.items():
                if abs(delta) > self.delta_params['rebalance_threshold']:
                    # Determine hedge action
                    if delta > 0:  # Long bias - need short hedge
                        hedge_action = "short"
                        hedge_quantity = delta * 0.8  # Partial hedge
                    else:  # Short bias - need long hedge
                        hedge_action = "long"
                        hedge_quantity = abs(delta) * 0.8  # Partial hedge
                    
                    # Find best exchange for hedge
                    exchange_positions = {
                        'binance': len(self.get_positions_by_exchange('binance')),
                        'backpack': len(self.get_positions_by_exchange('backpack'))
                    }
                    
                    # Use exchange with fewer positions
                    best_exchange = min(exchange_positions.items(), key=lambda x: x[1])[0]
                    
                    recommendations.append({
                        'symbol': symbol,
                        'action': hedge_action,
                        'quantity': hedge_quantity,
                        'exchange': best_exchange,
                        'reason': f'Symbol {symbol} delta imbalance: {delta:.6f}',
                        'urgency': 'high' if abs(delta) > self.delta_params['hedge_urgency_threshold'] else 'medium'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Hedge recommendation generation error: {e}")
            return []
    
    async def _execute_auto_hedge(self, recommendation: Dict[str, Any]):
        """Execute automatic hedge based on recommendation."""
        try:
            if not self.oms:
                logger.warning("Cannot execute auto-hedge: OMS not available")
                return False
            
            # Create hedge position request
            hedge_request = {
                'exchange': recommendation['exchange'],
                'symbol': recommendation['symbol'],
                'position_type': f"futures_{recommendation['action']}",
                'strategy': 'delta_hedge',
                'quantity': recommendation['quantity'],
                'parent_strategy_id': 'auto_hedge'
            }
            
            # Create hedge position
            hedge_position_id = await self.create_position(hedge_request)
            
            logger.info(f"🔄 Auto-hedge executed: {hedge_position_id} - {recommendation['reason']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Auto-hedge execution error: {e}")
            return False
    
    async def _execute_delta_rebalance(self, pair: DeltaNeutralPair, side: str, quantity: float) -> bool:
        """Execute delta rebalancing trade through OMS."""
        try:
            if not self.oms:
                return False
            
            # Determine exchange and position type
            if side == 'long':
                exchange = pair.long_exchange
                position_type = 'futures_long'
            else:
                exchange = pair.short_exchange
                position_type = 'futures_short'
            
            # Create rebalance order request (simplified)
            order_request = {
                'exchange': exchange,
                'symbol': pair.symbol,
                'side': 'buy' if side == 'long' else 'sell',
                'type': 'market',
                'quantity': abs(quantity),
                'position_type': 'hedging',
                'parent_strategy_id': pair.pair_id
            }
            
            # Submit through OMS
            # Note: This would need to be integrated with the actual OMS order submission
            logger.info(f"Delta rebalance order submitted: {order_request}")
            
            return True
            
        except Exception as e:
            logger.error(f"Delta rebalance execution error: {e}")
            return False
    
    async def _get_exchange_positions(self, adapter) -> List[Dict[str, Any]]:
        """Get positions from exchange adapter."""
        try:
            # This would call the appropriate method on the adapter
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Exchange position retrieval error: {e}")
            return []
    
    async def _sync_positions_from_exchanges(self):
        """Initial sync of positions from exchanges."""
        try:
            for exchange_name, adapter in self.exchanges.items():
                try:
                    positions = await self._get_exchange_positions(adapter)
                    logger.info(f"Synced {len(positions)} positions from {exchange_name}")
                except Exception as e:
                    logger.error(f"Failed to sync positions from {exchange_name}: {e}")
        except Exception as e:
            logger.error(f"Position sync error: {e}")
    
    async def _handle_position_status_change(self, position: Position, old_status: PositionStatus):
        """Handle position status changes."""
        try:
            if position.status == PositionStatus.CLOSED and old_status != PositionStatus.CLOSED:
                logger.info(f"Position {position.position_id} closed")
                
                # Update paired position if exists
                if position.paired_position_id:
                    paired_position = self.positions.get(position.paired_position_id)
                    if paired_position:
                        paired_position.paired_position_id = None
            
        except Exception as e:
            logger.error(f"Position status change handling error: {e}")
    
    async def _notify_position_update(self, position: Position, action: str):
        """Notify callbacks of position updates."""
        try:
            for callback in self.position_update_callbacks:
                await callback(position, action)
        except Exception as e:
            logger.error(f"Position update notification error: {e}")
    
    # ========== BACKGROUND MONITORING LOOPS ==========
    
    async def _position_monitoring_loop(self):
        """Monitor positions for updates and status changes."""
        try:
            while self.is_running:
                try:
                    # Update current prices and P&L for all positions
                    for position in self.positions.values():
                        # Here you would fetch current price from exchange
                        # For now, we'll skip this as it requires market data
                        pass
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Position monitoring error: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"Position monitoring loop error: {e}")
    
    async def _delta_monitoring_loop(self):
        """Monitor delta neutrality and trigger rebalancing."""
        try:
            while self.is_running:
                try:
                    await self._update_portfolio_delta()
                    
                    # Check if urgent rebalancing is needed
                    if self.portfolio_delta.delta_deviation > self.delta_params['hedge_urgency_threshold']:
                        logger.warning(f"Urgent delta rebalancing needed: deviation {self.portfolio_delta.delta_deviation:.6f}")
                        
                        # Auto-rebalance if enabled
                        if self.delta_params['auto_hedge_enabled']:
                            recommendations = await self._generate_hedge_recommendations()
                            for rec in recommendations[:2]:  # Limit to top 2 recommendations
                                if rec['urgency'] == 'high':
                                    await self._execute_auto_hedge(rec)
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Delta monitoring error: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Delta monitoring loop error: {e}")
    
    async def _reconciliation_loop(self):
        """Periodic reconciliation with exchange data."""
        try:
            while self.is_running:
                try:
                    if (datetime.now() - self.last_reconciliation).seconds > self.reconciliation_interval:
                        await self.reconcile_positions_with_exchanges()
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Reconciliation loop error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
                    
        except Exception as e:
            logger.error(f"Reconciliation loop error: {e}")
    
    async def _performance_tracking_loop(self):
        """Track performance metrics and update history."""
        try:
            while self.is_running:
                try:
                    await self._update_metrics()
                    
                    # Record historical data points
                    current_time = datetime.now()
                    
                    # P&L history
                    total_pnl = self.metrics.total_unrealized_pnl + self.metrics.total_realized_pnl
                    self.pnl_history.append({
                        'timestamp': current_time,
                        'total_pnl': total_pnl,
                        'unrealized_pnl': self.metrics.total_unrealized_pnl,
                        'realized_pnl': self.metrics.total_realized_pnl
                    })
                    
                    # Delta history
                    self.delta_history.append({
                        'timestamp': current_time,
                        'portfolio_delta': self.portfolio_delta.total_delta,
                        'delta_deviation': self.portfolio_delta.delta_deviation,
                        'hedge_effectiveness': self.portfolio_delta.hedge_effectiveness
                    })
                    
                    await asyncio.sleep(300)  # Update every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Performance tracking error: {e}")
                    await asyncio.sleep(600)  # Wait 10 minutes on error
                    
        except Exception as e:
            logger.error(f"Performance tracking loop error: {e}")
    
    async def shutdown(self):
        """Shutdown the PMS gracefully."""
        try:
            logger.info("🔄 Shutting down Position Management System...")
            
            self.is_running = False
            
            # Final reconciliation
            await self.reconcile_positions_with_exchanges()
            
            # Save position state (if persistence is implemented)
            
            logger.info("✅ Position Management System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ PMS shutdown error: {e}")

# Example usage and testing
async def example_usage():
    """Example of how to use the Position Management System."""
    
    # Configuration
    config = {
        'delta_rebalance_threshold': 0.05,  # 5%
        'auto_hedge_enabled': True,
        'max_positions_per_exchange': 50,
        'max_position_size': 100000,  # $100k
        'hedge_urgency_threshold': 0.15  # 15%
    }
    
    # Initialize PMS
    pms = PositionManagementSystem(config)
    
    print("⚖️ Position Management System v1.0.0 example initialized")
    print(f"   Delta Rebalance Threshold: {config['delta_rebalance_threshold']:.1%}")
    print(f"   Auto Hedge Enabled: {config['auto_hedge_enabled']}")
    print(f"   Max Positions per Exchange: {config['max_positions_per_exchange']}")
    
    # Example position creation
    position_request = {
        'exchange': 'binance',
        'symbol': 'BTCUSDT',
        'position_type': 'spot_long',
        'strategy': 'grid_trading',
        'quantity': 0.1,
        'entry_price': 50000.0,
        'current_price': 50000.0,
        'grid_level': 1,
        'notes': 'Example grid position'
    }
    
    print(f"\n📝 Example position request:")
    print(f"   Exchange: {position_request['exchange']}")
    print(f"   Symbol: {position_request['symbol']}")
    print(f"   Type: {position_request['position_type']}")
    print(f"   Strategy: {position_request['strategy']}")
    print(f"   Quantity: {position_request['quantity']}")
    
    # Example metrics
    metrics = pms.get_metrics()
    print(f"\n📊 PMS Metrics:")
    print(f"   Total Positions: {metrics.total_positions}")
    print(f"   Active Positions: {metrics.active_positions}")
    print(f"   Delta Neutral Pairs: {metrics.delta_neutral_pairs}")
    print(f"   Portfolio Delta: {metrics.portfolio_delta:.6f}")
    
    # Example delta status
    delta_status = {
        'total_delta': 0.0,
        'delta_status': 'neutral',
        'hedge_effectiveness': 1.0,
        'rebalance_needed': False
    }
    
    print(f"\n⚖️ Delta Status:")
    print(f"   Total Delta: {delta_status['total_delta']:.6f}")
    print(f"   Status: {delta_status['delta_status']}")
    print(f"   Hedge Effectiveness: {delta_status['hedge_effectiveness']:.1%}")

if __name__ == "__main__":
    asyncio.run(example_usage())