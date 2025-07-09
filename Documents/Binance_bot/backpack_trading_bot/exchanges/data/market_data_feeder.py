#!/usr/bin/env python3
"""
🎯 MULTI-EXCHANGE MARKET DATA FEEDER v1.0.0
Synchronized high-performance market data aggregation system

Features:
- 📡 Multi-exchange WebSocket data synchronization
- ⚡ Low-latency data feeds for arbitrage opportunities
- 📊 Real-time market data normalization
- 🔄 Automatic connection recovery and failover
- 📈 Historical data management and storage
- 🛡️ Data quality assurance and validation
- ⏱️ Timestamp synchronization across exchanges
- 📦 Flexible subscription management system
"""

import asyncio
import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import websockets
import aiohttp
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref

logger = logging.getLogger(__name__)

class DataType(Enum):
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINES = "klines"
    USER_DATA = "user_data"

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class MarketDataEvent:
    """Standardized market data event across exchanges."""
    exchange: str
    symbol: str
    data_type: DataType
    data: Dict[str, Any]
    timestamp: datetime
    server_timestamp: Optional[datetime] = None
    latency_ms: Optional[float] = None

@dataclass
class TickerData:
    """Normalized ticker data."""
    symbol: str
    exchange: str
    price: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    volume_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'price_change_pct_24h': self.price_change_pct_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class OrderBookData:
    """Normalized order book data."""
    symbol: str
    exchange: str
    bids: List[List[float]]  # [price, size]
    asks: List[List[float]]  # [price, size]
    timestamp: datetime
    
    def get_best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        return best_ask - best_bid if best_bid and best_ask else None

@dataclass
class TradeData:
    """Normalized trade data."""
    symbol: str
    exchange: str
    trade_id: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime

@dataclass
class KlineData:
    """Normalized kline/candlestick data."""
    symbol: str
    exchange: str
    interval: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    is_closed: bool

@dataclass
class SubscriptionConfig:
    """Configuration for data subscriptions."""
    exchange: str
    symbol: str
    data_type: DataType
    params: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

class DataBuffer:
    """Thread-safe circular buffer for market data."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self._total_count = 0
    
    def append(self, item: Any):
        with self.lock:
            self.buffer.append(item)
            self._total_count += 1
    
    def get_latest(self, n: int = 1) -> List[Any]:
        with self.lock:
            if n == 1:
                return [self.buffer[-1]] if self.buffer else []
            return list(self.buffer)[-n:] if self.buffer else []
    
    def get_all(self) -> List[Any]:
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)

class PerformanceMonitor:
    """Monitor data feed performance and latency."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'message_count': 0,
            'total_latency': 0.0,
            'min_latency': float('inf'),
            'max_latency': 0.0,
            'error_count': 0,
            'last_update': None
        })
        self.lock = threading.RLock()
    
    def record_message(self, exchange: str, symbol: str, latency_ms: float):
        key = f"{exchange}:{symbol}"
        with self.lock:
            metric = self.metrics[key]
            metric['message_count'] += 1
            metric['total_latency'] += latency_ms
            metric['min_latency'] = min(metric['min_latency'], latency_ms)
            metric['max_latency'] = max(metric['max_latency'], latency_ms)
            metric['last_update'] = datetime.now()
    
    def record_error(self, exchange: str, symbol: str):
        key = f"{exchange}:{symbol}"
        with self.lock:
            self.metrics[key]['error_count'] += 1
    
    def get_stats(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]:
        with self.lock:
            if exchange and symbol:
                key = f"{exchange}:{symbol}"
                if key in self.metrics:
                    metric = self.metrics[key]
                    avg_latency = metric['total_latency'] / metric['message_count'] if metric['message_count'] > 0 else 0
                    return {
                        'exchange': exchange,
                        'symbol': symbol,
                        'message_count': metric['message_count'],
                        'avg_latency_ms': avg_latency,
                        'min_latency_ms': metric['min_latency'] if metric['min_latency'] != float('inf') else 0,
                        'max_latency_ms': metric['max_latency'],
                        'error_count': metric['error_count'],
                        'last_update': metric['last_update']
                    }
                return {}
            
            # Return all stats
            stats = {}
            for key, metric in self.metrics.items():
                exchange, symbol = key.split(':', 1)
                avg_latency = metric['total_latency'] / metric['message_count'] if metric['message_count'] > 0 else 0
                stats[key] = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'message_count': metric['message_count'],
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': metric['min_latency'] if metric['min_latency'] != float('inf') else 0,
                    'max_latency_ms': metric['max_latency'],
                    'error_count': metric['error_count'],
                    'last_update': metric['last_update']
                }
            return stats

class DataValidator:
    """Validate market data quality and integrity."""
    
    def __init__(self):
        self.price_ranges = {}  # symbol -> (min_price, max_price)
        self.last_prices = {}   # (exchange, symbol) -> price
        self.validation_rules = {
            'max_price_deviation': 0.1,  # 10% max price change
            'min_price_value': 0.0001,   # Minimum valid price
            'max_spread_ratio': 0.05,    # 5% max spread
            'max_latency_ms': 5000       # 5 second max latency
        }
    
    def validate_ticker(self, ticker: TickerData) -> bool:
        """Validate ticker data integrity."""
        try:
            # Basic value checks
            if ticker.price <= self.validation_rules['min_price_value']:
                logger.warning(f"Invalid price for {ticker.symbol}: {ticker.price}")
                return False
            
            if ticker.bid >= ticker.ask:
                logger.warning(f"Invalid bid/ask for {ticker.symbol}: bid={ticker.bid}, ask={ticker.ask}")
                return False
            
            # Spread check
            spread_ratio = (ticker.ask - ticker.bid) / ticker.price
            if spread_ratio > self.validation_rules['max_spread_ratio']:
                logger.warning(f"Wide spread for {ticker.symbol}: {spread_ratio:.4f}")
                # Don't reject, but log warning
            
            # Price deviation check
            key = (ticker.exchange, ticker.symbol)
            if key in self.last_prices:
                last_price = self.last_prices[key]
                price_change = abs(ticker.price - last_price) / last_price
                if price_change > self.validation_rules['max_price_deviation']:
                    logger.warning(f"Large price change for {ticker.symbol}: {price_change:.4f}")
                    # Don't reject for now, could be legitimate
            
            self.last_prices[key] = ticker.price
            return True
            
        except Exception as e:
            logger.error(f"Ticker validation error: {e}")
            return False
    
    def validate_orderbook(self, orderbook: OrderBookData) -> bool:
        """Validate order book data integrity."""
        try:
            # Check if bids and asks exist
            if not orderbook.bids or not orderbook.asks:
                return False
            
            # Check bid/ask ordering
            for i in range(len(orderbook.bids) - 1):
                if orderbook.bids[i][0] <= orderbook.bids[i+1][0]:
                    logger.warning(f"Invalid bid ordering for {orderbook.symbol}")
                    return False
            
            for i in range(len(orderbook.asks) - 1):
                if orderbook.asks[i][0] >= orderbook.asks[i+1][0]:
                    logger.warning(f"Invalid ask ordering for {orderbook.symbol}")
                    return False
            
            # Check spread
            best_bid = orderbook.get_best_bid()
            best_ask = orderbook.get_best_ask()
            if best_bid >= best_ask:
                logger.warning(f"Invalid spread for {orderbook.symbol}: bid={best_bid}, ask={best_ask}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Orderbook validation error: {e}")
            return False

class HistoricalDataManager:
    """Manage historical market data storage and retrieval."""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for historical data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ticker data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticker_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Order book snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    best_bid REAL NOT NULL,
                    best_ask REAL NOT NULL,
                    spread REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    message_count INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_symbol_time ON ticker_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time ON orderbook_snapshots(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            logger.info("📊 Historical data database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    async def store_ticker(self, ticker: TickerData):
        """Store ticker data asynchronously."""
        try:
            def _store():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ticker_data 
                    (exchange, symbol, price, bid, ask, volume_24h, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker.exchange,
                    ticker.symbol,
                    ticker.price,
                    ticker.bid,
                    ticker.ask,
                    ticker.volume_24h,
                    ticker.timestamp.isoformat()
                ))
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(self.executor, _store)
            
        except Exception as e:
            logger.error(f"❌ Ticker storage error: {e}")
    
    async def store_orderbook_snapshot(self, orderbook: OrderBookData):
        """Store order book snapshot asynchronously."""
        try:
            def _store():
                spread = orderbook.get_spread()
                if spread is None:
                    return
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO orderbook_snapshots 
                    (exchange, symbol, best_bid, best_ask, spread, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    orderbook.exchange,
                    orderbook.symbol,
                    orderbook.get_best_bid(),
                    orderbook.get_best_ask(),
                    spread,
                    orderbook.timestamp.isoformat()
                ))
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(self.executor, _store)
            
        except Exception as e:
            logger.error(f"❌ Orderbook storage error: {e}")

class ExchangeConnector:
    """Base class for exchange-specific WebSocket connectors."""
    
    def __init__(self, exchange_name: str, config: Dict[str, Any]):
        self.exchange_name = exchange_name
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.websocket = None
        self.subscriptions = set()
        self.callbacks = defaultdict(list)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5.0
        self.last_heartbeat = None
        self.heartbeat_interval = 30.0
        
    async def connect(self):
        """Connect to exchange WebSocket."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from exchange WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.status = ConnectionStatus.DISCONNECTED
    
    async def subscribe(self, subscription: SubscriptionConfig):
        """Subscribe to data feed."""
        raise NotImplementedError
    
    async def unsubscribe(self, subscription: SubscriptionConfig):
        """Unsubscribe from data feed."""
        raise NotImplementedError
    
    def add_callback(self, data_type: DataType, callback: Callable):
        """Add callback for data type."""
        self.callbacks[data_type].append(callback)
    
    def remove_callback(self, data_type: DataType, callback: Callable):
        """Remove callback for data type."""
        if callback in self.callbacks[data_type]:
            self.callbacks[data_type].remove(callback)
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        raise NotImplementedError
    
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts reached for {self.exchange_name}")
            self.status = ConnectionStatus.ERROR
            return
        
        self.reconnect_attempts += 1
        self.status = ConnectionStatus.RECONNECTING
        
        logger.info(f"🔄 Reconnecting to {self.exchange_name} (attempt {self.reconnect_attempts})")
        
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self.connect()
            # Re-subscribe to all active subscriptions
            for subscription in list(self.subscriptions):
                await self.subscribe(subscription)
            
            self.reconnect_attempts = 0
            logger.info(f"✅ Reconnected to {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"❌ Reconnection failed for {self.exchange_name}: {e}")
            await self._reconnect()

class BinanceConnector(ExchangeConnector):
    """Binance-specific WebSocket connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("binance", config)
        self.base_url = config.get('ws_url', 'wss://stream.binance.com:9443/ws')
        
    async def connect(self):
        """Connect to Binance WebSocket."""
        try:
            self.status = ConnectionStatus.CONNECTING
            self.websocket = await websockets.connect(self.base_url)
            self.status = ConnectionStatus.CONNECTED
            
            # Start message handling
            asyncio.create_task(self._message_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            logger.info("✅ Connected to Binance WebSocket")
            
        except Exception as e:
            logger.error(f"❌ Binance connection error: {e}")
            self.status = ConnectionStatus.ERROR
            await self._reconnect()
    
    async def subscribe(self, subscription: SubscriptionConfig):
        """Subscribe to Binance data feed."""
        try:
            symbol = subscription.symbol.lower()
            
            if subscription.data_type == DataType.TICKER:
                stream = f"{symbol}@ticker"
            elif subscription.data_type == DataType.ORDERBOOK:
                depth = subscription.params.get('depth', 20)
                stream = f"{symbol}@depth{depth}"
            elif subscription.data_type == DataType.TRADES:
                stream = f"{symbol}@trade"
            elif subscription.data_type == DataType.KLINES:
                interval = subscription.params.get('interval', '1m')
                stream = f"{symbol}@kline_{interval}"
            else:
                logger.warning(f"Unsupported data type: {subscription.data_type}")
                return
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            self.subscriptions.add(subscription)
            
            logger.info(f"📡 Subscribed to Binance {subscription.data_type.value} for {subscription.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Binance subscription error: {e}")
    
    async def _message_loop(self):
        """Handle incoming messages."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 Binance WebSocket connection closed")
            await self._reconnect()
        except Exception as e:
            logger.error(f"❌ Binance message loop error: {e}")
            await self._reconnect()
    
    async def _handle_message(self, message: str):
        """Handle Binance WebSocket message."""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if 'result' in data or 'id' in data:
                return
            
            stream = data.get('stream', '')
            event_data = data.get('data', {})
            
            # Determine data type from stream
            if '@ticker' in stream:
                ticker_data = self._parse_ticker(event_data)
                for callback in self.callbacks[DataType.TICKER]:
                    try:
                        await callback(ticker_data)
                    except Exception as e:
                        logger.error(f"Ticker callback error: {e}")
                        
            elif '@depth' in stream:
                orderbook_data = self._parse_orderbook(event_data, stream)
                for callback in self.callbacks[DataType.ORDERBOOK]:
                    try:
                        await callback(orderbook_data)
                    except Exception as e:
                        logger.error(f"Orderbook callback error: {e}")
                        
            elif '@trade' in stream:
                trade_data = self._parse_trade(event_data)
                for callback in self.callbacks[DataType.TRADES]:
                    try:
                        await callback(trade_data)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")
                        
            elif '@kline' in stream:
                kline_data = self._parse_kline(event_data)
                for callback in self.callbacks[DataType.KLINES]:
                    try:
                        await callback(kline_data)
                    except Exception as e:
                        logger.error(f"Kline callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Binance message handling error: {e}")
    
    def _parse_ticker(self, data: Dict) -> TickerData:
        """Parse Binance ticker data."""
        return TickerData(
            symbol=data.get('s', ''),
            exchange='binance',
            price=float(data.get('c', 0)),
            bid=float(data.get('b', 0)),
            ask=float(data.get('a', 0)),
            bid_size=float(data.get('B', 0)),
            ask_size=float(data.get('A', 0)),
            volume_24h=float(data.get('v', 0)),
            price_change_24h=float(data.get('p', 0)),
            price_change_pct_24h=float(data.get('P', 0)),
            high_24h=float(data.get('h', 0)),
            low_24h=float(data.get('l', 0)),
            timestamp=datetime.now()
        )
    
    def _parse_orderbook(self, data: Dict, stream: str) -> OrderBookData:
        """Parse Binance order book data."""
        symbol = stream.split('@')[0].upper()
        
        return OrderBookData(
            symbol=symbol,
            exchange='binance',
            bids=[[float(price), float(size)] for price, size in data.get('b', [])],
            asks=[[float(price), float(size)] for price, size in data.get('a', [])],
            timestamp=datetime.now()
        )
    
    def _parse_trade(self, data: Dict) -> TradeData:
        """Parse Binance trade data."""
        return TradeData(
            symbol=data.get('s', ''),
            exchange='binance',
            trade_id=str(data.get('t', '')),
            price=float(data.get('p', 0)),
            size=float(data.get('q', 0)),
            side='buy' if data.get('m', False) else 'sell',
            timestamp=datetime.fromtimestamp(int(data.get('T', 0)) / 1000)
        )
    
    def _parse_kline(self, data: Dict) -> KlineData:
        """Parse Binance kline data."""
        kline = data.get('k', {})
        
        return KlineData(
            symbol=kline.get('s', ''),
            exchange='binance',
            interval=kline.get('i', ''),
            open_time=datetime.fromtimestamp(int(kline.get('t', 0)) / 1000),
            close_time=datetime.fromtimestamp(int(kline.get('T', 0)) / 1000),
            open_price=float(kline.get('o', 0)),
            high_price=float(kline.get('h', 0)),
            low_price=float(kline.get('l', 0)),
            close_price=float(kline.get('c', 0)),
            volume=float(kline.get('v', 0)),
            is_closed=kline.get('x', False)
        )
    
    async def _heartbeat_loop(self):
        """Maintain connection with periodic heartbeat."""
        while self.status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                # Binance doesn't require explicit heartbeat, but we track timing
                self.last_heartbeat = datetime.now()
            except Exception as e:
                logger.error(f"❌ Binance heartbeat error: {e}")
                break

class BackpackConnector(ExchangeConnector):
    """Backpack-specific WebSocket connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("backpack", config)
        self.base_url = config.get('ws_url', 'wss://ws.backpack.exchange')
        
    async def connect(self):
        """Connect to Backpack WebSocket."""
        try:
            self.status = ConnectionStatus.CONNECTING
            self.websocket = await websockets.connect(self.base_url)
            self.status = ConnectionStatus.CONNECTED
            
            # Start message handling
            asyncio.create_task(self._message_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            logger.info("✅ Connected to Backpack WebSocket")
            
        except Exception as e:
            logger.error(f"❌ Backpack connection error: {e}")
            self.status = ConnectionStatus.ERROR
            await self._reconnect()
    
    async def subscribe(self, subscription: SubscriptionConfig):
        """Subscribe to Backpack data feed."""
        try:
            symbol = subscription.symbol
            
            if subscription.data_type == DataType.TICKER:
                channel = "ticker"
            elif subscription.data_type == DataType.ORDERBOOK:
                channel = "orderbook"
            elif subscription.data_type == DataType.TRADES:
                channel = "trades"
            else:
                logger.warning(f"Unsupported data type for Backpack: {subscription.data_type}")
                return
            
            subscribe_msg = {
                "method": "subscribe",
                "params": {
                    "channel": channel,
                    "symbol": symbol
                },
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            self.subscriptions.add(subscription)
            
            logger.info(f"📡 Subscribed to Backpack {subscription.data_type.value} for {subscription.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Backpack subscription error: {e}")
    
    async def _message_loop(self):
        """Handle incoming messages."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 Backpack WebSocket connection closed")
            await self._reconnect()
        except Exception as e:
            logger.error(f"❌ Backpack message loop error: {e}")
            await self._reconnect()
    
    async def _handle_message(self, message: str):
        """Handle Backpack WebSocket message."""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations and pings
            if data.get('method') in ['subscribed', 'ping']:
                if data.get('method') == 'ping':
                    # Send pong response
                    pong_msg = {"method": "pong", "id": data.get('id')}
                    await self.websocket.send(json.dumps(pong_msg))
                return
            
            channel = data.get('channel', '')
            event_data = data.get('data', {})
            
            # Determine data type from channel
            if channel == 'ticker':
                ticker_data = self._parse_ticker(event_data)
                for callback in self.callbacks[DataType.TICKER]:
                    try:
                        await callback(ticker_data)
                    except Exception as e:
                        logger.error(f"Ticker callback error: {e}")
                        
            elif channel == 'orderbook':
                orderbook_data = self._parse_orderbook(event_data)
                for callback in self.callbacks[DataType.ORDERBOOK]:
                    try:
                        await callback(orderbook_data)
                    except Exception as e:
                        logger.error(f"Orderbook callback error: {e}")
                        
            elif channel == 'trades':
                trade_data = self._parse_trade(event_data)
                for callback in self.callbacks[DataType.TRADES]:
                    try:
                        await callback(trade_data)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Backpack message handling error: {e}")
    
    def _parse_ticker(self, data: Dict) -> TickerData:
        """Parse Backpack ticker data."""
        return TickerData(
            symbol=data.get('symbol', ''),
            exchange='backpack',
            price=float(data.get('lastPrice', 0)),
            bid=float(data.get('bidPrice', 0)),
            ask=float(data.get('askPrice', 0)),
            bid_size=float(data.get('bidSize', 0)),
            ask_size=float(data.get('askSize', 0)),
            volume_24h=float(data.get('volume', 0)),
            price_change_24h=float(data.get('priceChange', 0)),
            price_change_pct_24h=float(data.get('priceChangePercent', 0)),
            high_24h=float(data.get('highPrice', 0)),
            low_24h=float(data.get('lowPrice', 0)),
            timestamp=datetime.now()
        )
    
    def _parse_orderbook(self, data: Dict) -> OrderBookData:
        """Parse Backpack order book data."""
        return OrderBookData(
            symbol=data.get('symbol', ''),
            exchange='backpack',
            bids=[[float(level['price']), float(level['size'])] for level in data.get('bids', [])],
            asks=[[float(level['price']), float(level['size'])] for level in data.get('asks', [])],
            timestamp=datetime.now()
        )
    
    def _parse_trade(self, data: Dict) -> TradeData:
        """Parse Backpack trade data."""
        return TradeData(
            symbol=data.get('symbol', ''),
            exchange='backpack',
            trade_id=str(data.get('id', '')),
            price=float(data.get('price', 0)),
            size=float(data.get('size', 0)),
            side=data.get('side', ''),
            timestamp=datetime.fromtimestamp(int(data.get('timestamp', 0)) / 1000)
        )
    
    async def _heartbeat_loop(self):
        """Maintain connection with periodic heartbeat."""
        while self.status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                ping_msg = {"method": "ping", "id": int(time.time())}
                await self.websocket.send(json.dumps(ping_msg))
                self.last_heartbeat = datetime.now()
            except Exception as e:
                logger.error(f"❌ Backpack heartbeat error: {e}")
                break

class MarketDataFeeder:
    """
    High-performance multi-exchange market data aggregation system.
    Provides synchronized, normalized data feeds for arbitrage detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.connectors = {}
        self.data_buffers = defaultdict(lambda: defaultdict(DataBuffer))
        self.performance_monitor = PerformanceMonitor()
        self.validator = DataValidator()
        self.historical_manager = HistoricalDataManager(
            config.get('db_path', 'market_data.db')
        )
        
        # Subscriptions and callbacks
        self.subscriptions = []
        self.global_callbacks = defaultdict(list)
        
        # Synchronization
        self.sync_enabled = config.get('enable_sync', True)
        self.sync_tolerance_ms = config.get('sync_tolerance_ms', 100)
        self.sync_queues = defaultdict(lambda: defaultdict(queue.Queue))
        
        # Performance settings
        self.enable_historical_storage = config.get('enable_historical_storage', True)
        self.enable_validation = config.get('enable_validation', True)
        self.buffer_size = config.get('buffer_size', 10000)
        
        # Status
        self.is_running = False
        self._tasks = []
        
        logger.info("🎯 Market Data Feeder initialized")
    
    async def initialize(self):
        """Initialize all exchange connectors."""
        try:
            # Initialize Binance connector
            binance_config = self.config.get('binance', {})
            if binance_config.get('enabled', True):
                self.connectors['binance'] = BinanceConnector(binance_config)
                await self.connectors['binance'].connect()
                
                # Add data processing callbacks
                self.connectors['binance'].add_callback(DataType.TICKER, self._process_ticker)
                self.connectors['binance'].add_callback(DataType.ORDERBOOK, self._process_orderbook)
                self.connectors['binance'].add_callback(DataType.TRADES, self._process_trade)
                self.connectors['binance'].add_callback(DataType.KLINES, self._process_kline)
            
            # Initialize Backpack connector
            backpack_config = self.config.get('backpack', {})
            if backpack_config.get('enabled', True):
                self.connectors['backpack'] = BackpackConnector(backpack_config)
                await self.connectors['backpack'].connect()
                
                # Add data processing callbacks
                self.connectors['backpack'].add_callback(DataType.TICKER, self._process_ticker)
                self.connectors['backpack'].add_callback(DataType.ORDERBOOK, self._process_orderbook)
                self.connectors['backpack'].add_callback(DataType.TRADES, self._process_trade)
            
            # Start background tasks
            if self.sync_enabled:
                self._tasks.append(asyncio.create_task(self._sync_processor()))
            
            self._tasks.append(asyncio.create_task(self._performance_monitor_task()))
            
            self.is_running = True
            logger.info("✅ Market Data Feeder initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Market Data Feeder initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all connectors and tasks."""
        try:
            self.is_running = False
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Disconnect all connectors
            for connector in self.connectors.values():
                await connector.disconnect()
            
            # Close historical manager
            if hasattr(self.historical_manager, 'executor'):
                self.historical_manager.executor.shutdown(wait=True)
            
            logger.info("🔒 Market Data Feeder shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    async def subscribe(self, exchange: str, symbol: str, data_type: DataType, 
                       params: Dict[str, Any] = None, callback: Callable = None):
        """Subscribe to market data feed."""
        try:
            if exchange not in self.connectors:
                raise ValueError(f"Exchange {exchange} not available")
            
            subscription = SubscriptionConfig(
                exchange=exchange,
                symbol=symbol,
                data_type=data_type,
                params=params or {},
                callback=callback
            )
            
            await self.connectors[exchange].subscribe(subscription)
            self.subscriptions.append(subscription)
            
            # Add user callback if provided
            if callback:
                self.add_callback(data_type, callback)
            
            logger.info(f"📡 Subscribed to {exchange} {data_type.value} for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Subscription error: {e}")
            raise
    
    async def unsubscribe(self, exchange: str, symbol: str, data_type: DataType):
        """Unsubscribe from market data feed."""
        try:
            # Find and remove subscription
            subscription_to_remove = None
            for subscription in self.subscriptions:
                if (subscription.exchange == exchange and 
                    subscription.symbol == symbol and 
                    subscription.data_type == data_type):
                    subscription_to_remove = subscription
                    break
            
            if subscription_to_remove:
                await self.connectors[exchange].unsubscribe(subscription_to_remove)
                self.subscriptions.remove(subscription_to_remove)
                logger.info(f"📡 Unsubscribed from {exchange} {data_type.value} for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Unsubscription error: {e}")
    
    def add_callback(self, data_type: DataType, callback: Callable):
        """Add global callback for data type."""
        self.global_callbacks[data_type].append(callback)
    
    def remove_callback(self, data_type: DataType, callback: Callable):
        """Remove global callback for data type."""
        if callback in self.global_callbacks[data_type]:
            self.global_callbacks[data_type].remove(callback)
    
    def get_latest_data(self, exchange: str, symbol: str, data_type: DataType, 
                       count: int = 1) -> List[Any]:
        """Get latest data from buffer."""
        try:
            buffer = self.data_buffers[exchange][f"{symbol}:{data_type.value}"]
            return buffer.get_latest(count)
        except Exception as e:
            logger.error(f"❌ Get latest data error: {e}")
            return []
    
    def get_synchronized_data(self, symbol: str, data_type: DataType) -> Dict[str, Any]:
        """Get synchronized data across exchanges."""
        try:
            result = {}
            current_time = datetime.now()
            
            for exchange in self.connectors:
                latest_data = self.get_latest_data(exchange, symbol, data_type, 1)
                if latest_data:
                    data = latest_data[0]
                    # Check if data is recent enough
                    if hasattr(data, 'timestamp'):
                        age_ms = (current_time - data.timestamp).total_seconds() * 1000
                        if age_ms <= self.sync_tolerance_ms:
                            result[exchange] = data
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Get synchronized data error: {e}")
            return {}
    
    def get_performance_stats(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_monitor.get_stats(exchange, symbol)
    
    async def _process_ticker(self, ticker_data: TickerData):
        """Process incoming ticker data."""
        try:
            start_time = time.time()
            
            # Validate data if enabled
            if self.enable_validation:
                if not self.validator.validate_ticker(ticker_data):
                    logger.warning(f"Invalid ticker data rejected: {ticker_data.symbol}")
                    return
            
            # Store in buffer
            buffer_key = f"{ticker_data.symbol}:{DataType.TICKER.value}"
            self.data_buffers[ticker_data.exchange][buffer_key].append(ticker_data)
            
            # Store historical data if enabled
            if self.enable_historical_storage:
                await self.historical_manager.store_ticker(ticker_data)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_message(
                ticker_data.exchange, ticker_data.symbol, latency_ms
            )
            
            # Call global callbacks
            for callback in self.global_callbacks[DataType.TICKER]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(ticker_data)
                    else:
                        callback(ticker_data)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Ticker processing error: {e}")
            self.performance_monitor.record_error(ticker_data.exchange, ticker_data.symbol)
    
    async def _process_orderbook(self, orderbook_data: OrderBookData):
        """Process incoming order book data."""
        try:
            start_time = time.time()
            
            # Validate data if enabled
            if self.enable_validation:
                if not self.validator.validate_orderbook(orderbook_data):
                    logger.warning(f"Invalid orderbook data rejected: {orderbook_data.symbol}")
                    return
            
            # Store in buffer
            buffer_key = f"{orderbook_data.symbol}:{DataType.ORDERBOOK.value}"
            self.data_buffers[orderbook_data.exchange][buffer_key].append(orderbook_data)
            
            # Store historical snapshot if enabled
            if self.enable_historical_storage:
                await self.historical_manager.store_orderbook_snapshot(orderbook_data)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_message(
                orderbook_data.exchange, orderbook_data.symbol, latency_ms
            )
            
            # Call global callbacks
            for callback in self.global_callbacks[DataType.ORDERBOOK]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(orderbook_data)
                    else:
                        callback(orderbook_data)
                except Exception as e:
                    logger.error(f"Orderbook callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Orderbook processing error: {e}")
            self.performance_monitor.record_error(orderbook_data.exchange, orderbook_data.symbol)
    
    async def _process_trade(self, trade_data: TradeData):
        """Process incoming trade data."""
        try:
            start_time = time.time()
            
            # Store in buffer
            buffer_key = f"{trade_data.symbol}:{DataType.TRADES.value}"
            self.data_buffers[trade_data.exchange][buffer_key].append(trade_data)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_message(
                trade_data.exchange, trade_data.symbol, latency_ms
            )
            
            # Call global callbacks
            for callback in self.global_callbacks[DataType.TRADES]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trade_data)
                    else:
                        callback(trade_data)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Trade processing error: {e}")
            self.performance_monitor.record_error(trade_data.exchange, trade_data.symbol)
    
    async def _process_kline(self, kline_data: KlineData):
        """Process incoming kline data."""
        try:
            start_time = time.time()
            
            # Store in buffer
            buffer_key = f"{kline_data.symbol}:{DataType.KLINES.value}"
            self.data_buffers[kline_data.exchange][buffer_key].append(kline_data)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_message(
                kline_data.exchange, kline_data.symbol, latency_ms
            )
            
            # Call global callbacks
            for callback in self.global_callbacks[DataType.KLINES]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(kline_data)
                    else:
                        callback(kline_data)
                except Exception as e:
                    logger.error(f"Kline callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Kline processing error: {e}")
            self.performance_monitor.record_error(kline_data.exchange, kline_data.symbol)
    
    async def _sync_processor(self):
        """Process synchronized data across exchanges."""
        while self.is_running:
            try:
                await asyncio.sleep(1.0)  # Process every second
                
                # TODO: Implement advanced synchronization logic
                # This could include:
                # - Cross-exchange price validation
                # - Latency compensation
                # - Data fusion algorithms
                # - Arbitrage signal generation
                
            except Exception as e:
                logger.error(f"❌ Sync processor error: {e}")
    
    async def _performance_monitor_task(self):
        """Background task to monitor and log performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(60.0)  # Log every minute
                
                stats = self.performance_monitor.get_stats()
                if stats:
                    logger.info("📊 Performance Summary:")
                    for key, stat in stats.items():
                        logger.info(f"  {key}: {stat['message_count']} msgs, "
                                  f"{stat['avg_latency_ms']:.2f}ms avg latency, "
                                  f"{stat['error_count']} errors")
                
            except Exception as e:
                logger.error(f"❌ Performance monitor error: {e}")

# Example usage and testing
async def example_usage():
    """Example of how to use the Market Data Feeder."""
    
    # Configuration
    config = {
        'binance': {
            'enabled': True,
            'ws_url': 'wss://stream.binance.com:9443/ws'
        },
        'backpack': {
            'enabled': True,
            'ws_url': 'wss://ws.backpack.exchange'
        },
        'enable_sync': True,
        'enable_historical_storage': True,
        'enable_validation': True,
        'buffer_size': 10000
    }
    
    # Initialize feeder
    feeder = MarketDataFeeder(config)
    
    # Define callback functions
    async def on_ticker_update(ticker: TickerData):
        print(f"📊 {ticker.exchange} {ticker.symbol}: ${ticker.price:.2f} "
              f"(bid: ${ticker.bid:.2f}, ask: ${ticker.ask:.2f})")
    
    async def on_orderbook_update(orderbook: OrderBookData):
        spread = orderbook.get_spread()
        print(f"📖 {orderbook.exchange} {orderbook.symbol} orderbook: "
              f"spread ${spread:.2f}" if spread else "invalid spread")
    
    try:
        # Initialize feeder
        await feeder.initialize()
        
        # Add global callbacks
        feeder.add_callback(DataType.TICKER, on_ticker_update)
        feeder.add_callback(DataType.ORDERBOOK, on_orderbook_update)
        
        # Subscribe to data feeds
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            # Subscribe to ticker data from both exchanges
            await feeder.subscribe('binance', symbol, DataType.TICKER)
            await feeder.subscribe('backpack', symbol.replace('USDT', 'USDC'), DataType.TICKER)
            
            # Subscribe to order book data
            await feeder.subscribe('binance', symbol, DataType.ORDERBOOK, {'depth': 20})
            await feeder.subscribe('backpack', symbol.replace('USDT', 'USDC'), DataType.ORDERBOOK)
        
        # Run for demonstration
        logger.info("🚀 Market Data Feeder running...")
        await asyncio.sleep(30)  # Run for 30 seconds
        
        # Get performance stats
        stats = feeder.get_performance_stats()
        print("\n📊 Final Performance Stats:")
        for key, stat in stats.items():
            print(f"  {key}: {stat}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Stopping Market Data Feeder...")
    
    finally:
        await feeder.shutdown()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())