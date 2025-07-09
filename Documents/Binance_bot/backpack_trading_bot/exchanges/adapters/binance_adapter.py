#!/usr/bin/env python3
"""
🎯 BINANCE EXCHANGE API ADAPTER v1.0.0
Unified interface for Binance Exchange integration

Based on: https://binance-docs.github.io/apidocs/spot/en/

Features:
- 📡 RESTful API integration
- 🔄 WebSocket real-time data
- 📊 Market data normalization
- 💰 Order management
- 🛡️ Error handling & rate limiting
"""

import aiohttp
import asyncio
import websockets
import json
import hmac
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BinanceOrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class BinanceOrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

@dataclass
class BinanceTicker:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime

@dataclass
class BinanceOrderBook:
    symbol: str
    bids: List[List[float]]  # [price, quantity]
    asks: List[List[float]]  # [price, quantity]
    timestamp: datetime

@dataclass
class BinanceOrder:
    id: str
    symbol: str
    side: BinanceOrderSide
    type: BinanceOrderType
    quantity: float
    price: float
    filled: float
    status: str
    timestamp: datetime

@dataclass
class BinanceBalance:
    asset: str
    free: float
    locked: float
    total: float

class BinanceAdapter:
    """
    Unified adapter for Binance Exchange API.
    Provides normalized interface compatible with existing multi-exchange system.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.rest_base_url = "https://testnet.binance.vision"
            self.ws_base_url = "wss://testnet.binance.vision"
        else:
            self.rest_base_url = "https://api.binance.com"
            self.ws_base_url = "wss://stream.binance.com:9443"
        
        # Session for HTTP requests
        self.session = None
        
        # WebSocket connections
        self.ws_connections = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.weight_used = 0
        self.weight_reset_time = 0
        
        logger.info(f"🎯 Binance adapter initialized (testnet: {testnet})")
    
    async def initialize(self):
        """Initialize the adapter and connections."""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connectivity
            await self.test_connectivity()
            
            logger.info("✅ Binance adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Binance adapter initialization failed: {e}")
            raise
    
    async def close(self):
        """Close all connections."""
        if self.session:
            await self.session.close()
        
        for ws in self.ws_connections.values():
            await ws.close()
        
        logger.info("🔒 Binance adapter closed")
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC signature for authenticated requests."""
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           authenticated: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Binance API."""
        await self._rate_limit()
        
        url = f"{self.rest_base_url}{endpoint}"
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        
        if params is None:
            params = {}
        
        if authenticated:
            timestamp = int(time.time() * 1000)
            params["timestamp"] = timestamp
            
            # Create query string for signature
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params["signature"] = signature
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers
            ) as response:
                
                # Update rate limit tracking
                if "X-MBX-USED-WEIGHT-1M" in response.headers:
                    self.weight_used = int(response.headers["X-MBX-USED-WEIGHT-1M"])
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Binance API error: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            raise
    
    # ========== PUBLIC API METHODS ==========
    
    async def test_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            response = await self._make_request("GET", "/api/v3/ping")
            logger.info("✅ Binance connectivity test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Binance connectivity test failed: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> BinanceTicker:
        """Get 24hr ticker price statistics."""
        try:
            # Get 24hr ticker
            ticker_response = await self._make_request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})
            
            # Get current price
            price_response = await self._make_request("GET", "/api/v3/ticker/price", {"symbol": symbol})
            
            # Get order book ticker for bid/ask
            book_response = await self._make_request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
            
            return BinanceTicker(
                symbol=symbol,
                price=float(price_response.get("price", 0)),
                bid=float(book_response.get("bidPrice", 0)),
                ask=float(book_response.get("askPrice", 0)),
                volume=float(ticker_response.get("volume", 0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get ticker for {symbol}: {e}")
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> BinanceOrderBook:
        """Get order book for a symbol."""
        try:
            params = {"symbol": symbol, "limit": limit}
            response = await self._make_request("GET", "/api/v3/depth", params)
            
            return BinanceOrderBook(
                symbol=symbol,
                bids=[[float(price), float(qty)] for price, qty in response.get("bids", [])],
                asks=[[float(price), float(qty)] for price, qty in response.get("asks", [])],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get orderbook for {symbol}: {e}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get kline/candlestick data."""
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = await self._make_request("GET", "/api/v3/klines", params)
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get klines for {symbol}: {e}")
            raise
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        try:
            response = await self._make_request("GET", "/api/v3/exchangeInfo")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get exchange info: {e}")
            raise
    
    # ========== AUTHENTICATED API METHODS ==========
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            response = await self._make_request("GET", "/api/v3/account", authenticated=True)
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            raise
    
    async def get_balances(self) -> List[BinanceBalance]:
        """Get account balances."""
        try:
            account_info = await self.get_account_info()
            balances = []
            
            for balance_data in account_info.get("balances", []):
                free = float(balance_data.get("free", 0))
                locked = float(balance_data.get("locked", 0))
                
                # Only include balances with non-zero amounts
                if free > 0 or locked > 0:
                    balances.append(BinanceBalance(
                        asset=balance_data.get("asset"),
                        free=free,
                        locked=locked,
                        total=free + locked
                    ))
            
            return balances
            
        except Exception as e:
            logger.error(f"❌ Failed to get balances: {e}")
            raise
    
    async def place_order(self, symbol: str, side: BinanceOrderSide, 
                         order_type: BinanceOrderType, quantity: float,
                         price: float = None, time_in_force: str = "GTC",
                         stop_price: float = None) -> BinanceOrder:
        """Place a new order."""
        try:
            order_data = {
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": str(quantity),
                "timeInForce": time_in_force
            }
            
            # Add price for limit orders
            if order_type in [BinanceOrderType.LIMIT, BinanceOrderType.STOP_LOSS_LIMIT, 
                            BinanceOrderType.TAKE_PROFIT_LIMIT] and price:
                order_data["price"] = str(price)
            
            # Add stop price for stop orders
            if order_type in [BinanceOrderType.STOP_LOSS, BinanceOrderType.STOP_LOSS_LIMIT,
                            BinanceOrderType.TAKE_PROFIT, BinanceOrderType.TAKE_PROFIT_LIMIT] and stop_price:
                order_data["stopPrice"] = str(stop_price)
            
            # Remove timeInForce for market orders
            if order_type == BinanceOrderType.MARKET:
                del order_data["timeInForce"]
            
            response = await self._make_request("POST", "/api/v3/order", order_data, authenticated=True)
            
            return BinanceOrder(
                id=str(response.get("orderId")),
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=float(response.get("price", price or 0)),
                filled=float(response.get("executedQty", 0)),
                status=response.get("status"),
                timestamp=datetime.fromtimestamp(int(response.get("transactTime", 0)) / 1000)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            await self._make_request("DELETE", "/api/v3/order", params, authenticated=True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> BinanceOrder:
        """Get order status."""
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            response = await self._make_request("GET", "/api/v3/order", params, authenticated=True)
            
            return BinanceOrder(
                id=str(response.get("orderId")),
                symbol=response.get("symbol"),
                side=BinanceOrderSide(response.get("side")),
                type=BinanceOrderType(response.get("type")),
                quantity=float(response.get("origQty", 0)),
                price=float(response.get("price", 0)),
                filled=float(response.get("executedQty", 0)),
                status=response.get("status"),
                timestamp=datetime.fromtimestamp(int(response.get("time", 0)) / 1000)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[BinanceOrder]:
        """Get all open orders."""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            
            response = await self._make_request("GET", "/api/v3/openOrders", params, authenticated=True)
            
            orders = []
            for order_data in response:
                orders.append(BinanceOrder(
                    id=str(order_data.get("orderId")),
                    symbol=order_data.get("symbol"),
                    side=BinanceOrderSide(order_data.get("side")),
                    type=BinanceOrderType(order_data.get("type")),
                    quantity=float(order_data.get("origQty", 0)),
                    price=float(order_data.get("price", 0)),
                    filled=float(order_data.get("executedQty", 0)),
                    status=order_data.get("status"),
                    timestamp=datetime.fromtimestamp(int(order_data.get("time", 0)) / 1000)
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Failed to get open orders: {e}")
            raise
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        try:
            params = {"symbol": symbol}
            await self._make_request("DELETE", "/api/v3/openOrders", params, authenticated=True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel all orders for {symbol}: {e}")
            return False
    
    # ========== WEBSOCKET METHODS ==========
    
    async def start_ticker_stream(self, symbol: str, callback: Callable[[BinanceTicker], None]):
        """Start real-time ticker stream."""
        try:
            ws_url = f"{self.ws_base_url}/ws/{symbol.lower()}@ticker"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"📡 Started ticker stream for {symbol}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        ticker = BinanceTicker(
                            symbol=symbol,
                            price=float(data.get("c", 0)),
                            bid=float(data.get("b", 0)),
                            ask=float(data.get("a", 0)),
                            volume=float(data.get("v", 0)),
                            timestamp=datetime.now()
                        )
                        
                        callback(ticker)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing ticker data: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Ticker stream error: {e}")
            raise
    
    async def start_orderbook_stream(self, symbol: str, callback: Callable[[BinanceOrderBook], None]):
        """Start real-time order book stream."""
        try:
            ws_url = f"{self.ws_base_url}/ws/{symbol.lower()}@depth"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"📊 Started orderbook stream for {symbol}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        orderbook = BinanceOrderBook(
                            symbol=symbol,
                            bids=[[float(price), float(qty)] for price, qty in data.get("b", [])],
                            asks=[[float(price), float(qty)] for price, qty in data.get("a", [])],
                            timestamp=datetime.now()
                        )
                        
                        callback(orderbook)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing orderbook data: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Orderbook stream error: {e}")
            raise
    
    async def start_kline_stream(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """Start real-time kline/candlestick stream."""
        try:
            ws_url = f"{self.ws_base_url}/ws/{symbol.lower()}@kline_{interval}"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"📈 Started kline stream for {symbol} ({interval})")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        kline_data = data.get("k", {})
                        
                        if kline_data.get("x"):  # Only process closed klines
                            callback(kline_data)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing kline data: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Kline stream error: {e}")
            raise
    
    async def start_user_data_stream(self, callback: Callable[[Dict], None]):
        """Start user data stream for account updates."""
        try:
            # Get listen key for user data stream
            listen_key_response = await self._make_request("POST", "/api/v3/userDataStream", authenticated=True)
            listen_key = listen_key_response.get("listenKey")
            
            if not listen_key:
                raise Exception("Failed to get listen key for user data stream")
            
            ws_url = f"{self.ws_base_url}/ws/{listen_key}"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("👤 Started user data stream")
                
                # Keep alive task
                async def keep_alive():
                    while True:
                        try:
                            await asyncio.sleep(1800)  # 30 minutes
                            await self._make_request("PUT", "/api/v3/userDataStream", 
                                                   {"listenKey": listen_key}, authenticated=True)
                        except Exception as e:
                            logger.error(f"❌ Failed to keep alive user data stream: {e}")
                
                keep_alive_task = asyncio.create_task(keep_alive())
                
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            callback(data)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing user data: {e}")
                            
                finally:
                    keep_alive_task.cancel()
                    
        except Exception as e:
            logger.error(f"❌ User data stream error: {e}")
            raise

# Example usage
async def example_usage():
    """Example of how to use the Binance adapter."""
    # Initialize adapter
    adapter = BinanceAdapter(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True
    )
    
    try:
        await adapter.initialize()
        
        # Test public API
        ticker = await adapter.get_ticker("BTCUSDT")
        print(f"BTC Price: ${ticker.price:.2f}")
        
        # Test exchange info
        exchange_info = await adapter.get_exchange_info()
        print(f"Exchange symbols: {len(exchange_info.get('symbols', []))}")
        
        # Test private API (uncomment when ready)
        # balances = await adapter.get_balances()
        # print(f"Balances: {balances}")
        
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(example_usage())