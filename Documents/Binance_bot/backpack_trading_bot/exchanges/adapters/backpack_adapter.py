#!/usr/bin/env python3
"""
🎯 BACKPACK EXCHANGE API ADAPTER v1.0.0
Unified interface for Backpack Exchange integration

Based on: https://support.backpack.exchange/api-docs

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

logger = logging.getLogger(__name__)

class BackpackOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class BackpackOrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"

@dataclass
class BackpackTicker:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime

@dataclass
class BackpackOrderBook:
    symbol: str
    bids: List[List[float]]  # [price, quantity]
    asks: List[List[float]]  # [price, quantity]
    timestamp: datetime

@dataclass
class BackpackOrder:
    id: str
    symbol: str
    side: BackpackOrderSide
    type: BackpackOrderType
    quantity: float
    price: float
    filled: float
    status: str
    timestamp: datetime

@dataclass
class BackpackBalance:
    asset: str
    free: float
    locked: float
    total: float

class BackpackAdapter:
    """
    Unified adapter for Backpack Exchange API.
    Provides normalized interface compatible with existing Binance bot.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.rest_base_url = "https://api.backpack.exchange"  # Update when testnet available
            self.ws_base_url = "wss://ws.backpack.exchange"
        else:
            self.rest_base_url = "https://api.backpack.exchange"
            self.ws_base_url = "wss://ws.backpack.exchange"
        
        # Session for HTTP requests
        self.session = None
        
        # WebSocket connections
        self.ws_connections = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        logger.info(f"🎯 Backpack adapter initialized (testnet: {testnet})")
    
    async def initialize(self):
        """Initialize the adapter and connections."""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connectivity
            await self.test_connectivity()
            
            logger.info("✅ Backpack adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Backpack adapter initialization failed: {e}")
            raise
    
    async def close(self):
        """Close all connections."""
        if self.session:
            await self.session.close()
        
        for ws in self.ws_connections.values():
            await ws.close()
        
        logger.info("🔒 Backpack adapter closed")
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC signature for authenticated requests."""
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
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
                           data: Dict = None, authenticated: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Backpack API."""
        await self._rate_limit()
        
        url = f"{self.rest_base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json"
        }
        
        if authenticated:
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(data) if data else ""
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            headers.update({
                "X-API-Key": self.api_key,
                "X-Timestamp": timestamp,
                "X-Signature": signature
            })
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Backpack API error: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            raise
    
    # ========== PUBLIC API METHODS ==========
    
    async def test_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            response = await self._make_request("GET", "/api/v1/status")
            logger.info("✅ Backpack connectivity test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Backpack connectivity test failed: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> BackpackTicker:
        """Get 24hr ticker price statistics."""
        try:
            response = await self._make_request("GET", f"/api/v1/ticker", {"symbol": symbol})
            
            return BackpackTicker(
                symbol=symbol,
                price=float(response.get("price", 0)),
                bid=float(response.get("bid", 0)),
                ask=float(response.get("ask", 0)),
                volume=float(response.get("volume", 0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get ticker for {symbol}: {e}")
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> BackpackOrderBook:
        """Get order book for a symbol."""
        try:
            params = {"symbol": symbol, "limit": limit}
            response = await self._make_request("GET", "/api/v1/depth", params)
            
            return BackpackOrderBook(
                symbol=symbol,
                bids=[[float(price), float(qty)] for price, qty in response.get("bids", [])],
                asks=[[float(price), float(qty)] for price, qty in response.get("asks", [])],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get orderbook for {symbol}: {e}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict]:
        """Get kline/candlestick data."""
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = await self._make_request("GET", "/api/v1/klines", params)
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get klines for {symbol}: {e}")
            raise
    
    # ========== AUTHENTICATED API METHODS ==========
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            response = await self._make_request("GET", "/api/v1/account", authenticated=True)
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            raise
    
    async def get_balances(self) -> List[BackpackBalance]:
        """Get account balances."""
        try:
            account_info = await self.get_account_info()
            balances = []
            
            for balance_data in account_info.get("balances", []):
                balances.append(BackpackBalance(
                    asset=balance_data.get("asset"),
                    free=float(balance_data.get("free", 0)),
                    locked=float(balance_data.get("locked", 0)),
                    total=float(balance_data.get("free", 0)) + float(balance_data.get("locked", 0))
                ))
            
            return balances
            
        except Exception as e:
            logger.error(f"❌ Failed to get balances: {e}")
            raise
    
    async def place_order(self, symbol: str, side: BackpackOrderSide, 
                         order_type: BackpackOrderType, quantity: float,
                         price: float = None, time_in_force: str = "GTC") -> BackpackOrder:
        """Place a new order."""
        try:
            order_data = {
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": str(quantity),
                "timeInForce": time_in_force
            }
            
            if order_type == BackpackOrderType.LIMIT and price:
                order_data["price"] = str(price)
            
            response = await self._make_request("POST", "/api/v1/order", data=order_data, authenticated=True)
            
            return BackpackOrder(
                id=response.get("orderId"),
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price or 0,
                filled=0,
                status=response.get("status"),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            data = {
                "symbol": symbol,
                "orderId": order_id
            }
            await self._make_request("DELETE", "/api/v1/order", data=data, authenticated=True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> BackpackOrder:
        """Get order status."""
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            response = await self._make_request("GET", "/api/v1/order", params=params, authenticated=True)
            
            return BackpackOrder(
                id=response.get("orderId"),
                symbol=response.get("symbol"),
                side=BackpackOrderSide(response.get("side")),
                type=BackpackOrderType(response.get("type")),
                quantity=float(response.get("origQty", 0)),
                price=float(response.get("price", 0)),
                filled=float(response.get("executedQty", 0)),
                status=response.get("status"),
                timestamp=datetime.fromtimestamp(int(response.get("time", 0)) / 1000)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            raise
    
    # ========== WEBSOCKET METHODS ==========
    
    async def start_ticker_stream(self, symbol: str, callback: Callable[[BackpackTicker], None]):
        """Start real-time ticker stream."""
        try:
            ws_url = f"{self.ws_base_url}/ws/{symbol.lower()}@ticker"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"📡 Started ticker stream for {symbol}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        ticker = BackpackTicker(
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
    
    async def start_orderbook_stream(self, symbol: str, callback: Callable[[BackpackOrderBook], None]):
        """Start real-time order book stream."""
        try:
            ws_url = f"{self.ws_base_url}/ws/{symbol.lower()}@depth"
            
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"📊 Started orderbook stream for {symbol}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        orderbook = BackpackOrderBook(
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

# Example usage
async def example_usage():
    """Example of how to use the Backpack adapter."""
    # Initialize adapter
    adapter = BackpackAdapter(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True
    )
    
    try:
        await adapter.initialize()
        
        # Test public API
        ticker = await adapter.get_ticker("BTCUSDC")
        print(f"BTC Price: ${ticker.price:.2f}")
        
        # Test private API (uncomment when ready)
        # balances = await adapter.get_balances()
        # print(f"Balances: {balances}")
        
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(example_usage())