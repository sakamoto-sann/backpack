#!/usr/bin/env python3
"""
Triangular Arbitrage Bot for Binance
Real-time WebSocket-based arbitrage detection and execution
"""

import asyncio
import json
import logging
import logging.handlers
import time
from collections import defaultdict, deque
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
import websockets
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class RateLimiter:
    """Advanced rate limiter for Binance API compliance"""
    
    def __init__(self):
        self.request_timestamps = deque()
        self.order_timestamps_10s = deque()
        self.order_timestamps_1m = deque()
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, is_order: bool = False):
        """Wait if necessary to comply with rate limits"""
        async with self.lock:
            current_time = time.time()
            
            # Clean old timestamps
            self._clean_old_timestamps(current_time)
            
            # Check request rate limit (1000 per minute)
            if len(self.request_timestamps) >= 1000:
                sleep_time = 60 - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    self._clean_old_timestamps(time.time())
            
            # Check order rate limits if this is an order
            if is_order:
                if len(self.order_timestamps_10s) >= 100:
                    sleep_time = 10 - (current_time - self.order_timestamps_10s[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                
                if len(self.order_timestamps_1m) >= 1000:
                    sleep_time = 60 - (current_time - self.order_timestamps_1m[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Record this request/order
            current_time = time.time()
            self.request_timestamps.append(current_time)
            if is_order:
                self.order_timestamps_10s.append(current_time)
                self.order_timestamps_1m.append(current_time)
    
    def _clean_old_timestamps(self, current_time: float):
        """Remove timestamps older than their respective windows"""
        # Clean 1-minute request window
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Clean 10-second order window
        while self.order_timestamps_10s and current_time - self.order_timestamps_10s[0] > 10:
            self.order_timestamps_10s.popleft()
        
        # Clean 1-minute order window
        while self.order_timestamps_1m and current_time - self.order_timestamps_1m[0] > 60:
            self.order_timestamps_1m.popleft()

class DataStore:
    """Manages real-time price data and exchange information"""
    
    def __init__(self):
        self.prices: Dict[str, Dict] = {}
        self.symbol_info: Dict[str, Dict] = {}
        self.last_update: Dict[str, float] = {}
        self.lock = asyncio.Lock()
    
    async def update_price(self, symbol: str, bid: float, ask: float):
        """Update price data for a symbol"""
        async with self.lock:
            self.prices[symbol] = {
                'bid': bid,
                'ask': ask,
                'timestamp': time.time()
            }
            self.last_update[symbol] = time.time()
    
    async def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol"""
        async with self.lock:
            return self.prices.get(symbol)
    
    async def is_price_fresh(self, symbol: str, max_age: float = 5.0) -> bool:
        """Check if price data is fresh (within max_age seconds)"""
        async with self.lock:
            if symbol not in self.last_update:
                return False
            return time.time() - self.last_update[symbol] < max_age

class ArbitrageBot:
    """Main triangular arbitrage bot class"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.data_store = DataStore()
        self.rate_limiter = RateLimiter()
        self.is_running = False
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_trade_time = 0
        
        # Trading pairs for triangular arbitrage
        self.trading_pairs = [
            ('BTCUSDT', 'ETHBTC', 'ETHUSDT'),
            ('BTCUSDT', 'ADABTC', 'ADAUSDT'),
            ('BTCUSDT', 'BNBBTC', 'BNBUSDT'),
            ('ETHUSDT', 'ADAETH', 'ADAUSDT'),
            ('ETHUSDT', 'BNBETH', 'BNBUSDT'),
        ]
        
        # Initialize telegram notifier if configured
        self.telegram = None
        if hasattr(config, 'TELEGRAM_BOT_TOKEN') and config.TELEGRAM_BOT_TOKEN:
            self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    
    async def initialize(self):
        """Initialize the bot"""
        try:
            self.client = await AsyncClient.create(self.config.API_KEY, self.config.API_SECRET)
            
            # Get account info
            account_info = await self.client.get_account()
            logger.info(f"Account initialized. Trading status: {account_info.get('canTrade', False)}")
            
            # Load symbol information
            await self._load_symbol_info()
            
            logger.info("Bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    async def _load_symbol_info(self):
        """Load symbol information from exchange"""
        try:
            await self.rate_limiter.wait_if_needed()
            exchange_info = await self.client.get_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                
                # Extract relevant filters
                filters = {}
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        filters['minQty'] = float(filter_info['minQty'])
                        filters['maxQty'] = float(filter_info['maxQty'])
                        filters['stepSize'] = float(filter_info['stepSize'])
                    elif filter_info['filterType'] == 'PRICE_FILTER':
                        filters['minPrice'] = float(filter_info['minPrice'])
                        filters['maxPrice'] = float(filter_info['maxPrice'])
                        filters['tickSize'] = float(filter_info['tickSize'])
                    elif filter_info['filterType'] == 'MIN_NOTIONAL':
                        filters['minNotional'] = float(filter_info['minNotional'])
                
                self.data_store.symbol_info[symbol] = {
                    'status': symbol_info['status'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'filters': filters
                }
            
            logger.info(f"Loaded information for {len(self.data_store.symbol_info)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading symbol info: {e}")
    
    async def start_websocket(self):
        """Start WebSocket connection for real-time price updates"""
        try:
            # Create stream names for all symbols
            symbols = set()
            for pair in self.trading_pairs:
                symbols.update(pair)
            
            streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            logger.info(f"Starting WebSocket connection for {len(symbols)} symbols")
            
            async with websockets.connect(stream_url) as websocket:
                self.is_running = True
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Handle ticker updates
                        if 'stream' in data:
                            symbol = data['data']['s']
                            bid = float(data['data']['b'])
                            ask = float(data['data']['a'])
                            
                            await self.data_store.update_price(symbol, bid, ask)
                            
                            # Check for arbitrage opportunities
                            await self._check_arbitrage_opportunities()
                        
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket timeout, continuing...")
                        continue
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    
    async def _check_arbitrage_opportunities(self):
        """Check for triangular arbitrage opportunities"""
        try:
            for pair_set in self.trading_pairs:
                opportunity = await self._calculate_arbitrage_profit(pair_set)
                
                if opportunity and opportunity['profit_pct'] > self.config.MIN_PROFIT_THRESHOLD:
                    await self._execute_arbitrage(opportunity)
                    
        except Exception as e:
            logger.error(f"Error checking arbitrage opportunities: {e}")
    
    async def _calculate_arbitrage_profit(self, pair_set: Tuple[str, str, str]) -> Optional[Dict]:
        """Calculate potential profit from triangular arbitrage"""
        try:
            symbol_a, symbol_b, symbol_c = pair_set
            
            # Get current prices
            price_a = await self.data_store.get_price(symbol_a)
            price_b = await self.data_store.get_price(symbol_b)
            price_c = await self.data_store.get_price(symbol_c)
            
            if not all([price_a, price_b, price_c]):
                return None
            
            # Check if prices are fresh
            for symbol in pair_set:
                if not await self.data_store.is_price_fresh(symbol):
                    return None
            
            # Calculate arbitrage paths
            # Path 1: A -> B -> C -> A
            path1_result = self._calculate_path_profit(
                price_a['ask'], price_b['bid'], price_c['ask'], 
                symbol_a, symbol_b, symbol_c
            )
            
            # Path 2: A -> C -> B -> A
            path2_result = self._calculate_path_profit(
                price_a['ask'], price_c['bid'], price_b['ask'],
                symbol_a, symbol_c, symbol_b
            )
            
            # Return the more profitable path
            if path1_result and path2_result:
                if path1_result['profit_pct'] > path2_result['profit_pct']:
                    return path1_result
                else:
                    return path2_result
            elif path1_result:
                return path1_result
            elif path2_result:
                return path2_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage profit: {e}")
            return None
    
    def _calculate_path_profit(self, price1: float, price2: float, price3: float,
                              symbol1: str, symbol2: str, symbol3: str) -> Optional[Dict]:
        """Calculate profit for a specific arbitrage path"""
        try:
            # Start with base amount
            base_amount = self.config.TRADE_AMOUNT
            
            # Calculate quantities for each step
            step1_result = base_amount / price1
            step2_result = step1_result * price2
            step3_result = step2_result / price3
            
            # Calculate profit
            profit = step3_result - base_amount
            profit_pct = (profit / base_amount) * 100
            
            # Account for trading fees (0.1% per trade)
            fee_cost = base_amount * 0.001 * 3  # 3 trades
            net_profit = profit - fee_cost
            net_profit_pct = (net_profit / base_amount) * 100
            
            return {
                'path': [symbol1, symbol2, symbol3],
                'prices': [price1, price2, price3],
                'profit': profit,
                'profit_pct': profit_pct,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct,
                'base_amount': base_amount,
                'final_amount': step3_result
            }
            
        except Exception as e:
            logger.error(f"Error calculating path profit: {e}")
            return None
    
    async def _execute_arbitrage(self, opportunity: Dict):
        """Execute triangular arbitrage trade"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
                logger.warning("Daily loss limit reached, skipping trade")
                return
            
            # Check minimum time between trades
            if time.time() - self.last_trade_time < self.config.MIN_TRADE_INTERVAL:
                return
            
            # Check if we have sufficient balance
            if not await self._check_balance(opportunity):
                return
            
            logger.info(f"Executing arbitrage: {opportunity}")
            
            # Execute trades in sequence
            success = await self._execute_trade_sequence(opportunity)
            
            if success:
                self.trade_count += 1
                self.daily_pnl += opportunity['net_profit']
                self.last_trade_time = time.time()
                
                # Send notification
                if self.telegram:
                    await self.telegram.send_trade_notification(opportunity)
                
                logger.info(f"Arbitrage executed successfully. Daily P&L: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
    
    async def _check_balance(self, opportunity: Dict) -> bool:
        """Check if we have sufficient balance for the trade"""
        try:
            await self.rate_limiter.wait_if_needed()
            account_info = await self.client.get_account()
            
            # Find the base asset balance
            base_asset = opportunity['path'][0].replace('USDT', '')
            if base_asset == opportunity['path'][0]:
                base_asset = 'USDT'
            
            for balance in account_info['balances']:
                if balance['asset'] == base_asset:
                    free_balance = float(balance['free'])
                    required_balance = opportunity['base_amount']
                    
                    if free_balance >= required_balance:
                        return True
                    else:
                        logger.warning(f"Insufficient balance. Required: {required_balance}, Available: {free_balance}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return False
    
    async def _execute_trade_sequence(self, opportunity: Dict) -> bool:
        """Execute the sequence of trades for arbitrage"""
        try:
            path = opportunity['path']
            base_amount = opportunity['base_amount']
            
            # Execute each trade in sequence
            current_amount = base_amount
            
            for i, symbol in enumerate(path):
                if i == len(path) - 1:  # Last trade
                    break
                
                # Determine trade parameters
                next_symbol = path[i + 1] if i + 1 < len(path) else path[0]
                
                # Execute trade
                result = await self._execute_single_trade(symbol, current_amount, i == 0)
                
                if not result:
                    logger.error(f"Trade {i+1} failed for symbol {symbol}")
                    return False
                
                current_amount = result['executed_qty']
                
                # Small delay between trades
                await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade sequence: {e}")
            return False
    
    async def _execute_single_trade(self, symbol: str, quantity: float, is_first_trade: bool) -> Optional[Dict]:
        """Execute a single trade with proper error handling"""
        try:
            await self.rate_limiter.wait_if_needed(is_order=True)
            
            # Get symbol info for quantity precision
            symbol_info = self.data_store.symbol_info.get(symbol)
            if not symbol_info:
                logger.error(f"No symbol info for {symbol}")
                return None
            
            # Round quantity to proper precision
            step_size = symbol_info['filters']['stepSize']
            quantity = self._round_to_precision(quantity, step_size)
            
            # Place market order
            order = await self.client.create_order(
                symbol=symbol,
                side='BUY' if is_first_trade else 'SELL',
                type='MARKET',
                quantity=quantity,
                timeInForce='IOC'  # Immediate or Cancel
            )
            
            logger.info(f"Order executed: {order['orderId']} for {symbol}")
            
            return {
                'order_id': order['orderId'],
                'symbol': symbol,
                'executed_qty': float(order['executedQty']),
                'price': float(order['price']) if order['price'] else None
            }
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error executing trade: {e}")
            return None
        except Exception as e:
            logger.error(f"Error executing single trade: {e}")
            return None
    
    def _round_to_precision(self, value: float, step_size: float) -> float:
        """Round value to proper precision based on step size"""
        if step_size == 0:
            return value
        
        precision = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
        return round(value, precision)
    
    async def stop(self):
        """Stop the bot"""
        self.is_running = False
        if self.client:
            await self.client.close_connection()
        logger.info("Bot stopped")
    
    async def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'last_trade_time': self.last_trade_time
        }

# Placeholder for TelegramNotifier
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
    
    async def send_trade_notification(self, opportunity: Dict):
        """Send trade notification via Telegram"""
        try:
            message = f"🤖 Arbitrage Trade Executed\n"
            message += f"Path: {' -> '.join(opportunity['path'])}\n"
            message += f"Profit: ${opportunity['net_profit']:.2f} ({opportunity['net_profit_pct']:.2f}%)\n"
            message += f"Amount: ${opportunity['base_amount']:.2f}"
            
            logger.info(f"Telegram notification: {message}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")