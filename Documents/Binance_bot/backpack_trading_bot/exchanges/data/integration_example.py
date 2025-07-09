#!/usr/bin/env python3
"""
🎯 MARKET DATA FEEDER INTEGRATION EXAMPLE v1.0.0
Demonstration of Market Data Feeder integration with existing trading system

This example shows how to:
- Initialize the Market Data Feeder
- Connect to Arbitrage Detector
- Process real-time market data
- Detect arbitrage opportunities
- Monitor performance metrics
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.append('..')
sys.path.append('../strategies')

from market_data_feeder import (
    MarketDataFeeder,
    DataType,
    TickerData,
    OrderBookData
)

# Import arbitrage detector (if available)
try:
    from strategies.arbitrage_detector import ArbitrageDetector
    ARBITRAGE_AVAILABLE = True
except ImportError:
    ARBITRAGE_AVAILABLE = False
    logging.warning("Arbitrage detector not available for integration")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemIntegration:
    """Integration layer between Market Data Feeder and Trading System."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Market Data Feeder
        self.data_feeder = MarketDataFeeder(config.get('market_data_feeder', {}))
        
        # Initialize Arbitrage Detector if available
        self.arbitrage_detector = None
        if ARBITRAGE_AVAILABLE:
            self.arbitrage_detector = ArbitrageDetector(config.get('arbitrage_detector', {}))
        
        # Tracked symbols
        self.symbols = config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        
        # Statistics
        self.stats = {
            'tickers_processed': 0,
            'orderbooks_processed': 0,
            'arbitrage_opportunities': 0,
            'start_time': None
        }
        
        logger.info("🔗 Trading System Integration initialized")
    
    async def initialize(self):
        """Initialize all components."""
        try:
            self.stats['start_time'] = datetime.now()
            
            # Initialize Market Data Feeder
            await self.data_feeder.initialize()
            
            # Set up data callbacks
            self.data_feeder.add_callback(DataType.TICKER, self._on_ticker_update)
            self.data_feeder.add_callback(DataType.ORDERBOOK, self._on_orderbook_update)
            
            # Subscribe to market data
            await self._setup_subscriptions()
            
            logger.info("✅ Trading System Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all components."""
        try:
            await self.data_feeder.shutdown()
            
            # Print final statistics
            self._print_final_stats()
            
            logger.info("🔒 Trading System Integration shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    async def _setup_subscriptions(self):
        """Set up market data subscriptions."""
        for symbol in self.symbols:
            try:
                # Subscribe to Binance data
                await self.data_feeder.subscribe(
                    exchange='binance',
                    symbol=symbol,
                    data_type=DataType.TICKER
                )
                
                await self.data_feeder.subscribe(
                    exchange='binance',
                    symbol=symbol,
                    data_type=DataType.ORDERBOOK,
                    params={'depth': 20}
                )
                
                # Subscribe to Backpack data (adjust symbol format)
                backpack_symbol = symbol.replace('USDT', 'USDC')
                await self.data_feeder.subscribe(
                    exchange='backpack',
                    symbol=backpack_symbol,
                    data_type=DataType.TICKER
                )
                
                await self.data_feeder.subscribe(
                    exchange='backpack',
                    symbol=backpack_symbol,
                    data_type=DataType.ORDERBOOK
                )
                
                logger.info(f"📡 Subscribed to data feeds for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Subscription error for {symbol}: {e}")
    
    async def _on_ticker_update(self, ticker: TickerData):
        """Handle ticker data updates."""
        try:
            self.stats['tickers_processed'] += 1
            
            # Log ticker update (throttled)
            if self.stats['tickers_processed'] % 10 == 0:
                logger.info(f"📊 {ticker.exchange} {ticker.symbol}: "
                          f"${ticker.price:.2f} (spread: ${ticker.ask - ticker.bid:.4f})")
            
            # Update arbitrage detector if available
            if self.arbitrage_detector:
                await self.arbitrage_detector.update_market_data(
                    ticker.exchange,
                    ticker.symbol,
                    {
                        'price': ticker.price,
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'volume': ticker.volume_24h
                    }
                )
                
                # Check for arbitrage opportunities (every 10th update to avoid spam)
                if self.stats['tickers_processed'] % 10 == 0:
                    await self._check_arbitrage_opportunities(ticker.symbol)
            
        except Exception as e:
            logger.error(f"❌ Ticker update error: {e}")
    
    async def _on_orderbook_update(self, orderbook: OrderBookData):
        """Handle order book data updates."""
        try:
            self.stats['orderbooks_processed'] += 1
            
            # Log order book update (throttled)
            if self.stats['orderbooks_processed'] % 20 == 0:
                spread = orderbook.get_spread()
                best_bid = orderbook.get_best_bid()
                best_ask = orderbook.get_best_ask()
                
                logger.info(f"📖 {orderbook.exchange} {orderbook.symbol} orderbook: "
                          f"bid=${best_bid:.2f}, ask=${best_ask:.2f}, spread=${spread:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Orderbook update error: {e}")
    
    async def _check_arbitrage_opportunities(self, symbol: str):
        """Check for arbitrage opportunities for a symbol."""
        try:
            if not self.arbitrage_detector:
                return
            
            # Detect price arbitrage
            opportunity = await self.arbitrage_detector.detect_price_arbitrage(symbol)
            
            if opportunity:
                self.stats['arbitrage_opportunities'] += 1
                
                logger.info(f"🎯 ARBITRAGE OPPORTUNITY DETECTED!")
                logger.info(f"   Symbol: {opportunity.symbol}")
                logger.info(f"   Direction: {opportunity.direction.value}")
                logger.info(f"   Binance Price: ${opportunity.binance_price:.2f}")
                logger.info(f"   Backpack Price: ${opportunity.backpack_price:.2f}")
                logger.info(f"   Price Difference: {opportunity.price_diff_pct:.3f}%")
                logger.info(f"   Profit Potential: {opportunity.profit_potential_pct:.3f}%")
                logger.info(f"   Confidence Score: {opportunity.confidence_score:.2f}")
                logger.info(f"   Trade Size Range: ${opportunity.min_trade_size:.0f} - ${opportunity.max_trade_size:.0f}")
                
                # Here you would normally:
                # 1. Validate the opportunity
                # 2. Check available capital
                # 3. Execute trades through OMS
                # 4. Monitor position through PMS
                # 5. Manage risk through Risk Manager
                
        except Exception as e:
            logger.error(f"❌ Arbitrage check error: {e}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        if not self.stats['start_time']:
            return
        
        runtime = datetime.now() - self.stats['start_time']
        runtime_seconds = runtime.total_seconds()
        
        logger.info("📊 FINAL STATISTICS:")
        logger.info(f"   Runtime: {runtime}")
        logger.info(f"   Tickers Processed: {self.stats['tickers_processed']}")
        logger.info(f"   Orderbooks Processed: {self.stats['orderbooks_processed']}")
        logger.info(f"   Arbitrage Opportunities: {self.stats['arbitrage_opportunities']}")
        
        if runtime_seconds > 0:
            ticker_rate = self.stats['tickers_processed'] / runtime_seconds
            orderbook_rate = self.stats['orderbooks_processed'] / runtime_seconds
            logger.info(f"   Ticker Rate: {ticker_rate:.2f} updates/second")
            logger.info(f"   Orderbook Rate: {orderbook_rate:.2f} updates/second")
        
        # Print performance stats from data feeder
        performance_stats = self.data_feeder.get_performance_stats()
        if performance_stats:
            logger.info("📈 PERFORMANCE METRICS:")
            for key, stats in performance_stats.items():
                logger.info(f"   {key}: {stats['message_count']} msgs, "
                          f"{stats['avg_latency_ms']:.2f}ms avg, "
                          f"{stats['error_count']} errors")

async def run_integration_example():
    """Run the integration example."""
    
    # Configuration
    config = {
        'market_data_feeder': {
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
            'buffer_size': 10000,
            'sync_tolerance_ms': 100
        },
        'arbitrage_detector': {
            'trading_costs': {
                'binance_spot_fee': 0.001,
                'backpack_spot_fee': 0.001,
                'slippage_estimate': 0.0005,
                'min_profit_threshold': 0.003  # 0.3% minimum profit
            },
            'detection_params': {
                'min_confidence_score': 0.7,
                'max_execution_time': 30.0,
                'liquidity_threshold': 1000
            },
            'max_arbitrage_size': 10000  # $10k max position
        },
        'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    }
    
    # Initialize integration
    integration = TradingSystemIntegration(config)
    
    try:
        logger.info("🚀 Starting Market Data Feeder Integration Example...")
        
        # Initialize all components
        await integration.initialize()
        
        # Run for demonstration (in production this would run continuously)
        logger.info("🔄 Running data feed integration for 60 seconds...")
        await asyncio.sleep(60)
        
        logger.info("⏹️ Stopping integration example...")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Integration example interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Integration example error: {e}")
        
    finally:
        await integration.shutdown()

if __name__ == "__main__":
    # Note: This example requires actual exchange connections
    # For testing without connections, use the test_market_data_feeder.py instead
    
    logger.info("🎯 Market Data Feeder Integration Example")
    logger.info("⚠️  This example requires actual exchange WebSocket connections")
    logger.info("💡 For testing without connections, run: python test_market_data_feeder.py")
    
    # Uncomment the line below to run with real connections
    # asyncio.run(run_integration_example())
    
    logger.info("✅ Example script loaded successfully")