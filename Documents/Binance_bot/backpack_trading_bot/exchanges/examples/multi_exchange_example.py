#!/usr/bin/env python3
"""
Multi-Exchange Trading System Example
Demonstrates how to use both Binance and Backpack adapters together
"""

import asyncio
import logging
from typing import Dict, List, Any
from integrated_trading_system.exchanges import (
    BinanceAdapter,
    BackpackAdapter,
    BinanceOrderSide,
    BinanceOrderType,
    BackpackOrderSide,
    BackpackOrderType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiExchangeManager:
    """
    Manages multiple exchange connections and provides unified interface
    """
    
    def __init__(self):
        self.exchanges = {}
        self.adapters = {}
    
    async def add_binance(self, api_key: str, api_secret: str, testnet: bool = True):
        """Add Binance exchange to the manager."""
        adapter = BinanceAdapter(api_key, api_secret, testnet)
        await adapter.initialize()
        self.adapters['binance'] = adapter
        self.exchanges['binance'] = 'Binance'
        logger.info("✅ Binance adapter added")
    
    async def add_backpack(self, api_key: str, api_secret: str, testnet: bool = True):
        """Add Backpack exchange to the manager."""
        adapter = BackpackAdapter(api_key, api_secret, testnet)
        await adapter.initialize()
        self.adapters['backpack'] = adapter
        self.exchanges['backpack'] = 'Backpack'
        logger.info("✅ Backpack adapter added")
    
    async def get_all_tickers(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data from all connected exchanges."""
        tickers = {}
        
        for exchange_name, adapter in self.adapters.items():
            try:
                ticker = await adapter.get_ticker(symbol)
                tickers[exchange_name] = {
                    'price': ticker.price,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'volume': ticker.volume,
                    'timestamp': ticker.timestamp
                }
            except Exception as e:
                logger.error(f"❌ Failed to get ticker from {exchange_name}: {e}")
                tickers[exchange_name] = None
        
        return tickers
    
    async def get_all_balances(self) -> Dict[str, List[Any]]:
        """Get balances from all connected exchanges."""
        all_balances = {}
        
        for exchange_name, adapter in self.adapters.items():
            try:
                balances = await adapter.get_balances()
                all_balances[exchange_name] = balances
            except Exception as e:
                logger.error(f"❌ Failed to get balances from {exchange_name}: {e}")
                all_balances[exchange_name] = []
        
        return all_balances
    
    async def find_arbitrage_opportunities(self, symbol: str, min_profit_pct: float = 0.1) -> List[Dict]:
        """Find arbitrage opportunities between exchanges."""
        tickers = await self.get_all_tickers(symbol)
        opportunities = []
        
        exchange_names = list(tickers.keys())
        
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                exchange1 = exchange_names[i]
                exchange2 = exchange_names[j]
                
                ticker1 = tickers[exchange1]
                ticker2 = tickers[exchange2]
                
                if ticker1 and ticker2:
                    # Check if we can buy on exchange1 and sell on exchange2
                    buy_price = ticker1['ask']
                    sell_price = ticker2['bid']
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    if profit_pct > min_profit_pct:
                        opportunities.append({
                            'buy_exchange': exchange1,
                            'sell_exchange': exchange2,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct,
                            'symbol': symbol
                        })
                    
                    # Check reverse direction
                    buy_price = ticker2['ask']
                    sell_price = ticker1['bid']
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    if profit_pct > min_profit_pct:
                        opportunities.append({
                            'buy_exchange': exchange2,
                            'sell_exchange': exchange1,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'profit_pct': profit_pct,
                            'symbol': symbol
                        })
        
        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)
    
    async def execute_arbitrage(self, opportunity: Dict, quantity: float) -> Dict:
        """Execute an arbitrage opportunity (simulation)."""
        buy_exchange = opportunity['buy_exchange']
        sell_exchange = opportunity['sell_exchange']
        symbol = opportunity['symbol']
        
        logger.info(f"🎯 Executing arbitrage: Buy {quantity} {symbol} on {buy_exchange}, Sell on {sell_exchange}")
        
        # This is a simulation - in real trading, you would:
        # 1. Check balances
        # 2. Place buy order on buy_exchange
        # 3. Wait for fill
        # 4. Place sell order on sell_exchange
        # 5. Monitor execution
        
        return {
            'status': 'simulated',
            'buy_exchange': buy_exchange,
            'sell_exchange': sell_exchange,
            'quantity': quantity,
            'expected_profit_pct': opportunity['profit_pct']
        }
    
    async def close_all(self):
        """Close all exchange connections."""
        for adapter in self.adapters.values():
            await adapter.close()
        logger.info("🔒 All adapters closed")

async def example_usage():
    """Example of how to use the multi-exchange system."""
    manager = MultiExchangeManager()
    
    try:
        # Add exchanges (use your actual API keys)
        await manager.add_binance(
            api_key="your_binance_api_key",
            api_secret="your_binance_api_secret",
            testnet=True
        )
        
        await manager.add_backpack(
            api_key="your_backpack_api_key",
            api_secret="your_backpack_api_secret",
            testnet=True
        )
        
        # Get tickers from all exchanges
        print("\n📊 Getting tickers from all exchanges...")
        tickers = await manager.get_all_tickers("BTCUSDT")
        
        for exchange, ticker in tickers.items():
            if ticker:
                print(f"{exchange}: ${ticker['price']:.2f} (Bid: ${ticker['bid']:.2f}, Ask: ${ticker['ask']:.2f})")
            else:
                print(f"{exchange}: Failed to get ticker")
        
        # Find arbitrage opportunities
        print("\n🔍 Looking for arbitrage opportunities...")
        opportunities = await manager.find_arbitrage_opportunities("BTCUSDT", min_profit_pct=0.01)
        
        if opportunities:
            print(f"Found {len(opportunities)} arbitrage opportunities:")
            for i, opp in enumerate(opportunities[:3]):  # Show top 3
                print(f"{i+1}. Buy on {opp['buy_exchange']} (${opp['buy_price']:.2f}), "
                      f"Sell on {opp['sell_exchange']} (${opp['sell_price']:.2f}), "
                      f"Profit: {opp['profit_pct']:.2f}%")
        else:
            print("No arbitrage opportunities found")
        
        # Get balances (commented out for demo)
        # print("\n💰 Getting balances from all exchanges...")
        # balances = await manager.get_all_balances()
        # for exchange, balance_list in balances.items():
        #     print(f"{exchange}: {len(balance_list)} assets")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        print("This is expected if you don't have valid API keys configured")
    
    finally:
        await manager.close_all()

async def demo_public_api():
    """Demo using only public API (no authentication required)."""
    print("🚀 Multi-Exchange System Demo (Public API Only)\n")
    
    manager = MultiExchangeManager()
    
    try:
        # Add exchanges with dummy credentials (for public API only)
        await manager.add_binance("dummy", "dummy", testnet=True)
        
        # Get ticker data
        print("📊 Getting BTC ticker from Binance...")
        tickers = await manager.get_all_tickers("BTCUSDT")
        
        for exchange, ticker in tickers.items():
            if ticker:
                print(f"✅ {exchange}: ${ticker['price']:.2f}")
            else:
                print(f"❌ {exchange}: Failed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        await manager.close_all()

if __name__ == "__main__":
    # Run the public API demo
    asyncio.run(demo_public_api())
    
    # Uncomment to run full example with authentication
    # asyncio.run(example_usage())