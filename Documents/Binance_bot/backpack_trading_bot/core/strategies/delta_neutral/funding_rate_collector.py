"""
Advanced Funding Rate Collector for Delta Neutral Strategy
Monitors and captures funding fees from futures positions
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FundingRateData:
    """Funding rate information for a symbol"""
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    mark_price: float
    index_price: float
    timestamp: datetime
    annualized_rate: float

@dataclass
class FundingPosition:
    """Active position for funding rate capture"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    funding_collected: float
    days_held: int
    expected_daily_funding: float

class CompliantFundingFeeCollector:
    """
    Advanced funding fee collector with rate limiting compliance
    Optimized for delta neutral strategies to capture funding rates
    """
    
    def __init__(self, binance_client, rate_limiter, config):
        """
        Initialize funding fee collector.
        
        Args:
            binance_client: Binance futures client
            rate_limiter: API rate limiter
            config: Configuration settings
        """
        self.client = binance_client
        self.rate_limiter = rate_limiter
        self.config = config
        self.shutdown_event = asyncio.Event()
        self.monitoring_task = None
        
        # Funding data storage
        self.funding_history: Dict[str, List[FundingRateData]] = {}
        self.active_positions: Dict[str, FundingPosition] = {}
        self.funding_statistics: Dict[str, Dict] = {}
        
        # Configuration
        self.symbols = config.get('funding_symbols', ['BTCUSDT', 'ETHUSDT'])
        self.min_funding_rate = config.get('min_funding_rate', 0.0001)  # 0.01%
        self.max_position_size = config.get('max_funding_position', 1000)  # USDT
        self.funding_check_interval = config.get('funding_check_interval', 3600)  # 1 hour
        
        logger.info("Funding fee collector initialized")
    
    async def start_monitoring(self):
        """Start monitoring funding rates for all symbols"""
        if self.monitoring_task:
            logger.warning("Funding monitoring is already running")
            return
        
        logger.info("Starting funding rate monitoring")
        self.monitoring_task = asyncio.create_task(self._monitor_funding_rates())
    
    async def _monitor_funding_rates(self):
        """Main monitoring loop for funding rates"""
        while not self.shutdown_event.is_set():
            try:
                # Check funding rates for all symbols
                for symbol in self.symbols:
                    await self._check_symbol_funding(symbol)
                
                # Analyze funding opportunities
                await self._analyze_funding_opportunities()
                
                # Update position tracking
                await self._update_position_tracking()
                
                # Sleep until next check
                await asyncio.sleep(self.funding_check_interval)
                
            except Exception as e:
                logger.error(f"Error in funding monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_symbol_funding(self, symbol: str):
        """Check funding rate for a specific symbol"""
        try:
            # Rate limiting compliance
            if not await self.rate_limiter.wait_for_request_weight(5):
                logger.warning(f"Rate limit reached, skipping {symbol}")
                return
            
            # Get premium index (includes funding rate)
            premium_index = await self.client.get_premium_index(symbol=symbol)
            
            # Parse funding data
            funding_data = FundingRateData(
                symbol=symbol,
                funding_rate=float(premium_index['lastFundingRate']),
                next_funding_time=datetime.fromtimestamp(
                    int(premium_index['nextFundingTime']) / 1000
                ),
                mark_price=float(premium_index['markPrice']),
                index_price=float(premium_index['indexPrice']),
                timestamp=datetime.now(),
                annualized_rate=float(premium_index['lastFundingRate']) * 365 * 3  # 3 times daily
            )
            
            # Store funding data
            if symbol not in self.funding_history:
                self.funding_history[symbol] = []
            
            self.funding_history[symbol].append(funding_data)
            
            # Keep only recent history (last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.funding_history[symbol] = [
                data for data in self.funding_history[symbol]
                if data.timestamp > cutoff_time
            ]
            
            logger.info(
                f"Funding rate {symbol}: {funding_data.funding_rate:.4f} "
                f"(annualized: {funding_data.annualized_rate:.2%})"
            )
            
        except Exception as e:
            logger.error(f"Error checking funding for {symbol}: {e}")
    
    async def _analyze_funding_opportunities(self):
        """Analyze current funding rates for trading opportunities"""
        try:
            for symbol in self.symbols:
                if symbol not in self.funding_history or not self.funding_history[symbol]:
                    continue
                
                latest_funding = self.funding_history[symbol][-1]
                
                # Calculate funding statistics
                await self._calculate_funding_statistics(symbol)
                
                # Check if funding rate is attractive
                if abs(latest_funding.funding_rate) >= self.min_funding_rate:
                    await self._evaluate_funding_opportunity(symbol, latest_funding)
                
        except Exception as e:
            logger.error(f"Error analyzing funding opportunities: {e}")
    
    async def _calculate_funding_statistics(self, symbol: str):
        """Calculate funding rate statistics for a symbol"""
        try:
            funding_rates = [data.funding_rate for data in self.funding_history[symbol]]
            
            if len(funding_rates) < 10:  # Need minimum data
                return
            
            stats = {
                'mean_funding': np.mean(funding_rates),
                'std_funding': np.std(funding_rates),
                'median_funding': np.median(funding_rates),
                'percentile_75': np.percentile(funding_rates, 75),
                'percentile_25': np.percentile(funding_rates, 25),
                'positive_rate_frequency': sum(1 for rate in funding_rates if rate > 0) / len(funding_rates),
                'negative_rate_frequency': sum(1 for rate in funding_rates if rate < 0) / len(funding_rates),
                'max_rate': max(funding_rates),
                'min_rate': min(funding_rates),
                'current_percentile': self._calculate_current_percentile(symbol, funding_rates[-1])
            }
            
            self.funding_statistics[symbol] = stats
            
        except Exception as e:
            logger.error(f"Error calculating funding statistics for {symbol}: {e}")
    
    def _calculate_current_percentile(self, symbol: str, current_rate: float) -> float:
        """Calculate what percentile the current funding rate represents"""
        try:
            all_rates = [data.funding_rate for data in self.funding_history[symbol]]
            return (sum(1 for rate in all_rates if rate <= current_rate) / len(all_rates)) * 100
        except:
            return 50.0
    
    async def _evaluate_funding_opportunity(self, symbol: str, funding_data: FundingRateData):
        """Evaluate if current funding rate presents a good opportunity"""
        try:
            if symbol not in self.funding_statistics:
                return
            
            stats = self.funding_statistics[symbol]
            current_rate = funding_data.funding_rate
            
            # Opportunity scoring
            opportunity_score = 0
            
            # High absolute funding rate
            if abs(current_rate) > abs(stats['percentile_75']):
                opportunity_score += 30
            
            # Extreme funding rate (top 10%)
            if funding_data.annualized_rate > 0.50:  # 50% annualized
                opportunity_score += 40
            elif funding_data.annualized_rate > 0.30:  # 30% annualized
                opportunity_score += 25
            elif funding_data.annualized_rate > 0.15:  # 15% annualized
                opportunity_score += 15
            
            # Persistence check (same direction for multiple periods)
            recent_rates = [data.funding_rate for data in self.funding_history[symbol][-5:]]
            if len(recent_rates) >= 3:
                if all(rate > 0 for rate in recent_rates) or all(rate < 0 for rate in recent_rates):
                    opportunity_score += 20
            
            # Log opportunity if score is high enough
            if opportunity_score >= 50:
                logger.info(
                    f"HIGH FUNDING OPPORTUNITY {symbol}: "
                    f"Rate: {current_rate:.4f} ({funding_data.annualized_rate:.2%} annual) "
                    f"Score: {opportunity_score}/100"
                )
                
                # Suggest position direction
                suggested_side = 'short' if current_rate > 0 else 'long'
                logger.info(f"Suggested position: {suggested_side} {symbol} to collect funding")
            
        except Exception as e:
            logger.error(f"Error evaluating funding opportunity for {symbol}: {e}")
    
    async def _update_position_tracking(self):
        """Update tracking for active funding positions"""
        try:
            for position_id, position in self.active_positions.items():
                # Calculate days held
                # This would integrate with actual position management
                # to track real positions and funding collected
                pass
                
        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")
    
    async def add_funding_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Add a position for funding tracking"""
        try:
            position_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"
            
            # Get current funding data
            latest_funding = None
            if symbol in self.funding_history and self.funding_history[symbol]:
                latest_funding = self.funding_history[symbol][-1]
            
            expected_daily_funding = 0
            if latest_funding:
                expected_daily_funding = latest_funding.funding_rate * 3 * size  # 3 times daily
            
            position = FundingPosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                funding_collected=0.0,
                days_held=0,
                expected_daily_funding=expected_daily_funding
            )
            
            self.active_positions[position_id] = position
            
            logger.info(
                f"Added funding position: {side} {size} {symbol} @ {entry_price}"
                f" (expected daily funding: {expected_daily_funding:.4f})"
            )
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error adding funding position: {e}")
            return None
    
    async def remove_funding_position(self, position_id: str):
        """Remove a funding position from tracking"""
        try:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                del self.active_positions[position_id]
                
                logger.info(
                    f"Removed funding position: {position.symbol} "
                    f"(total funding collected: {position.funding_collected:.4f})"
                )
                
        except Exception as e:
            logger.error(f"Error removing funding position: {e}")
    
    def get_funding_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get funding rate summary for analysis"""
        try:
            if symbol:
                symbols = [symbol]
            else:
                symbols = self.symbols
            
            summary = {}
            
            for sym in symbols:
                if sym not in self.funding_history or not self.funding_history[sym]:
                    continue
                
                latest = self.funding_history[sym][-1]
                stats = self.funding_statistics.get(sym, {})
                
                summary[sym] = {
                    'current_funding_rate': latest.funding_rate,
                    'annualized_rate': latest.annualized_rate,
                    'next_funding_time': latest.next_funding_time.isoformat(),
                    'mark_price': latest.mark_price,
                    'statistics': stats,
                    'data_points': len(self.funding_history[sym])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating funding summary: {e}")
            return {}
    
    def get_best_funding_opportunities(self, min_annual_rate: float = 0.10) -> List[Dict[str, Any]]:
        """Get best current funding opportunities"""
        try:
            opportunities = []
            
            for symbol in self.symbols:
                if symbol not in self.funding_history or not self.funding_history[symbol]:
                    continue
                
                latest = self.funding_history[symbol][-1]
                
                if abs(latest.annualized_rate) >= min_annual_rate:
                    opportunities.append({
                        'symbol': symbol,
                        'funding_rate': latest.funding_rate,
                        'annualized_rate': latest.annualized_rate,
                        'suggested_side': 'short' if latest.funding_rate > 0 else 'long',
                        'next_funding': latest.next_funding_time.isoformat(),
                        'opportunity_rank': abs(latest.annualized_rate)
                    })
            
            # Sort by annualized rate (best opportunities first)
            opportunities.sort(key=lambda x: x['opportunity_rank'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error getting funding opportunities: {e}")
            return []
    
    async def shutdown(self, reason: str = ""):
        """Shutdown funding fee collector"""
        logger.info(f"Shutting down funding fee collector. Reason: {reason}")
        
        self.shutdown_event.set()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Funding fee collector shutdown complete")

if __name__ == '__main__':
    print("CompliantFundingFeeCollector - Advanced funding rate capture system")
    print("Key features:")
    print("- Real-time funding rate monitoring")
    print("- Statistical analysis and opportunity scoring")
    print("- Position tracking for funding collection")
    print("- Rate limiting compliance")
    print("- Delta neutral strategy optimization")