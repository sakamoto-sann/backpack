#!/usr/bin/env python3
"""
🎯 CROSS-EXCHANGE ARBITRAGE DETECTOR v1.0.0
Advanced arbitrage opportunity detection for Binance + Backpack integration

Features:
- 💱 Price arbitrage detection
- 💰 Funding rate arbitrage analysis  
- 📊 Basis trading opportunities
- ⚖️ Delta-neutral integration
- 🛡️ Risk-adjusted profit calculation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import aiohttp
import json

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    PRICE_ARBITRAGE = "price_arbitrage"
    FUNDING_RATE = "funding_rate"
    BASIS_TRADING = "basis_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"

class MarketDirection(Enum):
    BUY_BINANCE_SELL_BACKPACK = "buy_binance_sell_backpack"
    BUY_BACKPACK_SELL_BINANCE = "buy_backpack_sell_binance"
    NEUTRAL = "neutral"

@dataclass
class ArbitrageOpportunity:
    type: ArbitrageType
    symbol: str
    direction: MarketDirection
    binance_price: float
    backpack_price: float
    price_diff: float
    price_diff_pct: float
    profit_potential: float
    profit_potential_pct: float
    confidence_score: float
    execution_priority: int
    min_trade_size: float
    max_trade_size: float
    estimated_execution_time: float
    timestamp: datetime
    
    # Risk metrics
    slippage_estimate: float = 0.0
    liquidity_score: float = 0.0
    volatility_risk: float = 0.0
    
    # Trading costs
    binance_fee: float = 0.001
    backpack_fee: float = 0.001
    total_fees: float = 0.002

@dataclass
class FundingRateOpportunity:
    symbol: str
    binance_funding_rate: float
    backpack_funding_rate: float
    rate_diff: float
    rate_diff_annualized: float
    profit_potential_8h: float
    confidence_score: float
    position_direction: str  # "long_binance_short_backpack" or vice versa
    next_funding_time: datetime
    required_margin: float
    estimated_fees: float
    risk_score: float
    timestamp: datetime

@dataclass
class BasisTradingOpportunity:
    symbol: str
    exchange: str
    spot_price: float
    futures_price: float
    basis: float
    basis_pct: float
    time_to_expiry: timedelta
    annualized_return: float
    confidence_score: float
    position_type: str  # "contango" or "backwardation"
    required_margin: float
    estimated_fees: float
    liquidity_score: float
    risk_score: float
    expiry_date: datetime
    contract_size: float
    timestamp: datetime

class ArbitrageDetector:
    """
    Advanced arbitrage detector for multi-exchange opportunities.
    Integrates with existing delta-neutral institutional trading system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Market data storage
        self.binance_data = {}
        self.backpack_data = {}
        
        # Funding rate data storage
        self.funding_rates = {
            'binance': {},
            'backpack': {}
        }
        
        # Futures data storage
        self.futures_data = {
            'binance': {},
            'backpack': {}
        }
        
        # Price history for statistical analysis
        self.price_history = {}
        
        # Trading costs and thresholds
        self.trading_costs = config.get('trading_costs', {
            'binance_spot_fee': 0.001,
            'backpack_spot_fee': 0.001,
            'binance_futures_fee': 0.0004,
            'backpack_futures_fee': 0.0004,
            'slippage_estimate': 0.0005,
            'min_profit_threshold': 0.003  # 0.3% minimum profit
        })
        
        # Detection parameters
        self.detection_params = config.get('detection_params', {
            'price_update_interval': 1.0,  # seconds
            'min_confidence_score': 0.7,
            'max_execution_time': 30.0,  # seconds
            'liquidity_threshold': 1000,  # USD
            'volatility_window': 100,  # price samples
            'min_funding_rate_diff': 0.0001,  # 0.01% minimum funding rate difference
            'min_basis_threshold': 0.005,  # 0.5% minimum basis threshold
        })
        
        # HTTP session for API calls
        self.session = None
        
        logger.info("🎯 Arbitrage detector initialized")
    
    async def update_market_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """Update market data from exchanges."""
        try:
            if exchange == "binance":
                self.binance_data[symbol] = data
            elif exchange == "backpack":
                self.backpack_data[symbol] = data
            
            # Update price history for statistical analysis
            if symbol not in self.price_history:
                self.price_history[symbol] = {
                    'binance': [],
                    'backpack': [],
                    'timestamps': []
                }
            
            # Store price history (last 1000 points)
            if exchange in ["binance", "backpack"] and 'price' in data:
                history = self.price_history[symbol]
                if len(history['timestamps']) >= 1000:
                    # Remove oldest data
                    for key in history:
                        history[key] = history[key][-999:]
                
                history['timestamps'].append(datetime.now())
                history['binance'].append(
                    data['price'] if exchange == "binance" 
                    else history['binance'][-1] if history['binance'] else 0
                )
                history['backpack'].append(
                    data['price'] if exchange == "backpack" 
                    else history['backpack'][-1] if history['backpack'] else 0
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to update market data: {e}")
    
    async def detect_price_arbitrage(self, symbol: str) -> Optional[ArbitrageOpportunity]:
        """Detect price arbitrage opportunities between exchanges."""
        try:
            if (symbol not in self.binance_data or 
                symbol not in self.backpack_data):
                return None
            
            binance_ticker = self.binance_data[symbol]
            backpack_ticker = self.backpack_data[symbol]
            
            binance_price = float(binance_ticker.get('price', 0))
            backpack_price = float(backpack_ticker.get('price', 0))
            
            if binance_price == 0 or backpack_price == 0:
                return None
            
            # Calculate price difference
            price_diff = abs(binance_price - backpack_price)
            price_diff_pct = price_diff / min(binance_price, backpack_price) * 100
            
            # Determine direction
            if binance_price < backpack_price:
                direction = MarketDirection.BUY_BINANCE_SELL_BACKPACK
                profit_before_fees = backpack_price - binance_price
            else:
                direction = MarketDirection.BUY_BACKPACK_SELL_BINANCE
                profit_before_fees = binance_price - backpack_price
            
            # Calculate trading costs
            total_fees = (self.trading_costs.get('binance_spot_fee', 0.001) + 
                         self.trading_costs.get('backpack_spot_fee', 0.001) +
                         self.trading_costs.get('slippage_estimate', 0.0005)) * min(binance_price, backpack_price)
            
            # Net profit potential
            profit_potential = profit_before_fees - total_fees
            profit_potential_pct = profit_potential / min(binance_price, backpack_price) * 100
            
            # Check minimum profit threshold
            if profit_potential_pct < self.trading_costs.get('min_profit_threshold', 0.003) * 100:
                return None
            
            # Calculate confidence score
            confidence_score = self._calculate_arbitrage_confidence(
                symbol, price_diff_pct, binance_ticker, backpack_ticker
            )
            
            # Calculate trade sizing
            min_trade_size, max_trade_size = self._calculate_trade_sizing(
                symbol, binance_ticker, backpack_ticker
            )
            
            return ArbitrageOpportunity(
                type=ArbitrageType.PRICE_ARBITRAGE,
                symbol=symbol,
                direction=direction,
                binance_price=binance_price,
                backpack_price=backpack_price,
                price_diff=price_diff,
                price_diff_pct=price_diff_pct,
                profit_potential=profit_potential,
                profit_potential_pct=profit_potential_pct,
                confidence_score=confidence_score,
                execution_priority=self._calculate_execution_priority(profit_potential_pct, confidence_score),
                min_trade_size=min_trade_size,
                max_trade_size=max_trade_size,
                estimated_execution_time=self._estimate_execution_time(symbol),
                timestamp=datetime.now(),
                slippage_estimate=self.trading_costs['slippage_estimate'],
                liquidity_score=self._calculate_liquidity_score(binance_ticker, backpack_ticker),
                volatility_risk=self._calculate_volatility_risk(symbol),
                total_fees=total_fees
            )
            
        except Exception as e:
            logger.error(f"❌ Price arbitrage detection error for {symbol}: {e}")
            return None
    
    async def detect_funding_rate_arbitrage(self, symbol: str) -> Optional[FundingRateOpportunity]:
        """Detect funding rate arbitrage opportunities."""
        try:
            # Fetch current funding rates from both exchanges
            binance_funding_data = await self._fetch_binance_funding_rate(symbol)
            backpack_funding_data = await self._fetch_backpack_funding_rate(symbol)
            
            if not binance_funding_data or not backpack_funding_data:
                logger.warning(f"❌ Insufficient funding rate data for {symbol}")
                return None
            
            binance_funding = binance_funding_data['funding_rate']
            backpack_funding = backpack_funding_data['funding_rate']
            
            # Calculate rate difference
            rate_diff = abs(binance_funding - backpack_funding)
            rate_diff_annualized = rate_diff * 3 * 365  # 3 times per day, 365 days
            
            # Check if difference meets minimum threshold
            if rate_diff < self.detection_params['min_funding_rate_diff']:
                return None
            
            # Determine position direction
            if binance_funding > backpack_funding:
                position_direction = "long_backpack_short_binance"
                profit_rate = rate_diff
            else:
                position_direction = "long_binance_short_backpack"
                profit_rate = rate_diff
            
            # Calculate profit potential for standard position size
            standard_position_size = 10000  # $10k
            profit_potential_8h = profit_rate * standard_position_size
            
            # Calculate required margin (assuming 10x leverage)
            required_margin = standard_position_size * 0.1
            
            # Estimate trading fees
            estimated_fees = self._calculate_funding_arbitrage_fees(standard_position_size)
            
            # Calculate risk score
            risk_score = self._calculate_funding_risk_score(binance_funding_data, backpack_funding_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_funding_confidence_score(
                rate_diff, binance_funding_data, backpack_funding_data
            )
            
            # Get next funding time
            next_funding_time = self._get_next_funding_time()
            
            # Only proceed if profitable after fees
            if profit_potential_8h > estimated_fees:
                return FundingRateOpportunity(
                    symbol=symbol,
                    binance_funding_rate=binance_funding,
                    backpack_funding_rate=backpack_funding,
                    rate_diff=rate_diff,
                    rate_diff_annualized=rate_diff_annualized,
                    profit_potential_8h=profit_potential_8h - estimated_fees,
                    confidence_score=confidence_score,
                    position_direction=position_direction,
                    next_funding_time=next_funding_time,
                    required_margin=required_margin,
                    estimated_fees=estimated_fees,
                    risk_score=risk_score,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Funding rate arbitrage detection error: {e}")
            return None
    
    async def detect_basis_trading_opportunities(self, symbol: str) -> List[BasisTradingOpportunity]:
        """Detect basis trading opportunities on each exchange."""
        opportunities = []
        
        try:
            # Fetch futures contracts for both exchanges
            binance_contracts = await self._fetch_binance_futures_contracts(symbol)
            backpack_contracts = await self._fetch_backpack_futures_contracts(symbol)
            
            # Process Binance opportunities
            for contract in binance_contracts:
                opportunity = await self._analyze_basis_opportunity("binance", symbol, contract)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Process Backpack opportunities
            for contract in backpack_contracts:
                opportunity = await self._analyze_basis_opportunity("backpack", symbol, contract)
                if opportunity:
                    opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Basis trading detection error: {e}")
        
        return opportunities
    
    async def _fetch_binance_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current funding rate from Binance."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Binance futures funding rate endpoint
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'funding_rate': float(data.get('lastFundingRate', 0)),
                        'next_funding_time': datetime.fromtimestamp(int(data.get('nextFundingTime', 0)) / 1000),
                        'mark_price': float(data.get('markPrice', 0)),
                        'index_price': float(data.get('indexPrice', 0))
                    }
                else:
                    logger.error(f"❌ Binance funding rate API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to fetch Binance funding rate: {e}")
            return None
    
    async def _fetch_backpack_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current funding rate from Backpack."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Backpack funding rate endpoint (adjust based on actual API)
            url = f"https://api.backpack.exchange/api/v1/funding"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'funding_rate': float(data.get('fundingRate', 0)),
                        'next_funding_time': datetime.fromtimestamp(int(data.get('nextFundingTime', 0)) / 1000),
                        'mark_price': float(data.get('markPrice', 0)),
                        'index_price': float(data.get('indexPrice', 0))
                    }
                else:
                    logger.error(f"❌ Backpack funding rate API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to fetch Backpack funding rate: {e}")
            return None
    
    async def _fetch_binance_futures_contracts(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch available futures contracts from Binance."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Fetch exchange info for futures
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    contracts = []
                    
                    for contract_info in data.get('symbols', []):
                        if contract_info.get('baseAsset') == symbol.replace('USDT', ''):
                            # Fetch current price
                            price_data = await self._fetch_binance_futures_price(contract_info['symbol'])
                            if price_data:
                                contracts.append({
                                    'symbol': contract_info['symbol'],
                                    'contract_type': contract_info.get('contractType', 'PERPETUAL'),
                                    'price': price_data['price'],
                                    'volume': price_data['volume'],
                                    'delivery_date': contract_info.get('deliveryDate'),
                                    'contract_size': float(contract_info.get('contractSize', 1))
                                })
                    
                    return contracts
                else:
                    logger.error(f"❌ Binance futures contracts API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Failed to fetch Binance futures contracts: {e}")
            return []
    
    async def _fetch_backpack_futures_contracts(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch available futures contracts from Backpack."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Backpack futures contracts endpoint (adjust based on actual API)
            url = "https://api.backpack.exchange/api/v1/futures/contracts"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    contracts = []
                    
                    for contract in data:
                        if symbol.lower() in contract.get('symbol', '').lower():
                            contracts.append({
                                'symbol': contract['symbol'],
                                'contract_type': contract.get('type', 'PERPETUAL'),
                                'price': float(contract.get('price', 0)),
                                'volume': float(contract.get('volume', 0)),
                                'delivery_date': contract.get('deliveryDate'),
                                'contract_size': float(contract.get('contractSize', 1))
                            })
                    
                    return contracts
                else:
                    logger.error(f"❌ Backpack futures contracts API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Failed to fetch Backpack futures contracts: {e}")
            return []
    
    async def _fetch_binance_futures_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current futures price from Binance."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"https://fapi.binance.com/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': float(data.get('lastPrice', 0)),
                        'volume': float(data.get('volume', 0))
                    }
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to fetch Binance futures price: {e}")
            return None
    
    async def _analyze_basis_opportunity(self, exchange: str, symbol: str, contract: Dict[str, Any]) -> Optional[BasisTradingOpportunity]:
        """Analyze a specific basis trading opportunity."""
        try:
            # Get spot price
            spot_price = await self._get_spot_price(exchange, symbol)
            if not spot_price:
                return None
            
            futures_price = contract['price']
            
            # Calculate basis
            basis = futures_price - spot_price
            basis_pct = basis / spot_price * 100
            
            # Check if basis meets minimum threshold
            if abs(basis_pct) < self.detection_params['min_basis_threshold']:
                return None
            
            # Calculate time to expiry
            if contract.get('delivery_date'):
                expiry_date = datetime.fromtimestamp(int(contract['delivery_date']) / 1000)
                time_to_expiry = expiry_date - datetime.now()
            else:
                # Perpetual contract
                expiry_date = datetime.now() + timedelta(days=365)
                time_to_expiry = timedelta(days=365)
            
            # Calculate annualized return
            if time_to_expiry.days > 0:
                annualized_return = (basis_pct / time_to_expiry.days) * 365
            else:
                annualized_return = 0
            
            # Calculate required margin (assuming 10x leverage)
            position_size = 10000  # $10k
            required_margin = position_size * 0.1
            
            # Estimate fees
            estimated_fees = self._calculate_basis_trading_fees(position_size)
            
            # Calculate liquidity score
            liquidity_score = min(contract['volume'] / 1000000, 1.0)  # Normalize to $1M
            
            # Calculate risk score
            risk_score = self._calculate_basis_risk_score(basis_pct, time_to_expiry, liquidity_score)
            
            # Calculate confidence score
            confidence_score = self._calculate_basis_confidence_score(
                basis_pct, time_to_expiry, liquidity_score, risk_score
            )
            
            return BasisTradingOpportunity(
                symbol=symbol,
                exchange=exchange,
                spot_price=spot_price,
                futures_price=futures_price,
                basis=basis,
                basis_pct=basis_pct,
                time_to_expiry=time_to_expiry,
                annualized_return=annualized_return,
                confidence_score=confidence_score,
                position_type="contango" if basis > 0 else "backwardation",
                required_margin=required_margin,
                estimated_fees=estimated_fees,
                liquidity_score=liquidity_score,
                risk_score=risk_score,
                expiry_date=expiry_date,
                contract_size=contract.get('contract_size', 1.0),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Basis opportunity analysis error: {e}")
            return None
    
    async def _get_spot_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get current spot price from exchange."""
        try:
            if exchange == "binance":
                if symbol in self.binance_data:
                    return float(self.binance_data[symbol].get('price', 0))
            elif exchange == "backpack":
                if symbol in self.backpack_data:
                    return float(self.backpack_data[symbol].get('price', 0))
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get spot price: {e}")
            return None
    
    def _calculate_funding_arbitrage_fees(self, position_size: float) -> float:
        """Calculate estimated fees for funding rate arbitrage."""
        # Entry and exit fees for both exchanges
        binance_fees = position_size * self.trading_costs.get('binance_futures_fee', 0.0004) * 2  # Open and close
        backpack_fees = position_size * self.trading_costs.get('backpack_futures_fee', 0.0004) * 2  # Open and close
        
        # Slippage
        slippage = position_size * self.trading_costs.get('slippage_estimate', 0.0005)
        
        return binance_fees + backpack_fees + slippage
    
    def _calculate_basis_trading_fees(self, position_size: float) -> float:
        """Calculate estimated fees for basis trading."""
        # Spot trading fees
        spot_fees = position_size * self.trading_costs.get('binance_spot_fee', 0.001)
        
        # Futures trading fees
        futures_fees = position_size * self.trading_costs.get('binance_futures_fee', 0.0004) * 2  # Open and close
        
        # Slippage
        slippage = position_size * self.trading_costs.get('slippage_estimate', 0.0005)
        
        return spot_fees + futures_fees + slippage
    
    def _calculate_funding_risk_score(self, binance_data: Dict, backpack_data: Dict) -> float:
        """Calculate risk score for funding rate arbitrage."""
        try:
            # Risk factors:
            # 1. Mark price divergence
            # 2. Funding rate volatility
            # 3. Exchange reliability
            
            mark_price_diff = abs(binance_data['mark_price'] - backpack_data['mark_price'])
            mark_price_divergence = mark_price_diff / min(binance_data['mark_price'], backpack_data['mark_price'])
            
            # Higher divergence = higher risk
            divergence_risk = min(mark_price_divergence * 10, 1.0)
            
            # Base exchange risk (lower for established exchanges)
            exchange_risk = 0.1
            
            # Combine risks
            total_risk = (divergence_risk + exchange_risk) / 2
            
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Funding risk calculation error: {e}")
            return 0.5
    
    def _calculate_basis_risk_score(self, basis_pct: float, time_to_expiry: timedelta, liquidity_score: float) -> float:
        """Calculate risk score for basis trading."""
        try:
            # Risk factors:
            # 1. Time to expiry (longer = higher risk)
            # 2. Basis magnitude (extreme basis = higher risk)
            # 3. Liquidity (lower liquidity = higher risk)
            
            # Time risk (0-1, higher for longer time)
            time_risk = min(time_to_expiry.days / 365, 1.0)
            
            # Basis risk (0-1, higher for extreme basis)
            basis_risk = min(abs(basis_pct) / 10, 1.0)
            
            # Liquidity risk (0-1, higher for lower liquidity)
            liquidity_risk = 1.0 - liquidity_score
            
            # Combine risks
            total_risk = (time_risk + basis_risk + liquidity_risk) / 3
            
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Basis risk calculation error: {e}")
            return 0.5
    
    def _calculate_funding_confidence_score(self, rate_diff: float, binance_data: Dict, backpack_data: Dict) -> float:
        """Calculate confidence score for funding rate arbitrage."""
        try:
            # Confidence factors:
            # 1. Rate difference magnitude
            # 2. Mark price alignment
            # 3. Data freshness
            
            # Rate difference factor (higher difference = higher confidence)
            rate_factor = min(rate_diff / 0.001, 1.0)  # Normalize to 0.1%
            
            # Mark price alignment (closer = higher confidence)
            mark_price_diff = abs(binance_data['mark_price'] - backpack_data['mark_price'])
            alignment_factor = max(0, 1 - mark_price_diff / min(binance_data['mark_price'], backpack_data['mark_price']) * 10)
            
            # Data freshness factor (assume recent data)
            freshness_factor = 1.0
            
            # Combine factors
            confidence = (rate_factor * 0.5 + alignment_factor * 0.3 + freshness_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Funding confidence calculation error: {e}")
            return 0.5
    
    def _calculate_basis_confidence_score(self, basis_pct: float, time_to_expiry: timedelta, 
                                        liquidity_score: float, risk_score: float) -> float:
        """Calculate confidence score for basis trading."""
        try:
            # Confidence factors:
            # 1. Basis magnitude (higher = higher confidence up to a point)
            # 2. Time to expiry (shorter = higher confidence)
            # 3. Liquidity (higher = higher confidence)
            # 4. Risk (lower = higher confidence)
            
            # Basis factor (optimal around 1-3%)
            basis_factor = min(abs(basis_pct) / 2, 1.0) if abs(basis_pct) < 5 else max(0, 1 - abs(basis_pct) / 10)
            
            # Time factor (shorter time = higher confidence)
            time_factor = max(0, 1 - time_to_expiry.days / 365)
            
            # Risk factor (lower risk = higher confidence)
            risk_factor = 1.0 - risk_score
            
            # Combine factors
            confidence = (basis_factor * 0.4 + time_factor * 0.2 + liquidity_score * 0.2 + risk_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Basis confidence calculation error: {e}")
            return 0.5
    
    def _get_next_funding_time(self) -> datetime:
        """Get the next funding time (every 8 hours)."""
        now = datetime.now()
        
        # Funding times are typically at 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        for hour in funding_hours:
            next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_time > now:
                return next_time
        
        # If no funding time today, get first one tomorrow
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    def _calculate_arbitrage_confidence(self, symbol: str, price_diff_pct: float, 
                                      binance_data: Dict, backpack_data: Dict) -> float:
        """Calculate confidence score for arbitrage opportunity."""
        try:
            confidence_factors = []
            
            # Price difference factor (higher difference = higher confidence up to a point)
            price_factor = min(price_diff_pct / 2.0, 1.0)  # Cap at 100%
            confidence_factors.append(price_factor * 0.3)
            
            # Volume factor (higher volume = higher confidence)
            binance_volume = float(binance_data.get('volume', 0))
            backpack_volume = float(backpack_data.get('volume', 0))
            min_volume = min(binance_volume, backpack_volume)
            volume_factor = min(min_volume / 1000000, 1.0)  # Normalize to $1M
            confidence_factors.append(volume_factor * 0.2)
            
            # Spread factor (tighter spreads = higher confidence)
            binance_bid = float(binance_data.get('bid', 0))
            binance_ask = float(binance_data.get('ask', 0))
            backpack_bid = float(backpack_data.get('bid', 0))
            backpack_ask = float(backpack_data.get('ask', 0))
            
            if all([binance_bid, binance_ask, backpack_bid, backpack_ask]):
                binance_spread = (binance_ask - binance_bid) / binance_bid
                backpack_spread = (backpack_ask - backpack_bid) / backpack_bid
                avg_spread = (binance_spread + backpack_spread) / 2
                spread_factor = max(0, 1 - avg_spread * 100)  # Lower spread = higher confidence
                confidence_factors.append(spread_factor * 0.2)
            else:
                confidence_factors.append(0.1)  # Default if spread data unavailable
            
            # Historical volatility factor
            volatility_risk = self._calculate_volatility_risk(symbol)
            volatility_factor = max(0, 1 - volatility_risk)
            confidence_factors.append(volatility_factor * 0.3)
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation error: {e}")
            return 0.5  # Default moderate confidence
    
    def _calculate_trade_sizing(self, symbol: str, binance_data: Dict, backpack_data: Dict) -> Tuple[float, float]:
        """Calculate minimum and maximum trade sizes."""
        try:
            # Base sizing on available liquidity
            binance_volume = float(binance_data.get('volume', 0))
            backpack_volume = float(backpack_data.get('volume', 0))
            
            # Conservative sizing: 0.1% of daily volume
            max_volume_based = min(binance_volume, backpack_volume) * 0.001
            
            # Minimum trade size (e.g., $100)
            min_trade_size = 100.0
            
            # Maximum trade size (limited by liquidity and risk parameters)
            max_trade_size = min(
                max_volume_based,
                self.config.get('max_arbitrage_size', 10000)  # $10k default max
            )
            
            return max(min_trade_size, 100), max(max_trade_size, min_trade_size)
            
        except Exception as e:
            logger.error(f"❌ Trade sizing calculation error: {e}")
            return 100.0, 1000.0  # Default values
    
    def _calculate_execution_priority(self, profit_pct: float, confidence: float) -> int:
        """Calculate execution priority (1=highest, 5=lowest)."""
        score = profit_pct * confidence
        
        if score > 2.0:
            return 1  # Highest priority
        elif score > 1.0:
            return 2
        elif score > 0.5:
            return 3
        elif score > 0.2:
            return 4
        else:
            return 5  # Lowest priority
    
    def _estimate_execution_time(self, symbol: str) -> float:
        """Estimate time to execute arbitrage (seconds)."""
        # Base execution time + market dependent factors
        base_time = 5.0  # 5 seconds base
        
        # Add volatility penalty
        volatility_risk = self._calculate_volatility_risk(symbol)
        volatility_penalty = volatility_risk * 10.0  # Up to 10 seconds penalty
        
        return base_time + volatility_penalty
    
    def _calculate_liquidity_score(self, binance_data: Dict, backpack_data: Dict) -> float:
        """Calculate liquidity score (0-1)."""
        try:
            volumes = [
                float(binance_data.get('volume', 0)),
                float(backpack_data.get('volume', 0))
            ]
            
            min_volume = min(volumes)
            # Normalize to $10M daily volume = score of 1.0
            return min(min_volume / 10000000, 1.0)
            
        except:
            return 0.5
    
    def _calculate_volatility_risk(self, symbol: str) -> float:
        """Calculate volatility risk (0-1)."""
        try:
            if (symbol not in self.price_history or 
                len(self.price_history[symbol]['binance']) < 20):
                return 0.5  # Default moderate risk
            
            # Calculate price volatility
            prices = self.price_history[symbol]['binance'][-20:]  # Last 20 prices
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(len(returns))
            
            # Normalize volatility (0.02 = high volatility = risk score 1.0)
            return min(volatility / 0.02, 1.0)
            
        except:
            return 0.5
    
    async def scan_all_opportunities(self, symbols: List[str]) -> Dict[str, List[Any]]:
        """Scan all symbols for arbitrage opportunities."""
        opportunities = {
            'price_arbitrage': [],
            'funding_rate': [],
            'basis_trading': []
        }
        
        try:
            for symbol in symbols:
                # Price arbitrage
                price_opp = await self.detect_price_arbitrage(symbol)
                if price_opp and price_opp.confidence_score >= self.detection_params['min_confidence_score']:
                    opportunities['price_arbitrage'].append(price_opp)
                
                # Funding rate arbitrage
                funding_opp = await self.detect_funding_rate_arbitrage(symbol)
                if funding_opp and funding_opp.confidence_score >= self.detection_params['min_confidence_score']:
                    opportunities['funding_rate'].append(funding_opp)
                
                # Basis trading
                basis_opps = await self.detect_basis_trading_opportunities(symbol)
                for opp in basis_opps:
                    if opp.confidence_score >= self.detection_params['min_confidence_score']:
                        opportunities['basis_trading'].append(opp)
                
        except Exception as e:
            logger.error(f"❌ Opportunity scanning error: {e}")
        
        return opportunities
    
    async def get_funding_rate_history(self, symbol: str, exchange: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical funding rates for analysis."""
        try:
            if exchange == "binance":
                url = "https://fapi.binance.com/fapi/v1/fundingRate"
                params = {"symbol": symbol, "limit": limit}
            elif exchange == "backpack":
                # Adjust based on actual Backpack API
                url = "https://api.backpack.exchange/api/v1/funding/history"
                params = {"symbol": symbol, "limit": limit}
            else:
                return []
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"❌ {exchange} funding history API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Failed to get funding rate history: {e}")
            return []
    
    def analyze_opportunity_trends(self, opportunities: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze trends in detected opportunities."""
        try:
            analysis = {
                'total_opportunities': sum(len(opps) for opps in opportunities.values()),
                'by_type': {},
                'top_opportunities': [],
                'risk_distribution': {},
                'profit_potential': {}
            }
            
            for opp_type, opps in opportunities.items():
                if opps:
                    analysis['by_type'][opp_type] = {
                        'count': len(opps),
                        'avg_confidence': sum(opp.confidence_score for opp in opps) / len(opps),
                        'max_profit': max(getattr(opp, 'profit_potential', 0) for opp in opps),
                        'avg_profit': sum(getattr(opp, 'profit_potential', 0) for opp in opps) / len(opps)
                    }
                    
                    # Add to top opportunities (sorted by profit potential)
                    for opp in opps:
                        profit = getattr(opp, 'profit_potential', 0)
                        if hasattr(opp, 'profit_potential_8h'):
                            profit = opp.profit_potential_8h
                        
                        analysis['top_opportunities'].append({
                            'type': opp_type,
                            'symbol': opp.symbol,
                            'profit': profit,
                            'confidence': opp.confidence_score,
                            'timestamp': opp.timestamp
                        })
            
            # Sort top opportunities by profit potential
            analysis['top_opportunities'].sort(key=lambda x: x['profit'], reverse=True)
            analysis['top_opportunities'] = analysis['top_opportunities'][:10]  # Top 10
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Opportunity trend analysis error: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detector performance metrics."""
        try:
            return {
                'total_scans': getattr(self, '_total_scans', 0),
                'opportunities_found': getattr(self, '_opportunities_found', 0),
                'success_rate': getattr(self, '_success_rate', 0.0),
                'avg_scan_time': getattr(self, '_avg_scan_time', 0.0),
                'last_scan': getattr(self, '_last_scan_time', None),
                'data_sources': {
                    'binance_symbols': len(self.binance_data),
                    'backpack_symbols': len(self.backpack_data),
                    'funding_rates_cached': len(self.funding_rates.get('binance', {})) + len(self.funding_rates.get('backpack', {})),
                    'futures_data_cached': len(self.futures_data.get('binance', {})) + len(self.futures_data.get('backpack', {}))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics error: {e}")
            return {}
    
    async def validate_opportunity(self, opportunity: Any) -> bool:
        """Validate an arbitrage opportunity before execution."""
        try:
            # Common validation checks
            if opportunity.confidence_score < self.detection_params['min_confidence_score']:
                return False
            
            # Type-specific validation
            if isinstance(opportunity, ArbitrageOpportunity):
                return await self._validate_price_arbitrage(opportunity)
            elif isinstance(opportunity, FundingRateOpportunity):
                return await self._validate_funding_arbitrage(opportunity)
            elif isinstance(opportunity, BasisTradingOpportunity):
                return await self._validate_basis_trading(opportunity)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Opportunity validation error: {e}")
            return False
    
    async def _validate_price_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate price arbitrage opportunity."""
        try:
            # Re-fetch current prices to confirm opportunity still exists
            current_binance_data = self.binance_data.get(opportunity.symbol)
            current_backpack_data = self.backpack_data.get(opportunity.symbol)
            
            if not current_binance_data or not current_backpack_data:
                return False
            
            current_binance_price = float(current_binance_data.get('price', 0))
            current_backpack_price = float(current_backpack_data.get('price', 0))
            
            # Check if price difference still exists
            current_diff_pct = abs(current_binance_price - current_backpack_price) / min(current_binance_price, current_backpack_price) * 100
            
            # Allow some tolerance for price movement
            tolerance = 0.1  # 0.1%
            return current_diff_pct >= (opportunity.price_diff_pct - tolerance)
            
        except Exception as e:
            logger.error(f"❌ Price arbitrage validation error: {e}")
            return False
    
    async def _validate_funding_arbitrage(self, opportunity: FundingRateOpportunity) -> bool:
        """Validate funding rate arbitrage opportunity."""
        try:
            # Check if funding time is approaching
            time_to_funding = (opportunity.next_funding_time - datetime.now()).total_seconds()
            
            # Must have at least 30 minutes before funding
            if time_to_funding < 1800:  # 30 minutes
                return False
            
            # Re-fetch funding rates to confirm
            binance_data = await self._fetch_binance_funding_rate(opportunity.symbol)
            backpack_data = await self._fetch_backpack_funding_rate(opportunity.symbol)
            
            if not binance_data or not backpack_data:
                return False
            
            current_rate_diff = abs(binance_data['funding_rate'] - backpack_data['funding_rate'])
            
            # Allow some tolerance
            tolerance = 0.00005  # 0.005%
            return current_rate_diff >= (opportunity.rate_diff - tolerance)
            
        except Exception as e:
            logger.error(f"❌ Funding arbitrage validation error: {e}")
            return False
    
    async def _validate_basis_trading(self, opportunity: BasisTradingOpportunity) -> bool:
        """Validate basis trading opportunity."""
        try:
            # Check if sufficient time to expiry
            if opportunity.time_to_expiry.total_seconds() < 86400:  # At least 1 day
                return False
            
            # Re-fetch current prices
            current_spot_price = await self._get_spot_price(opportunity.exchange, opportunity.symbol)
            if not current_spot_price:
                return False
            
            # For futures price, would need to re-fetch from specific contract
            # For now, assume it's still valid if spot price hasn't moved too much
            price_change_pct = abs(current_spot_price - opportunity.spot_price) / opportunity.spot_price * 100
            
            # If spot price moved more than 2%, re-evaluate
            return price_change_pct < 2.0
            
        except Exception as e:
            logger.error(f"❌ Basis trading validation error: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    config = {
        'trading_costs': {
            'binance_spot_fee': 0.001,
            'backpack_spot_fee': 0.001,
            'binance_futures_fee': 0.0004,
            'backpack_futures_fee': 0.0004,
            'slippage_estimate': 0.0005,
            'min_profit_threshold': 0.003  # 0.3% minimum
        },
        'detection_params': {
            'min_confidence_score': 0.1,  # Lowered for testing
            'min_funding_rate_diff': 0.0001,
            'min_basis_threshold': 0.005
        },
        'max_arbitrage_size': 5000
    }
    
    detector = ArbitrageDetector(config)
    
    # Example: comprehensive arbitrage detection testing
    async def test_comprehensive_detector():
        print("🎯 ENHANCED ARBITRAGE DETECTOR TEST")
        print("=" * 50)
        
        try:
            # Test symbols
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            # Simulate market data for price arbitrage
            await detector.update_market_data("binance", "BTCUSDT", {
                'price': 50000.0,
                'bid': 49995.0,
                'ask': 50005.0,
                'volume': 1000000
            })
            
            await detector.update_market_data("backpack", "BTCUSDT", {
                'price': 50200.0,  # $200 higher than Binance
                'bid': 50195.0,
                'ask': 50205.0,
                'volume': 500000
            })
            
            # Test 1: Price Arbitrage Detection
            print("\n1. PRICE ARBITRAGE DETECTION")
            print("-" * 30)
            price_opp = await detector.detect_price_arbitrage("BTCUSDT")
            if price_opp:
                print(f"✅ Price arbitrage found:")
                print(f"   Symbol: {price_opp.symbol}")
                print(f"   Profit: {price_opp.profit_potential_pct:.3f}%")
                print(f"   Direction: {price_opp.direction.value}")
                print(f"   Confidence: {price_opp.confidence_score:.3f}")
                print(f"   Priority: {price_opp.execution_priority}")
            else:
                print("❌ No price arbitrage found")
            
            # Test 2: Funding Rate Arbitrage Detection
            print("\n2. FUNDING RATE ARBITRAGE DETECTION")
            print("-" * 40)
            funding_opp = await detector.detect_funding_rate_arbitrage("BTCUSDT")
            if funding_opp:
                print(f"✅ Funding rate arbitrage found:")
                print(f"   Symbol: {funding_opp.symbol}")
                print(f"   Binance Rate: {funding_opp.binance_funding_rate:.6f}")
                print(f"   Backpack Rate: {funding_opp.backpack_funding_rate:.6f}")
                print(f"   Rate Diff: {funding_opp.rate_diff:.6f}")
                print(f"   Annualized Return: {funding_opp.rate_diff_annualized:.3f}%")
                print(f"   8h Profit: ${funding_opp.profit_potential_8h:.2f}")
                print(f"   Position: {funding_opp.position_direction}")
                print(f"   Next Funding: {funding_opp.next_funding_time}")
                print(f"   Required Margin: ${funding_opp.required_margin:.2f}")
                print(f"   Risk Score: {funding_opp.risk_score:.3f}")
            else:
                print("❌ No funding rate arbitrage found")
            
            # Test 3: Basis Trading Detection
            print("\n3. BASIS TRADING DETECTION")
            print("-" * 30)
            basis_opps = await detector.detect_basis_trading_opportunities("BTCUSDT")
            if basis_opps:
                print(f"✅ Found {len(basis_opps)} basis trading opportunities:")
                for i, opp in enumerate(basis_opps):
                    print(f"   Opportunity {i+1}:")
                    print(f"     Exchange: {opp.exchange}")
                    print(f"     Spot Price: ${opp.spot_price:.2f}")
                    print(f"     Futures Price: ${opp.futures_price:.2f}")
                    print(f"     Basis: ${opp.basis:.2f} ({opp.basis_pct:.3f}%)")
                    print(f"     Type: {opp.position_type}")
                    print(f"     Annualized Return: {opp.annualized_return:.3f}%")
                    print(f"     Time to Expiry: {opp.time_to_expiry.days} days")
                    print(f"     Confidence: {opp.confidence_score:.3f}")
                    print(f"     Liquidity Score: {opp.liquidity_score:.3f}")
                    print(f"     Risk Score: {opp.risk_score:.3f}")
            else:
                print("❌ No basis trading opportunities found")
            
            # Test 4: Comprehensive Scan
            print("\n4. COMPREHENSIVE OPPORTUNITY SCAN")
            print("-" * 40)
            all_opportunities = await detector.scan_all_opportunities(symbols)
            
            print(f"📊 SCAN RESULTS:")
            print(f"   Price Arbitrage: {len(all_opportunities['price_arbitrage'])} opportunities")
            print(f"   Funding Rate: {len(all_opportunities['funding_rate'])} opportunities")
            print(f"   Basis Trading: {len(all_opportunities['basis_trading'])} opportunities")
            
            # Test 5: Opportunity Analysis
            print("\n5. OPPORTUNITY TREND ANALYSIS")
            print("-" * 35)
            analysis = detector.analyze_opportunity_trends(all_opportunities)
            print(f"📈 ANALYSIS RESULTS:")
            print(f"   Total Opportunities: {analysis.get('total_opportunities', 0)}")
            
            for opp_type, stats in analysis.get('by_type', {}).items():
                print(f"   {opp_type.upper()}:")
                print(f"     Count: {stats['count']}")
                print(f"     Avg Confidence: {stats['avg_confidence']:.3f}")
                print(f"     Max Profit: ${stats['max_profit']:.2f}")
                print(f"     Avg Profit: ${stats['avg_profit']:.2f}")
            
            # Test 6: Performance Metrics
            print("\n6. DETECTOR PERFORMANCE METRICS")
            print("-" * 35)
            metrics = detector.get_performance_metrics()
            print(f"🔧 PERFORMANCE:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"   {key.upper()}:")
                    for subkey, subvalue in value.items():
                        print(f"     {subkey}: {subvalue}")
                else:
                    print(f"   {key}: {value}")
            
            # Test 7: Opportunity Validation
            if price_opp:
                print("\n7. OPPORTUNITY VALIDATION")
                print("-" * 30)
                is_valid = await detector.validate_opportunity(price_opp)
                print(f"✅ Price arbitrage validation: {'PASSED' if is_valid else 'FAILED'}")
            
            print("\n" + "=" * 50)
            print("✅ ENHANCED ARBITRAGE DETECTOR TEST COMPLETE")
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            
        finally:
            # Clean up
            await detector.close()
    
    # Run the comprehensive test
    asyncio.run(test_comprehensive_detector())