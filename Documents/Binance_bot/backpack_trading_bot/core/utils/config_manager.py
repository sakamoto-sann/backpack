"""
Unified Configuration Management System
Handles loading and validation of all bot configurations
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration settings"""
    key: str
    secret: str
    testnet: bool = False

@dataclass
class GridTradingConfig:
    """Grid trading strategy configuration"""
    base_currency: str
    quote_currency: str
    trading_pair: str
    grid_count: int
    grid_spacing: float
    base_order_size: float
    supertrend_enabled: bool = True
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adaptive_supertrend_enabled: bool = True
    adaptive_supertrend_base_period: int = 10
    adaptive_supertrend_base_multiplier: float = 2.5
    supertrend_signal_weight: float = 0.3
    signal_agreement_bonus: float = 0.05

@dataclass
class MarketRegimeConfig:
    """Market regime detection configuration"""
    ma_fast: int
    ma_slow: int
    atr_period: int
    volatility_periods: List[int]
    breakout_threshold: float
    sideways_threshold: float
    supertrend_enabled: bool = True
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    adaptive_supertrend_enabled: bool = True
    adaptive_supertrend_base_period: int = 10
    adaptive_supertrend_base_multiplier: float = 2.5
    supertrend_signal_weight: float = 0.3
    signal_agreement_bonus: float = 0.05

@dataclass
class ArbitrageConfig:
    """Arbitrage strategy configuration"""
    trading_pairs: List[List[str]]
    trade_amount: float
    min_profit_threshold: float
    min_trade_interval: int

@dataclass
class DeltaNeutralConfig:
    """Delta neutral strategy configuration"""
    spot_symbol: str
    futures_symbol: str
    base_position_size: float
    hedge_ratio: float
    grid_levels: int
    grid_spacing: float
    rebalance_threshold: float

@dataclass
class AdvancedConfig:
    """Advanced multi-strategy configuration"""
    strategy_weights: Dict[str, float]
    timeframes: List[str]
    signal_weights: List[float]

@dataclass
class StrategyConfig:
    """Combined strategy configuration"""
    type: str
    grid_trading: GridTradingConfig
    arbitrage: ArbitrageConfig
    delta_neutral: DeltaNeutralConfig
    advanced: AdvancedConfig

@dataclass
class PositionConfig:
    """Position management configuration"""
    max_position_size: float
    sizing_method: str
    base_position_pct: float
    allocation: Dict[str, float]

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_daily_loss: float
    max_drawdown: float
    max_position_size: float
    stop_loss_enabled: bool
    stop_loss_pct: float
    take_profit_enabled: bool
    take_profit_pct: float
    max_volatility_pct: float
    min_volume_24h: float

@dataclass
class NotificationConfig:
    """Notification configuration"""
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    email_enabled: bool
    email_settings: Dict[str, str]
    triggers: List[str]

@dataclass
class BotConfig:
    """Main bot configuration"""
    api: APIConfig
    strategy: StrategyConfig
    position: PositionConfig
    risk: RiskConfig
    notifications: NotificationConfig
    market_regime: MarketRegimeConfig
    logging_level: str = "INFO"
    paper_trading: bool = True
    debug_mode: bool = False

class ConfigManager:
    """Unified configuration management system"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache: Dict[str, Any] = {}
        self.environment_vars: Dict[str, str] = {}
        
        # Load environment variables
        self._load_environment_variables()
        
        logger.info(f"Configuration manager initialized with config dir: {config_dir}")
    
    def _load_environment_variables(self):
        """Load environment variables for configuration substitution"""
        env_vars = [
            'BINANCE_API_KEY', 'BINANCE_API_SECRET',
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID',
            'SMTP_SERVER', 'SMTP_USERNAME', 'SMTP_PASSWORD',
            'NOTIFICATION_EMAIL'
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                self.environment_vars[var] = value
            else:
                logger.warning(f"Environment variable {var} not set")
    
    def load_config(self, config_files: Optional[List[str]] = None) -> BotConfig:
        """
        Load complete bot configuration from YAML files.
        
        Args:
            config_files: List of config files to load. If None, loads default files.
            
        Returns:
            Complete bot configuration
        """
        try:
            if config_files is None:
                config_files = [
                    'trading_config.yaml',
                    'risk_config.yaml',
                    'api_config.yaml'
                ]
            
            # Load and merge configurations
            merged_config = {}
            for config_file in config_files:
                file_path = self.config_dir / config_file
                if file_path.exists():
                    config_data = self._load_yaml_file(file_path)
                    merged_config = self._deep_merge(merged_config, config_data)
                else:
                    logger.warning(f"Config file not found: {file_path}")
            
            # Parse into structured configuration
            bot_config = self._parse_config(merged_config)
            
            # Validate configuration
            self._validate_config(bot_config)
            
            logger.info("Configuration loaded successfully")
            return bot_config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse YAML file with environment variable substitution"""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse YAML
            config_data = yaml.safe_load(content)
            
            logger.debug(f"Loaded config file: {file_path}")
            return config_data
            
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content"""
        try:
            template = Template(content)
            return template.safe_substitute(self.environment_vars)
        except Exception as e:
            logger.error(f"Error substituting environment variables: {e}")
            return content
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _parse_config(self, config_data: Dict[str, Any]) -> BotConfig:
        """Parse configuration data into structured objects"""
        try:
            # API Configuration
            api_config = APIConfig(
                key=config_data.get('api', {}).get('key', ''),
                secret=config_data.get('api', {}).get('secret', ''),
                testnet=config_data.get('api', {}).get('testnet', False)
            )
            
            # Strategy Configuration
            strategy_data = config_data.get('strategy', {})
            
            # Grid Trading Config
            grid_data = strategy_data.get('grid_trading', {})
            market_regime_data = grid_data.get('market_regime', {})
            
            market_regime_config = MarketRegimeConfig(
                ma_fast=market_regime_data.get('ma_fast', 12),
                ma_slow=market_regime_data.get('ma_slow', 26),
                atr_period=market_regime_data.get('atr_period', 14),
                volatility_periods=market_regime_data.get('volatility_periods', [20, 50, 100]),
                breakout_threshold=market_regime_data.get('breakout_threshold', 2.0),
                sideways_threshold=market_regime_data.get('sideways_threshold', 0.02),
                supertrend_enabled=grid_data.get('supertrend_enabled', True),
                supertrend_period=grid_data.get('supertrend_period', 10),
                supertrend_multiplier=grid_data.get('supertrend_multiplier', 3.0),
                adaptive_supertrend_enabled=grid_data.get('adaptive_supertrend_enabled', True),
                adaptive_supertrend_base_period=grid_data.get('adaptive_supertrend_base_period', 10),
                adaptive_supertrend_base_multiplier=grid_data.get('adaptive_supertrend_base_multiplier', 2.5),
                supertrend_signal_weight=grid_data.get('supertrend_signal_weight', 0.3),
                signal_agreement_bonus=grid_data.get('signal_agreement_bonus', 0.05)
            )
            
            grid_config = GridTradingConfig(
                base_currency=grid_data.get('base_currency', 'USDT'),
                quote_currency=grid_data.get('quote_currency', 'BTC'),
                trading_pair=grid_data.get('trading_pair', 'BTCUSDT'),
                grid_count=grid_data.get('grid_count', 20),
                grid_spacing=grid_data.get('grid_spacing', 0.005),
                base_order_size=grid_data.get('base_order_size', 100),
                supertrend_enabled=grid_data.get('supertrend_enabled', True),
                supertrend_period=grid_data.get('supertrend_period', 10),
                supertrend_multiplier=grid_data.get('supertrend_multiplier', 3.0),
                adaptive_supertrend_enabled=grid_data.get('adaptive_supertrend_enabled', True),
                adaptive_supertrend_base_period=grid_data.get('adaptive_supertrend_base_period', 10),
                adaptive_supertrend_base_multiplier=grid_data.get('adaptive_supertrend_base_multiplier', 2.5),
                supertrend_signal_weight=grid_data.get('supertrend_signal_weight', 0.3),
                signal_agreement_bonus=grid_data.get('signal_agreement_bonus', 0.05)
            )
            
            # Arbitrage Config
            arbitrage_data = strategy_data.get('arbitrage', {})
            arbitrage_config = ArbitrageConfig(
                trading_pairs=arbitrage_data.get('trading_pairs', []),
                trade_amount=arbitrage_data.get('trade_amount', 100),
                min_profit_threshold=arbitrage_data.get('min_profit_threshold', 0.1),
                min_trade_interval=arbitrage_data.get('min_trade_interval', 30)
            )
            
            # Delta Neutral Config
            delta_data = strategy_data.get('delta_neutral', {})
            delta_config = DeltaNeutralConfig(
                spot_symbol=delta_data.get('spot_symbol', 'BTCUSDT'),
                futures_symbol=delta_data.get('futures_symbol', 'BTCUSDT'),
                base_position_size=delta_data.get('base_position_size', 1000),
                hedge_ratio=delta_data.get('hedge_ratio', 0.95),
                grid_levels=delta_data.get('grid_levels', 10),
                grid_spacing=delta_data.get('grid_spacing', 0.003),
                rebalance_threshold=delta_data.get('rebalance_threshold', 0.05)
            )
            
            # Advanced Config
            advanced_data = strategy_data.get('advanced', {})
            advanced_config = AdvancedConfig(
                strategy_weights=advanced_data.get('strategy_weights', {}),
                timeframes=advanced_data.get('timeframes', []),
                signal_weights=advanced_data.get('signal_weights', [])
            )
            
            strategy_config = StrategyConfig(
                type=strategy_data.get('type', 'grid_trading'),
                grid_trading=grid_config,
                arbitrage=arbitrage_config,
                delta_neutral=delta_config,
                advanced=advanced_config
            )
            
            # Position Configuration
            position_data = config_data.get('position', {})
            position_config = PositionConfig(
                max_position_size=position_data.get('max_position_size', 0.8),
                sizing_method=position_data.get('sizing_method', 'percentage'),
                base_position_pct=position_data.get('base_position_pct', 0.1),
                allocation=position_data.get('allocation', {})
            )
            
            # Risk Configuration
            risk_data = config_data.get('account', {})
            risk_config = RiskConfig(
                max_daily_loss=risk_data.get('max_daily_loss', 500),
                max_drawdown=risk_data.get('max_drawdown', 0.15),
                max_position_size=risk_data.get('max_position_size', 0.8),
                stop_loss_enabled=True,
                stop_loss_pct=0.02,
                take_profit_enabled=True,
                take_profit_pct=0.04,
                max_volatility_pct=0.05,
                min_volume_24h=1000000
            )
            
            # Notification Configuration
            notification_data = config_data.get('notifications', {})
            telegram_data = notification_data.get('telegram', {})
            
            notification_config = NotificationConfig(
                telegram_enabled=telegram_data.get('enabled', False),
                telegram_bot_token=telegram_data.get('bot_token', ''),
                telegram_chat_id=telegram_data.get('chat_id', ''),
                email_enabled=False,
                email_settings={},
                triggers=telegram_data.get('triggers', [])
            )
            
            # Create main configuration
            bot_config = BotConfig(
                api=api_config,
                strategy=strategy_config,
                position=position_config,
                risk=risk_config,
                notifications=notification_config,
                market_regime=market_regime_config,
                logging_level=config_data.get('logging', {}).get('level', 'INFO'),
                paper_trading=config_data.get('development', {}).get('paper_trading', True),
                debug_mode=config_data.get('development', {}).get('debug_mode', False)
            )
            
            return bot_config
            
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def _validate_config(self, config: BotConfig):
        """Validate configuration for correctness and completeness"""
        try:
            # Validate API configuration
            if not config.api.key or not config.api.secret:
                raise ValueError("API key and secret are required")
            
            # Validate strategy configuration
            if config.strategy.type not in ['grid_trading', 'arbitrage', 'delta_neutral', 'advanced']:
                raise ValueError(f"Invalid strategy type: {config.strategy.type}")
            
            # Validate risk parameters
            if config.risk.max_daily_loss <= 0:
                raise ValueError("Max daily loss must be positive")
            
            if config.risk.max_drawdown <= 0 or config.risk.max_drawdown >= 1:
                raise ValueError("Max drawdown must be between 0 and 1")
            
            # Validate position sizing
            if config.position.max_position_size <= 0 or config.position.max_position_size > 1:
                raise ValueError("Max position size must be between 0 and 1")
            
            # Validate grid trading parameters
            if config.strategy.type == 'grid_trading':
                if config.strategy.grid_trading.grid_count <= 0:
                    raise ValueError("Grid count must be positive")
                
                if config.strategy.grid_trading.grid_spacing <= 0:
                    raise ValueError("Grid spacing must be positive")
            
            # Validate arbitrage parameters
            if config.strategy.type == 'arbitrage':
                if not config.strategy.arbitrage.trading_pairs:
                    raise ValueError("Trading pairs are required for arbitrage")
                
                if config.strategy.arbitrage.min_profit_threshold <= 0:
                    raise ValueError("Min profit threshold must be positive")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def save_config(self, config: BotConfig, filename: str = "current_config.yaml"):
        """Save current configuration to file"""
        try:
            config_dict = self._config_to_dict(config)
            
            output_path = self.config_dir / filename
            with open(output_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def _config_to_dict(self, config: BotConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        # This is a simplified implementation
        # In practice, you'd want to properly serialize all configuration objects
        return {
            'api': {
                'key': config.api.key,
                'secret': config.api.secret,
                'testnet': config.api.testnet
            },
            'strategy': {
                'type': config.strategy.type
            },
            'logging_level': config.logging_level,
            'paper_trading': config.paper_trading,
            'debug_mode': config.debug_mode
        }
    
    def get_config_summary(self, config: BotConfig) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            'strategy_type': config.strategy.type,
            'paper_trading': config.paper_trading,
            'debug_mode': config.debug_mode,
            'max_daily_loss': config.risk.max_daily_loss,
            'max_position_size': config.position.max_position_size,
            'notifications_enabled': config.notifications.telegram_enabled,
            'api_testnet': config.api.testnet
        }

# Global configuration instance
config_manager = ConfigManager()

def load_config(config_files: Optional[List[str]] = None) -> BotConfig:
    """Convenience function to load configuration"""
    return config_manager.load_config(config_files)

def get_config_summary(config: BotConfig) -> Dict[str, Any]:
    """Convenience function to get configuration summary"""
    return config_manager.get_config_summary(config)