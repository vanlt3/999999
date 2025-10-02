"""
Configuration Management System for Advanced Trading Bot
Centralized configuration for all trading bot components
"""

import os
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class APIConfig:
    """API Configuration"""
    # Trading APIs
    vantage_api_key: str = "FK3YQ1IKSC4E1AL5"
    finnhub_api_key: str = "d1b3ichr01qjhvtsbj8g"
    marketaux_api_key: str = "CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql"
    newsapi_key: str = "abd8f43b808f42fdb8d28fb1c429af72"
    eodhd_api_key: str = "68bafd7d44a7f0.25202650"
    
    # OANDA Trading Configuration
    oanda_api_key: str = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"
    oanda_url: str = "https://api-fxtrade.oanda.com/v3"
    
    # Rate limits (requests per minute)
    rate_limits: Dict[str, int] = None
    
    def __post_init__(self):
        if self.rate_limits is None:
            self.rate_limits = {
                "alpha_vantage": 500,
                "finnhub": 60,
                "marketaux": 100,
                "newsapi": 1000,
                "eodhd": 1000,
                "oanda": 2000
            }

@dataclass
class TradingConfig:
    """Trading Configuration"""
    # Trading symbols
    symbols: List[str] = None
    
    # Risk parameters
    max_position_size: float = 0.02  # 2% of account per position
    max_total_exposure: float = 0.1   # 10% total exposure
    correlation_threshold: float = 0.7  # Max correlation between positions
    
    # Stop Loss/Take Profit parameters
    default_sl_multiplier: float = 2.0  # SL = entry_price + ATR * multiplier
    default_tp_multiplier: float = 3.0  # TP = entry_price + ATR * multiplier
    
    # Trailing stop parameters
    trailing_stop_atr_multiplier: float = 1.5
    trailing_stop_min_profit: float = 0.005  # Minimum 0.5% profit before trailing
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["XAUUSD", "EURUSD", "NASDAQ100", "BTCUSD"]

@dataclass
class MLConfig:
    """Machine Learning Configuration"""
    # Model parameters
    ensemble_models: List[str] = None
    lstm_sequence_length: int = 60
    lstm_hidden_units: int = 100
    lstm_dropout_rate: float = 0.3
    
    # Training parameters
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    retrain_threshold_performance: float = -0.05  # Retrain if profit < -5%
    
    # Feature engineering
    technical_indicators: List[str] = None
    min_samples_for_training: int = 1000
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ["xgboost", "lightgbm", "random_forest"]
        
        if self.technical_indicators is None:
            self.technical_indicators = [
                "sma", "ema", "rsi", "macd", "atr", "adx", "bb", "stoch", 
                "cci", "willr", "roc", "momentum", "ad", "obv", "vwap"
            ]

@dataclass
class SecurityConfig:
    """Security Configuration"""
    # Weekend trading (crypto only)
    allow_weekend_trading_crypto: bool = True
    weekend_position_close_hour: int = 22  # Friday 22:00 UTC
    
    # News filtering
    news_gating_events: List[str] = None
    news_gating_hours: float = 2.0  # Hours before/after major events
    
    def __post_init__(self):
        if self.news_gating_events is None:
            self.news_gating_events = [
                "nonfarm", "cpi", "fomc", "ecb", "rba", "boe", "employment"
            ]

class LogLevel(Enum):
    """Log levels with emoji indicators"""
    TRADE = "🚨"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    DEBUG = "🔍"
    ML = "🤖"
    DATA = "📊"
    SYSTEM = "⚙️"

# Global configuration instance
api_config = APIConfig()
trading_config = TradingConfig()
ml_config = MLConfig()
security_config = SecurityConfig()

# Database configuration
DATABASE_URL = "sqlite:///trading_bot.db"

# Discord configuration
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1419645732218081290/xamfJQdl5kay1wo6w6gxQRrW77d1jpSzKBstQ16Qvb4t5ncGJ3nIHMmm3MQPNT_E-Bt2"

# Market timeframe configuration
TIMEFRAMES = {
    "M15": "15m",
    "H1": "1h", 
    "H4": "4h",
    "D1": "1d",
    "W1": "1w"
}

# Asset type mapping
ASSET_TYPES = {
    "XAUUSD": "commodity",
    "EURUSD": "forex",
    "NASDAQ100": "index",
    "BTCUSD": "crypto"
}

# Market hours (UTC)
MARKET_HOURS = {
    "forex": {"open": 0, "close": 24},  # 24/5
    "crypto": {"open": 0, "close": 24},  # 24/7
    "commodity": {"open": 0, "close": 24},  # Similar to forex
    "index": {"open": 9, "close": 16}  # Business hours
}

def get_config() -> Dict[str, Any]:
    """Return all configuration as dictionary"""
    return {
        "api": api_config,
        "trading": trading_config,
        ":ml": ml_config,
        "security": security_config,
        "database_url": DATABASE_URL,
        "discord_webhook": DISCORD_WEBHOOK_URL,
        "timeframes": TIMEFRAMES,
        "asset_types": ASSET_TYPES,
        "market_hours": MARKET_HOURS
    }