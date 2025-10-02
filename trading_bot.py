#!/usr/bin/env python3
"""
Bot Giao dịch Tự động Hoàn chính với AI/ML
================================================

Một bot giao dịch tự động với các tính năng:
- Hệ thống Machine Learning Ensemble với XGBoost, LightGBM, Random Forest
- Reinforcement Learning với PPO
- Phân tích tin tức với AI (Gemini 1.5 Flash)
- Quản lý rủi ro đa tầng
- Giám sát real-time và báo cáo Discord

Author: AI Trading Bot System
Version: 1.0.0
Email: ai@trading.com
Created: 2024
"""

# ========================================
# PART 1: IMPORTS & INITIAL CONFIGURATION
# ========================================

import os
import sys
import asyncio
import warnings
import json
import datetime
import time
import logging
import sqlite3
import threading
import hashlib
import math
import random
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback
import functools
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64
from urllib.parse import urlencode
import re

# Tối ưu hóa hiệu suất và tắt cảnh báo không cần thiết
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import các thư viện chính
try:
    import pandas as pd
    import numpy as np
    import requests
    import aiohttp
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    from optuna.integration import OptunaSearchCV
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    import ta
    from river import linear_model, preprocessing, ensemble
    from river import optim
    import openai
    import discord
    from discord import Embed
except ImportError as e:
    print(f"Lỗi: Thiếu thư viện {e}")
    print("Vui lòng chạy: pip install -r requirements.txt")
    sys.exit(1)

# ========================================
# PART 2: GLOBAL CONFIGURATION
# ========================================

# API Keys và Configuration
ALPHA_VANTAGE_API_KEY = "FK3YQ1IKSC4E1AL5"
FINNHUB_API_KEY = "d1b3ichr01qjhvtsbj8g"
MARKETAUX_API_KEY = "CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql"
NEWSAPI_API_KEY = "abd8f43b808f42fdb8d28fb1c429af72"
EODHD_API_KEY = "68bafd7d44a7f0.25202650"
OANDA_API_KEY = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"

# Endpoints
OANDA_URL = "https://api-fxtrade.oanda.com/v3"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1419645732218081290/xamfJQdl5kay1wo6w6gxQRrW77d1jpSzKBstQ16Qvb4t5ncGJ3nIHMmm3MQPNT_E-Bt2"
GEMINI_API_KEY = ""  # Sẽ được cấu hình sau
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Trading Configuration
SYMBOLS = ["XAUUSD", "EURUSD", "NAS100", "BTCUSD"]
SYMBOL_MAPPING = {
    "NAS100": "NAS100_USD",
    "BTCUSD": "USD_JPY",  # Placeholder cho crypto
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD"
}

# Risk Management
MAX_POSITIONS = 4
MAX_DAILY_TRADES = 20
MAX_DRAWDOWN = 0.05
PORTFOLIO_VALUE = 10000.0

# Timeframes
TIMEFRAMES = ['M15', 'H1', 'H4', 'D1']

# Model Configuration
MODEL_RETRAIN_FREQUENCY = 24  # hours
CONCEPT_DRIFT_THRESHOLD = 0.1
OVERFIT_CONFIDENCE_THRESHOLD = 0.95

# ========================================
# PART 3: ENUMS & DATA CLASSES
# ========================================

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"

class SpecialistType(Enum):
    TREND = "TREND"
    NEWS = "NEWS"
    RISK = "RISK"
    SENTIMENT = "SENTIMENT"
    VOLATILITY = "VOLATILITY"
    PORTFOLIO = "PORTFOLIO"

@dataclass
class TradeSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime.datetime
    features: Dict[str, float]
    specialist_votes: Dict[SpecialistType, SignalType] = field(default_factory=dict)
    
class TradingPosition:
    def __init__(self, symbol: str, side: str, size: float, entry_price: float,
                 stop_loss: float, take_profit: float, timestamp: datetime.datetime):
        self.id = hashlib.md5(f"{symbol}{side}{timestamp}".encode()).hexdigest()[:8]
        self.symbol = symbol
        self.side = side  # LONG hoặc SHORT
        self.size = size
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = timestamp
        self.unrealized_pnl = 0.0
        self.status = "OPEN"
        self.trailing_stop_distance = 0.0
        self.trailing_stop_active = False

@dataclass
class MarketData:
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime.datetime
    features: Dict[str, float] = field(default_factory=dict)

@dataclass
class NewsData:
    title: str
    content: str
    source: str
    timestamp: datetime.datetime
    sentiment_score: float
    impact_level: str
    symbols_mentioned: List[str] = field(default_factory=list)

# ========================================
# PART 4: ADVANCED LOGGING SYSTEM
# ========================================

class ColoredFormatter(logging.Formatter):
    """Formatter tùy chỉnh với màu sắc và biểu tượng cảm xúc"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    EMOTICONS = {
        'DEBUG': '🔍',
        'INFO': '📢',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        record.levelname = (f"{self.COLORS.get(record.levelname, '')}"
                          f"{self.EMOTICONS.get(record.levelname, '')} "
                          f"{record.levelname}{self.RESET}")
        return super().format(record)

class AdvancedLogManager:
    """Hệ thống logging nâng cao với phân tách theo module"""
    
    def __init__(self):
        self.loggers = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Thiết lập hệ thống logging"""
        
        # Formatter có màu sắc
        colored_formatter = ColoredFormatter(
            '%(name)-12s | %(levelname)-8s | %(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Formatter cho file log
        file_formatter = logging.Formatter(
            '%(name)-12s | %(levelname)-8s | %(asctime)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Logger chính
        main_logger = logging.getLogger('TradingBot')
        main_logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        main_logger.addHandler(console_handler)
        
        # File handler
        log_file = 'logs/trading_bot.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        main_logger.addHandler(file_handler)
        
        # Logger cho các module cụ thể
        module_names = [
            'BotCore', 'DataManager', 'FeatureEngineer', 'NewsManager',
            'EnsembleModel', 'RLAgent', 'MasterAgent', 'RiskManager', 'Discord'
        ]
        
        for module_name in module_names:
            module_logger = logging.getLogger(f'TradingBot.{module_name}')
            module_logger.setLevel(logging.DEBUG)
            
            # Console handler cho từng module
            module_console_handler = logging.StreamHandler()
            module_console_handler.setFormatter(colored_formatter)
            module_logger.addHandler(module_console_handler)
            
            # File handler cho từng module
            module_file_handler = logging.FileHandler(log_file, encoding='utf-8')
            module_file_handler.setFormatter(file_formatter)
            module_logger.addHandler(module_file_handler)
            
            self.loggers[module_name] = module_logger
    
    def get_logger(self, module_name: str):
        """Lấy logger cho module cụ thể"""
        return self.loggers.get(module_name, logging.getLogger(f'TradingBot.{module_name}'))
    
├    def create_summary_report(self) -> str:
        """Tạo báo cáo tóm tắt từ log file"""
        try:
            log_file = 'logs/trading_bot.log'
            if not os│.path.exists(log_file):
                return "Không có log file để phân tích"
            
            # Đọc 100 dòng cuối cùng
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-100:]
            
            # Thống kê các loại log
            stats = {'ERROR': 0, 'WARNING': 0, 'INFO': 0, 'CRITICAL': 0}
            for line in lines:
                for level in stats.keys():
                    if level in line:
                        stats[level] += 1
            
            report = "=== BÁO CÁO LOG TÓM TẮT ===\n"
            report += f"📊 Tổng số log: {len(lines)}\n"
            report += f"❌ Lỗi: {stats['ERROR']}\n"
            report += f"⚠️ Cảnh báo: {stats['WARNING']}\n"
            report += f"📢 Thông tin: {stats['INFO']}\n"
            report += f"🚨 Nghiêm trọng: {stats['CRITICAL']}\n"
            
            return report
            
        except Exception as e:
            return f"Lỗi tạo báo cáo: {str(e)}"

# Khởi tạo hệ thống logging
log_manager = AdvancedLogManager()
main_logger = log_manager.get_logger('BotCore')

# ========================================
# PART 5: API MANAGEMENT SYSTEM  
# ========================================

class APIHealthMonitor:
    """Theo dõi sức khỏe của các API"""
    
    def __init__(self):
        self.health_status = {}
        self.request_counts = {}
        self.last_check = {}
    
    def record_request(self, api_name: str, success: bool, response_time: float):
        """Ghi nhận kết quả request"""
        if api_name not in self.request_counts:
            self.request_counts[api_name] = {'total': 0, 'success': 0, 'failed': 0}
        
        self.request_counts[api_name]['total'] += 1
        if success:
            self.request_counts[api_name]['success'] += 1
        else:
            self.request_counts[api_name]['failed'] += 1
        
        # Cập nhật health status
        success_rate = self.request_counts[api_name]['success'] / self.request_counts[api_name]['total']
        self.health_status[api_name] = {
            'healthy': success_rate > 0.8,
            'success_rate': success_rate,
            'response_time': response_time,
            'last_check': datetime.datetime.now()
        }

class APIManager:
    """Quản lý tất cả API calls với rate limiting và retry"""
    
    def __init__(self):
        self.api_keys = {
            'alpha_vantage': ALPHA_VANTAGE_API_KEY,
            'finnhub': FINNHUB_API_KEY,
            'marketaux': MARKETAUX_API_KEY,
            'newsapi': NEWSAPI_API_KEY,
            'eodhd': EODHD_API_KEY,
            'oanda': OANDA_API_KEY,
            'gemini': GEMINI_API_KEY
        }
        
        self.endpoints = {
            'oanda_base': OANDA_URL,
            'discord_webhook': DISCORD_WEBHOOK,
            'gemini_api': GEMINI_URL
        }
        
        self.rate_limits = {
            'alpha_vantage': {'requests_per_min': 5, 'last_request': 0},
            'finnhub': {'requests_per_min': 60, 'last_request': 0},
            'oanda': {'requests_per_min': 100, 'last_request': 0},
            'marketaux': {'requests_per_min': 100, 'last_request': 0},
            'newsapi': {'requests_per_min': 60, 'last_request': 0},
            'eodhd': {'requests_per_min': 100, 'last_request': 0},
            'gemini': {'requests_per_min': 60, 'last_request': 0}
        }
        
        self.session = None
        self.health_monitor = APIHealthMonitor()
        self.logger = log_manager.get_logger('BotCore')
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Kiểm tra rate limit"""
        now = time.time()
        limit_info = self.rate_limits[api_name]
        
        if now - limit_info['last_request'] < 60:  # Trong cùng 1 phút
            return False
        
        limit_info['last_request'] = now
        return True
    
    async def _make_request_with_retry(self, url: str, params: dict = None, 
                                     headers: dict = None, api_name: str = None,
                                     max_retries: int = 3) -> Optional[dict]:
        """Thực hiện request với retry và theo dõi health"""
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Kiểm tra rate limit
                if api_name and not self._check_rate_limit(api_name):
                    await asyncio.sleep(1)
                    continue
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        self.health_monitor.record_request(api_name or 'unknown', True, response_time)
                        return data
                    else:
                        self.logger.warning(f"API {api_name} trả về status {response.status}")
                        self.health_monitor.record_request(api_name or 'unknown', False, response_time)
                        
            except Exception as e:
                self.logger.error(f"Lỗi API {api_name}: {str(e)}")
                self.health_monitor.record_request(api_name or 'unknown', False, 0)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None

class OANDADataFetcher:
    """Chuyên dụng để lấy dữ liệu từ OANDA API"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.logger = log_manager.get_logger('DataManager')
        self.headers = {
            'Authorization': f'Bearer {OANDA_API_KEY}',
            'Content-Type': 'application/json'
        }
    
    async def get_account_info(self) -> Optional[dict]:
        """Lấy thông tin tài khoản"""
        url = f"{OANDA_URL}/accounts"
        return await self.api_manager._make_request_with_retry(
            url, headers=self.headers, api_name='oanda'
        )
    
    async def get_pricing(self, instruments: List[str]) -> Optional[dict]:
        """Lấy giá hiện tại cho các instruments"""
        instruments_str = ",".join(instruments)
        url = f"{OANDA_URL}/accounts/{self._get_account_id()}/pricing"
        params = {'instruments': instruments_str}
        
        return await self.api_manager._make_request_with_retry(
            url, params=params, headers=self.headers, api_name='oanda'
        )
    
    async def get_candlestick_data(self, instrument: str, granularity: str = 'M15',
                                 count: int = 500) -> Optional[dict]:
        """Lấy dữ liệu candlestick"""
        
        instrument_mapped = SYMBOL_MAPPING.get(instrument, instrument)
        url = f"{OANDA_URL}/instruments/{instrument_mapped}/candles"
        params = {
            'granularity': granularity,
            'count': count,
            'price': 'M'  # Mid prices
        }
        
        return await self.api_manager._make_request_with_retry(
            url, params=params, headers=self.headers, api_name='oanda'
        )
    
    def _get_account_id(self) -> str:
        """Lấy account ID (sẽ được implement theo tài khoản thực tế)"""
        return "101-001-"  # Placeholder

# ========================================
# PART 6: ENHANCED DATA MANAGER
# ========================================

class DataFreshnessManager:
    """Theo dõi độ mới của dữ liệu"""
    
    def __init__(self):
        self.data_timestamps = {}
        self.logger = log_manager.get_logger('DataManager')
    
    def update_timestamp(self, symbol_timeframe: str, timestamp: datetime.datetime):
        """Cập nhật timestamp của dữ liệu"""
        self.data_timestamps[symbol_timeframe] = timestamp
    
    def is_data_fresh(self, symbol_timeframe: str, freshness_threshold_minutes: int = 15) -> bool:
        """Kiểm tra dữ liệu có còn mới không"""
        if symbol_timeframe not in self.data_timestamps:
            return False
        
        last_update = self.data_timestamps[symbol_timeframe]
        threshold = datetime.timedelta(minutes=freshness_threshold_minutes)
        
        return datetime.datetime.now() - last_update < threshold
    
    def get_stale_symbols(self) -> List[str]:
        """Lấy danh sách symbols có dữ liệu cũ"""
        stale_symbols = []
        for symbol_tf, timestamp in self.data_timestamps.items():
            if not self.is_data_fresh(symbol_tf, 15):
                stale_symbols.append(symbol_tf)
        
        return stale_symbols

class EnhancedDataManager:
    """Quản lý dữ liệu nâng cao với đa khung thời gian"""
    
    def __init__(self):
        self.api_manager = None
        self.oanda_fetcher = None
        self.freshness_manager = DataFreshnessManager()
        self.data_cache = {}
        self.logger = log_manager.get_logger('DataManager')
    
    async def initialize(self):
        """Khởi tạo data manager"""
        self.api_manager = APIManager()
        self.oanda_fetcher = OANDADataFetcher(self.api_manager)
        
        async with self.api_manager:
            # Test connection
            account_info = await self.oanda_fetcher.get_account_info()
            if account_info:
                self.logger.info("🚀 Kết nối OANDA thành công!")
            else:
                self.logger.error("❌ Không thể kết nối OANDA API")
        
        self.logger.info("✅ Enhanced Data Manager khởi tạo thành công")
    
    async def fetch_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu đa khung thời gian"""
        
        timeframes_data = {}
        
        async with self.api_manager:
            self.oanda_fetcher = OANDADataFetcher(self.api_manager)
            
            for timeframe in TIMEFRAMES:
                try:
                    cache_key = f"{symbol}_{timeframe}"
                    
                    # Kiểm tra cache và độ mới của dữ liệu
                    if (cache_key in self.data_cache and 
                        self.freshness_manager.is_data_fresh(cache_key)):
                        
                        timeframes_data[timeframe] = self.data_cache[cache_key]
                        continue
                    
                    # Lấy dữ liệu mới
                    candle_data = await self.oanda_fetcher.get_candlestick_data(
                        symbol, timeframe, 500
                    )
                    
                    if candle_data and 'candles' in candle_data:
                        df = self._convert_candlestick_to_dataframe(candle_data['candles'])
                        
                        if not df.empty:
                            timeframes_data[timeframe] = df
                            self.data_cache[cache_key] = df
                            self.freshness_manager.update_timestamp(
                                cache_key, df.index[-1].to_pydatetime()
                            )
                            
                            self.logger.debug(f"✅ Lấy dữ liệu {symbol} {timeframe}: {len(df)} nến")
                        else:
                            self.logger.warning(f"⚠️ Dữ liệu {symbol} {timeframe} trống")
                    else:
                        self.logger.error(f"❌ Không thể lấy dữ liệu {symbol} {timeframe}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Lỗi lấy dữ liệu {symbol} {timeframe}: {str(e)}")
        
        return timeframes_data
    
    def _convert_candlestick_to_dataframe(self, candles: List[dict]) -> pd.DataFrame:
        """Chuyển đổi dữ liệu candlestick thành DataFrame"""
        
        if not candles:
            return pd.DataFrame()
        
        data = []
        for candle in candles:
            if candle['complete']:  # Chỉ lấy nến đã hoàn thành
                data.append({
                    'timestamp': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': float(candle['volume'])
                })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_real_time_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu real-time đã cache"""
        """Lấy dữ liệu real-time từ cache"""
        
        # Tìm dữ liệu M15 gần nhất
        cache_key = f"{symbol}_M15"
        if cache_key in self.data_cache:
            df = self.data_cache[cache_key]
            if not df.empty:
                return df.tail(1)  # Chỉ lấy bar cuối cùng
        
        return None
    
    def get_all_cached_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Lấy toàn bộ dữ liệu đã cache"""
        return dict(self.data_cache)
    
    def clear_cache(self, symbol: str = None):
        """Xóa cache"""
        if symbol:
            # Xóa cache của symbol cụ thể
            keys_to_remove = [key for key in self.data_cache.keys() if symbol in key]
            for key in keys_to_remove:
                del self.data_cache[key]
                self.freshness_manager.data_timestamps.pop(key, None)
        else:
            # Xóa toàn bộ cache
            self.data_cache.clear()
            self.freshness_manager.data_timestamps.clear()
        
        self.logger.info(f"🧹 Xóa cache {'symbol ' + symbol if symbol else 'toàn bộ'}")

# ========================================
# PART 7: ADVANCED FEATURE ENGINEER
# ========================================

class TechnicalIndicators:
    """Bộ công cụ tính toán các chỉ báo kỹ thuật"""
    
    @staticmethod
    def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo cơ bản"""
        
        result_df = df.copy()
        
        # Average True Range (ATR)
        result_df['atr_14'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        # ADX (Average Directional Movement Index)
        result_df['adx_14'] = ta.trend.ADXIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).adx()
        
        # Pivot Points
        result_df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        result_df['support1'] = 2 * result_df['pivot'] - df['high']
        result_df['resistance1'] = 2 * result_df['pivot'] - df['low']
        
        # RSI
        result_df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        result_df['macd'] = macd.macd()
        result_df['macd_signal'] = macd.macd_signal()
        result_df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)
        result_df['bb_upper'] = bollinger.bollinger_hband()
        result_df['bb_middle'] = bollinger.bollinger_mavg()
        result_df['bb_lower'] = bollinger.bollinger_lband()
        result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = stoch.stoch()
        result_df['stoch_d'] = stoch.stoch_signal()
        
        return result_df
    
    @staticmethod
    def calculate_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các đặc trưng thống kê"""
        
        result_df = df.copy()
        
        # Rolling returns
        result_df['returns'] = df['close'].pct_change()
        result_df['returns_5'] = df['close'].pct_change(5)
        result_df['returns_10'] = df['close'].pct_change(10)
        
        # Volatility measures
        result_df['volatility_10'] = result_df['returns'].rolling(10).std()
        result_df['volatility_20'] = result_df['returns'].rolling(20).std()
        
        # Skewness và Kurtosis
        result_df['skew_10'] = result_df['returns'].rolling(10).skew()
        result_df['kurtosis_10'] = result_df['returns'].rolling(10).kurt()
        
        # Price momentum
        result_df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        result_df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Price acceleration
        result_df['acceleration'] = result_df['returns'].diff()
        
        return result_df
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Phát hiện các mô hình nến Nhật"""
        
        result_df = df.copy()
        
        # Doji (giá mở và đóng gần như bằng nhau)
        doji_threshold = 0.001  # 0.1%
        result_df['doji'] = abs(df['open'] - df['close']) <= (df['high'] - df['low']) * doji_threshold
        
        # Hammer (nến búa)
        body = abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        
        result_df['hammer'] = (lower_shadow > 2 * body) & (upper_shadow < body * 0.5)
        
        # Shooting Star (sao băng)
        result_df['shooting_star'] = (upper_shadow > 2 * body) & (lower_shadow < body * 0.5)
        
        # Engulfing patterns
        result_df['bullish_engulfing'] = False
        result_df['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            prev_open = df.iloc[i-1]['open']
            prev_close = df.iloc[i-1]['close']
            curr_open = df.iloc[i]['open']
            curr_close = df.iloc[i]['close']
            
            # Bullish Engulfing
            if (prev_close < prev_open and  # Previous candle bearish
                curr_close > curr_open and   # Current candle bullish
                curr_close > prev_open and   # Bullish engulfing
                curr_open < prev_close):
                result_df.iloc[i, result_df.columns.get_loc('bullish_engulfing')] = True
            
            # Bearish Engulfing
            if (prev_close > prev_open and  # Previous candle bullish
                curr_close < curr_open and   # Current candle bearish
                curr_close < prev_open and   # Bearish engulfing
                curr_open > prev_close):
                result_df.iloc[i, result_df.columns.get_loc('bearish_engulfing')] = True
        
        return result_df

class WyckoffAnalyzer:
    """Phân tích Wyckoff cho Supply/Demand zones"""
    
    def __init__(cls, lookback_periods: int = 50):
        cls.lookback_periods = lookback_periods
    
    @staticmethod
    def identify_wyckoff_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Xác định các tín hiệu Wyckoff"""
        
        result_df = df.copy()
        
        # Spring signal (giá vướt qua support nhưng không giữ được)
        result_df['spring_detected'] = False
        result_df['spring_strength'] = 0.0
        
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            support_level = window['low'].min()
            current_low = df.iloc[i]['low']
            
            if current_low < support_level * 0.998:  # Dip below support
                # Check if it recovers quickly
                recovery_threshold = support_level * 1.005
                future_closes = df.iloc[i:i+5]['close'].values if i+5 < len(df) else df.iloc[i:]['close'].values
                
                if len(future_closes) > 0 and future_closes[-1] > recovery_threshold:
                    result_df.iloc[i, result_df.columns.get_loc('spring_detected')] = True
                    result_df.iloc[i, result_df.columns.get_loc('spring_strength')] = 1.0
        
        # Upthrust signal (giá vướt qua resistance nhưng không giữ được)
        result_df['upthrust_detected'] = False
        result_df['upthrust_strength'] = 0.0
        
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            resistance_level = window['high'].max()
            current_high = df.iloc[i]['high']
            
            if current_high > resistance_level * 1.002:  # Above resistance
                # Check if it falls back quickly
                rejection_threshold = resistance_level * 0.995
                future_closes = df.iloc[i:i+5]['close'].values if i+5 < len(df) else df.iloc[i:]['close'].values
                
                if len(future_closes) > 0 and future_closes[-1] < rejection_threshold:
                    result_df.iloc[i, result_df.columns.get_loc('upthrust_detected')] = True
                    result_df.iloc[i, result_df.columns.get_loc('upthrust_strength')] = 1.0
        
        # Accumulation/Distribution phases
        result_df['accumulation_score'] = 0.0
        result_df['distribution_score'] = 0.0
        
        # Simplified accumulation detection: low volatility + sideways movement
        for i in range(50, len(df)):
            window_volatility = df.iloc[i-50:i]['volatility_10'].mean()
            price_change_pct = abs((df.iloc[i]['close'] - df.iloc[i-50]['close']) / df.iloc[i-50]['close'])
            
            # Accumulation: low volatility, minimal price change
            if window_volatility < df.iloc[i]['volatility_10'] * 0.8 and price_change_pct < 0.02:
                result_df.iloc[i, result_df.columns.get_loc('accumulation_score')] = 1.0
            
            # Distribution: decreasing volume trends (simplified)
            recent_volume = df.iloc[i-10:i]['volume'].mean()
            older_volume = df.iloc[i-50:i-40]['volume'].mean()
            
            if recent_volume < older_volume * 0.8:
                result_df.iloc[i, result_df.columns.get_loc('distribution_score')] = 1.0
        
        return result_df

class SupplyDemandAnalyzer:
    """Phân tích Supply/Demand zones"""
    
    @staticmethod
    def find_supply_demand_zones(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Tìm các vùng Supply/Demand"""
        
        result_df = df.copy()
        
        # Demand zones (support levels)
        result_df['demand_zones'] = []
        result_df['supply_zones'] = []
        result_df['nearest_demand_distance'] = np.nan
        result_df['nearest_supply_distance'] = np.nan
        
        for i in range(window, len(df)):
            current_price = df.iloc[i]['close']
            
            # Find demand zones (local lows)
            lookback_window = df.iloc[i-window:i]
            lowest_price = lookback_window['low'].min()
            lowest_idx = lookback_window['low'].idxmin()
            
            # Check if this low is significant
            surrounding_window = df.iloc[max(0, i-window*2):min(len(df), i+window)]
            min_surrounding = surrounding_window['low'].min()
            
            if lowest_price == min_surrounding:
                # This is a significant low - potential demand zone
                result_df.iloc[i, result_df.columns.get_loc('nearest_demand_distance')] = abs(current_price - lowest_price) / current_price
            
            # Find supply zones (local highs)
            highest_price = lookback_window['high'].max()
            highest_idx = lookback_window['high'].idxmax()
            
            # Check if this high is significant
            max_surrounding = surrounding_window['high'].max()
            
            if highest_price == max_surrounding:
                # This is a significant high - potential supply zone
                result_df.iloc[i, result_df.columns.get_loc('nearest_supply_distance')] = abs(current_price - highest_price) / current_price
        
        return result_df

class RSIDivergenceDetector:
    """Phát hiện phân kỳ RSI"""
    
    @staticmethod
    def detect_rsi_divergence(df: pd.DataFrame, rsi_window: int = 14,
                            price_min_periods: int = 20) -> pd.DataFrame:
        """Phát hiện phân kỳ RSI"""
        
        result_df = df.copy()
        result_df['rsi_bullish_divergence'] = False
        result_df['rsi_bearish_divergence'] = False
        result_df['is_rsi_extreme'] = False  # RSI quá mua/quá bán
        
        for i in range(max(rsi_window, price_min_periods), len(df)):
            current_price = df.iloc[i]['close']
            current_rsi = df.iloc[i]['rsi_14']
            
            # Mark extreme RSI levels
            if current_rsi > 70:
                result_df.iloc[i, result_df.columns.get_loc('is_rsi_extreme')] = True
            elif current_rsi < 30:
                result_df.iloc[i, result_df.columns.get_loc('is_rsi_extreme')] = True
            
            # Look for divergences in the last 20 periods
            lookback_start = max(0, i - 20)
            price_window = df.iloc[lookback_start:i+1]
            rsi_window_data = df.iloc[lookback_start:i+1]
            
            if len(price_window) < 5:
                continue
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lows = []
            rsi_lows = []
            
            for j in range(len(price_window) - 1):
                if (price_window.iloc[j]['low'] < price_window.iloc[j-1]['low'] and 
                    price_window.iloc[j]['low'] < price_window.iloc[j+1]['low']):
                    price_lows.append(j)
                
                if (rsi_window_data.iloc[j]['rsi_14'] < rsi_window_data.iloc[j-1]['rsi_14'] and 
                    rsi_window_data.iloc[j]['rsi_14'] < rsi_window_data.iloc[j+1]['rsi_14']):
                    rsi_lows.append(j)
            
            # Check for bullish divergence
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                recent_price_low_idx = price_lows[-1]
                prev_price_low_idx = price_lows[-2]
                recent_price_low = price_window.iloc[recent_price_low_idx]['low']
                prev_price_low = price_window.iloc[prev_price_low_idx]['low']
                
                recent_rsi_low_idx = rsi_lows[-1]
                prev_rsi_low_idx = rsi_lows[-2]
                recent_rsi_low = rsi_window_data.iloc[recent_rsi_low_idx]['rsi_14']
                prev_rsi_low = rsi_window_data.iloc[prev_rsi_low_idx]['rsi_14']
                
                # Bullish divergence: price lower low, RSI higher low
                if (recent_price_low < prev_price_low and recent_rsi_low > prev_rsi_low):
                    result_df.iloc[i, result_df.columns.get_loc('rsi_bullish_divergence')] = True
            
            # Check for bearish divergence  
            price_highs = []
            rsi_highs = []
            
            for j in range(len(price_window) - 1):
                if (price_window.iloc[j]['high'] > price_window.iloc[j-1]['high'] and 
                    price_window.iloc[j]['high'] > price_window.iloc[j+1]['high']):
                    price_highs.append(j)
                
                if (rsi_window_data.iloc[j]['rsi_14'] > rsi_window_data.iloc[j-1]['rsi_14'] and 
                    rsi_window_data.iloc[j]['rsi_14'] > rsi_window_data.iloc[j+1]['rsi_14']):
                    rsi_highs.append(j)
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                recent_price_high_idx = price_highs[-1]
                prev_price_high_idx = price_highs[-2]
                recent_price_high = price_window.iloc[recent_price_high_idx]['high']
                prev_price_high = price_window.iloc[prev_price_high_idx]['high']
                
                recent_rsi_high_idx = rsi_highs[-1]
                prev_rsi_high_idx = rsi_highs[-2]
                recent_rsi_high = rsi_window_data.iloc[recent_rsi_high_idx]['rsi_14']
                prev_rsi_high = rsi_window_data.iloc[prev_rsi_high_idx]['rsi_14']
                
                # Bearish divergence: price higher high, RSI lower high
                if (recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high):
                    result_df.iloc[i, result_df.columns.get_loc('rsi_bearish_divergence')] = True
        
        return result_df

class MarketRegimeDetector:
    """Phát hiện chế độ thị trường (Trending/Sideways)"""
    
    @staticmethod
    def classify_market_regime(df: pd.DataFrame, volatility_window: int = 20,
                             trend_window: int = 50) -> pd.DataFrame:
        """Phân loại chế độ thị trường"""
        
        result_df = df.copy()
        result_df['market_regime'] = 'SIDEWAYS'
        result_df['trend_strength'] = 0.0
        result_df['volatility_regime'] = 'NORMAL'
        
        for i in range(max(volatility_window, trend_window), len(df)):
            # Calculate trend strength
            price_changes = df.iloc[i-trend_window:i+1]['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = df.iloc[i-volatility_window:i+1]['volatility_10'].mean()
            avg_volatility = df.iloc[i-volatility_window*2:i-volatility_window]['volatility_10'].mean()
            
            # Trend detection using linear regression slope
            y = df.iloc[i-trend_window:i+1]['close'].values
            x = range(len(y))
            
            if len(y) > 1:
                slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)
                trend_strength = abs(r_value[0]) * 100 if len(r_value) > 0 else 0
                
                # Trend direction
                upward_trend = sum(price_changes > 0) / len(price_changes) if len(price_changes) > 0 else 0.5
                
                result_df.iloc[i, result_df.columns.get_loc('trend_strength')] = trend_strength
                
                # Market regime classification
                if trend_strength > 0.6:  # Strong trend
                    if upward_trend > 0.55:
                        result_df.iloc[i, result_df.columns.get_loc('market_regime')] = 'TRENDING'
                    elif upward_trend < 0.45:
                        result_df.iloc[i, result_df.columns.get_loc('market_regime')] = 'TRENDING'
                else:
                    result_df.iloc[i, result_df.columns.get_loc('market_regime')] = 'SIDEWAYS'
            
            # Volatility regime
            if volatility > avg_volatility * 1.5:
                result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = 'HIGH'
            elif volatility < avg_volatility * 0.5:
                result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = 'LOW'
            else:
                result_df.iloc[i, result_df.columns.get_loc('volatility_regime')] = 'NORMAL'
        
        return result_df

class AdvancedFeatureEngineer:
    """Engineer tính toán các đặc trưng nâng cao"""
    
    def __init__(self):
        self.logger = log_manager.get_logger('FeatureEngineer')
        self.technical_indicators = TechnicalIndicators()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.supply_demand_analyzer = SupplyDemandAnalyzer()
        self.rsi_divergence_detector = RSIDivergenceDetector()
        self.market_regime_detector = MarketRegimeDetector()
    
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Tính toán toàn bộ các đặc trưng cho DataFrame"""
        
        try:
            if df.empty:
                self.logger.warning(f"⚠️ DataFrame trống cho {symbol}")
                return df
            
            self.logger.info(f"🔧 Bắt đầu engineering features cho {symbol}")
            
            # Copy để không modify original
            result_df = df.copy()
            
            # 1. Basic technical indicators
            result_df = self.technical_indicators.calculate_basic_indicators(result_df)
            
            # 2. Statistical features
            result_df = self.technical_indicators.calculate_statistical_features(result_df)
            
            # 3. Candlestick patterns
            result_df = self.technical_indicators.detect_candlestick_patterns(result_df)
            
            # 4. Wyckoff signals
            result_df = self.wyckoff_analyzer.identify_wyckoff_signals(result_df)
            
            # 5. Supply/Demand zones
            result_df = self.supply_demand_analyzer.find_supply_demand_zones(result_df)
            
            # 6. RSI divergences
            result_df = self.rsi_divergence_detector.detect_rsi_divergence(result_df)
            
            # 7. Market regime detection
            result_df = self.market_regime_detector.classify_market_regime(result_df)
            
            # 8. Additional custom features
            result_df = self._add_custom_features(result_df, symbol)
            
            self.logger.info(f"✅ Hoàn thành engineering {len(result_df.columns)} features cho {symbol}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi engineering features cho {symbol}: {str(e)}")
            return df
    
    def _add_custom_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Thêm các đặc trưng tùy chỉnh"""
        
        result_df = df.copy()
        
        # Price momentum acceleration
        result_df['price_acceleration'] = result_df['returns'].diff()
        
        # Volume-Price Trend
        if 'volume' in df.columns:
            result_df['volume_price_trend'] = result_df['volume'] * result_df['returns']
            
            # Volume spikes
            volume_ma = result_df['volume'].rolling(20).mean()
            result_df['volume_spike'] = result_df['volume'] > volume_ma * 2
        
        # Support/Resistance strength
        result_df['support_resistance_strength'] = 0.0
        
        for i in range(20, len(result_df)):
            # Check how many times price touched near current levels
            current_close = result_df.iloc[i]['close']
            lookback_window = result_df.iloc[i-100:i]
            
            tolerance = current_close * 0.005  # 0.5% tolerance
            
            # Count touches
            touches = abs(current_close - lookback_window['close']).le(tolerance).sum()
            result_df.iloc[i, result_df.columns.get_loc('support_resistance_strength')] = touches / 100
        
        # Market session features (simplified)
        result_df['hour'] = result_df.index.hour
        result_df['is_asia_session'] = ((result_df['hour'] >= 2) & (result_df['hour'] <= 8)).astype(int)
        result_df['is_london_session'] = ((result_df['hour'] >= 8) & (result_df['hour'] <= 16)).astype(int)
        result_df['is_ny_session'] = ((result_df['hour'] >= 13) & (result_df['hour'] <= 21)).astype(int)
        
        # Fibonacci levels (simplified)
        lookback_window = 50
        result_df['fib_382'] = np.nan
        result_df['fib_618'] = np.nan
        
        for i in range(lookback_window, len(result_df)):
            window_data = result_df.iloc[i-lookback_window:i]
            high_price = window_data['high'].max()
            low_price = window_data['low'].min()
            
            price_range = high_price - low_price
            fib_382_level = low_price + price_range * 0.382
            fib_618_level = low_price + price_range * 0.618
            
            current_close = result_df.iloc[i]['close']
            
            distance_to_fib382 = abs(current_close - fib_382_level) / current_close
            distance_to_fib618 = abs(current_close - fib_618_level) / current_close
            
            result_df.iloc[i, result_df.columns.get_loc('fib_382')] = distance_to_fib382
            result_df.iloc[i, result_df.columns.get_loc('fib_618')] = distance_to_fib618
        
        return result_df
    
    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Trả về mức độ quan trọng của các features (placeholder)"""
        # Đây sẽ được implement sau khi có trained models
        return {
            'rsi_14': 0.15,
            'atr_14': 0.12,
            'macd': 0.10,
            'trend_strength': 0.08,
            'market_regime': 0.07,
            'volatility_20': 0.06,
            'bb_width': 0.05,
            'momentum_5': 0.04,
            'support_resistance_strength': 0.03,
            'volume_spike': 0.02
        }
    
    def get_latest_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Lấy đặc trưng mới nhất từ DataFrame"""
        if df.empty:
            return {}
        
        latest_row = df.iloc[-1]
        features = latest_row.to_dict()
        
        # Convert boolean features to float
        for key, value in features.items():
            if isinstance(value, bool):
                features[key] = float(value)
            elif pd.isna(value):
                features[key] = 0.0
        
        return features

# ========================================
# PART 8: NEWS & ECONOMIC DATA MANAGER
# ========================================

class GeminiSentimentAnalyzer:
    """Phân tích cảm tính tin tức bằng Gemini 1.5 Flash"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.logger = log_manager.get_logger('NewsManager')
        self.session = None
    
    async def analyze_sentiment(self, title: str, content: str, symbol: str) -> float:
        """Phân tích cảm tính của tin tức bằng Gemini"""
        
        try:
            prompt = f"""
            Phân tích cảm tính của tin tức tài chính sau đây cho cặp tiền tệ/{symbol} và trả về điểm số từ -1 (rất tiêu cực) đến +1 (rất tích cực):
            
            Tiêu đề: {title}
            Nội dung: {content[:500]}  # Giới hạn nội dung để tiết kiệm token
            
            Xem xét:
            1. Tác động trực tiếp lên giá của {symbol}
            2. Cảm tính tổng thể của thị trường
            3. Xu hướng tăng/giảm giá tiềm năng
            
            Trả về chỉ điểm số số thực từ -1.0 đến 1.0, không có giải thích thêm.
            """
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': GEMINI_API_KEY
            }
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            async with self.session.post(
                GEMINI_URL, 
                json=payload, 
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if 'candidates' in data and len(data['candidates']) > 0:
                        response_text = data['candidates'][0]['content']['parts'][0]['text']
                        
                        # Extract sentiment score
                        import re
                        sentiment_match = re.search(r'-?\d+\.?\d*', response_text.strip())
                        
                        if sentiment_match:
                            sentiment_score = float(sentiment_match.group())
                            # Ensure score is in range [-1, 1]
                            sentiment_score = max(-1.0, min(1.0, sentiment_score))
                            return sentiment_score
                        
            # Fallback sentiment analysis (simple keyword based)
            return self._simple_sentiment_analysis(title + " " + content, symbol)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi phân tích sentiment: {str(e)}")
            return 0.0  # Neutral sentiment on error
    
    def _simple_sentiment_analysis(self, text: str, symbol: str) -> float:
        """Phân tích cảm tính đơn giản dựa trên từ khóa"""
        
        positive_words = [
            'tăng', 'tích cực', 'phát triển', 'thịnh vượng', 'mạnh mẽ',
            'tốt', 'cao', 'xuất sắc', 'thành công', 'lãi', 'profit',
            'bullish', 'uptrend', 'breakout', 'rally'
        ]
        
        negative_words = [
            'giảm', 'tiêu cực', 'suy thoái', 'khủng hoảng', 'yếu',
            'xấu', 'thấp', 'thất bại', 'lỗ', 'loss', 'bearish',
            'downtrend', 'crash', 'correction', 'decline'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_ratio = (positive_count - negative_count) / total_words
        return sentiment_ratio

class FinancialCalendarReader:
    """Đọc lịch kinh tế từ các nguồn công khai"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.logger = log_manager.get_logger('NewsManager')
    
    async def get_economic_events(self, symbols: List[str], days_ahead: int = 7) -> List[dict]:
        """Lấy các sự kiện kinh tế quan trọng"""
        
        try:
            events = []
            
            # API Free Economics Calendar (ForexFactory-style)
            url = "https://api.marketaux.com/v1/news/economy"
            params = {
                'api_token': MARKETAUX_API_KEY,
                'days_ahead': days_ahead,
                'limit': 50
            }
            
            async with self.api_manager.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data:
                        for event in data['data']:
                            # Filter events that might affect our symbols
                            if self._is_relevant_event(event, symbols):
                                events.append({
                                    'title': event.get('title', ''),
                                    'description': event.get('description', ''),
                                    'date': event.get('published_at', ''),
                                    'importance': event.get('severity', 'medium'),
                                    'currency': self._extract_currency(event.get('title', '')),
                                    'impact_score': self._calculate_impact_score(event)
                                })
            
            self.logger.info(f"📅 Lấy được {len(events)} sự kiện kinh tế")
            return events
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy lịch kinh tế: {str(e)}")
            return []
    
    def _is_relevant_event(self, event: dict, symbols: List[str]) -> bool:
        """Kiểm tra xem sự kiện có liên quan đến symbols không"""
        
        title = event.get('title', '').lower()
        
        # Keywords for major currencies/pairs
        currency_keywords = {
            'EURUSD': ['eur', 'euro', 'european', 'german', 'french', 'italian'],
            'GBPUSD': ['gbp', 'pound', 'british', 'uk', 'england'],
            'USDJPY': ['jpy', 'yen', 'japan', 'japanese', 'bank of japan'],
            'XAUUSD': ['gold', 'bullion', 'precious', 'metal'],
            'BTCUSD': ['bitcoin', 'crypto', 'cryptocurrency', 'digital currency', 'btc'],
            'NAS100': ['nasdaq', 'tech', 'technology', 'stock', 'market', 'dow', 'sp500']
        }
        
        for symbol in symbols:
            if symbol in currency_keywords:
                for keyword in currency_keywords[symbol]:
                    if keyword in title:
                        return True
        
        # Major economic indicators
        economic_keywords = [
            'interest rate', 'federal reserve', 'fed', 'inflation', 'cpi', 'pp the',
            'employment', 'nfp', 'gdp', 'manufacturing', 'retail sales',
            'consumer confidence', 'trade balance', 'current account'
        ]
        
        for keyword in economic_keywords:
            if keyword in title:
                return True
        
        return False
    
    def _extract_currency(self, title: str) -> str:
        """Trích xuất currency code từ title"""
        
        title_upper = title.upper()
        
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        
        for currency in currencies:
            if currency in title_upper:
                return currency
        
        return 'USD'  # Default
    
    def _calculate_impact_score(self, event: dict) -> float:
        """Tính điểm tác động của sự kiện"""
        
        importance_map = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        return importance_map.get(event.get('severity', 'medium'), 0.6)

class MultiSourceNewsAggregator:
    """Tổng hợp tin tức từ nhiều nguồn"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.gemini_analyzer = GeminiSentimentAnalyzer(api_manager)
        self.calendar_reader = FinancialCalendarReader(api_manager)
        self.logger = log_manager.get_logger('NewsManager')
        self.news_cache = {}
        self.last_update = {}
    
    async def initialize_gemini_session(self):
        """Khởi tạo session cho Gemini"""
        self.gemini_analyzer.session = aiohttp.ClientSession()
    
    async def close_gemini_session(self):
        """Đóng session Gemini"""
        if self.gemini_analyzer.session:
            await self.gemini_analyzer.session.close()
    
    async def fetch_all_news(self, symbols: List[str]) -> Dict[str, List[NewsData]]:
        """Lấy tin tức từ tất cả các nguồn"""
        
        symbol_news = {symbol: [] for symbol in symbols}
        
        try:
            await self.initialize_gemini_session()
            
            # Fetch news from multiple sources concurrently
            tasks = []
            
            for symbol in symbols:
                tasks.append(self._fetch_symbol_news(symbol))
            
            # Wait for all news fetching to complete
            news_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(news_results):
                symbol = symbols[i]
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Lỗi lấy tin tức cho {symbol}: {str(result)}")
                else:
                    symbol_news[symbol] = result
            
            await self.close_gemini_session()
            
            total_news = sum(len(news_list) for news_list in symbol_news.values())
            self.logger.info(f"📰 Tổng hợp được {total_news} tin tức từ {len(symbols)} symbols")
            
            return symbol_news
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tổng hợp tin tức: {str(e)}")
            await self.close_gemini_session()
            return symbol_news
    
    async def _fetch_symbol_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin tức cho một symbol cụ thể"""
        
        news_list = []
        
        try:
            # Get from multiple news APIs
            sources = [
                ('finnhub', self._fetch_finnhub_news),
                ('marketaux', self._fetch_marketaux_news),
                ('newsapi', self._fetch_newsapi_news),
                ('eodhd', self._fetch_eodhd_news)
            ]
            
            for source_name, fetch_func in sources:
                try:
                    source_news = await fetch_func(symbol)
                    news_list.extend(source_news)
                except Exception as e:
                    self.logger.warning(f"⚠️ Lỗi lấy tin từ {source_name} cho {symbol}: {str(e)}")
            
            # Remove duplicates based on title similarity
            news_list = self._remove_duplicate_news(news_list)
            
            # Cache the results
            cache_key = f"news_{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H')}"
            self.news_cache[cache_key] = news_list
            self.last_update[symbol] = datetime.datetime.now()
            
            self.logger.info(f"✅ Lấy được {len(news_list)} tin tức cho {symbol}")
            
            return news_list
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy tin cho {symbol}: {str(e)}")
            return []
    
    async def _fetch_finnhub_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin từ Finnhub API"""
        
        news_list = []
        
        url = "https://finnhub.io/api/v1/news"
        params = {
            'category': 'general',
            'token': FINNHUB_API_KEY
        }
        
        async with self.api_manager.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                for item in data[:10]:  # Limit to 10 articles
                    title = item.get('headline', '')
                    content = item.get('summary', '')
                    
                    # Check if relevant to our symbol
                    if self._is_relevant_to_symbol(title + " " + content, symbol):
                        # Analyze sentiment
                        sentiment_score = await self.gemini_analyzer.analyze_sentiment(title, content, symbol)
                        
                        news_data = NewsData(
                            title=title,
                            content=content,
                            source='Finnhub',
                            timestamp=datetime.datetime.fromtimestamp(item.get('datetime', 0)),
                            sentiment_score=sentiment_score,
                            impact_level='medium',
                            symbols_mentioned=[symbol]
                        )
                        
                        news_list.append(news_data)
        
        return news_list
    
    async def _fetch_marketaux_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin từ Marketaux API"""
        
        news_list = []
        
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'symbols': symbol,
            'language': 'en',
            'api_token': MARKETAUX_API_KEY,
            'limit': 10
        }
        
        async with self.api_manager.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if 'data' in data:
                    for item in data['data']:
                        title = item.get('title', '')
                        content = item.get('description', '')
                        
                        # Analyze sentiment
                        sentiment_score = await self.gemini_analyzer.analyze_sentiment(title, content, symbol)
                        
                        news_data = NewsData(
                            title=title,
                            content=content,
                            source='Marketaux',
                            timestamp=datetime.datetime.fromisoformat(item.get('published_at', '').replace('Z', '+00:00')),
                            sentiment_score=sentiment_score,
                            impact_level='medium',
                            symbols_mentioned=[symbol]
                        )
                        
                        news_list.append(news_data)
        
        return news_list
    
    async def _fetch_newsapi_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin từ NewsAPI"""
        
        news_list = []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"forex trading" OR "{symbol}"',
            'sources': 'financial-news,business-insider,reuters',
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_API_KEY,
            'pageSize': 10
        }
        
        async with self.api_manager.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if 'articles' in data:
                    for item in data['articles']:
                        title = item.get('title', '')
                        content = item.get('description', '')
                        
                        if self._is_relevant_to_symbol(title + " " + content, symbol):
                            # Analyze sentiment
                            sentiment_score = await self.gemini_analyzer.analyze_sentiment(title, content, symbol)
                            
                            news_data = NewsData(
                                title=title,
                                content=content,
                                source='NewsAPI',
                                timestamp=datetime.datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                                sentiment_score=sentiment_score,
                                impact_level='medium',
                                symbols_mentioned=[symbol]
                            )
                            
                            news_list.append(news_data)
        
        return news_list
    
    async def _fetch_eodhd_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin từ EODHD API"""
        
        news_list = []
        
        url = "https://eodhd.com/api/eod/news"
        params = {
            's': symbol,
            'api_token': EODHD_API_KEY,
            'limit': 10
        }
        
        async with self.api_manager.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                for item in data[:10]:
                    title = item.get('title', '')
                    content = item.get('content', '')
                    
                    if self._is_relevant_to_symbol(title + " " + content, symbol):
                        # Analyze sentiment
                        sentiment_score = await self.gemini_analyzer.analyze_sentiment(title, content, symbol)
                        
                        news_data = NewsData(
                            title=title,
                            content=content,
                            source='EODHD',
                            timestamp=datetime.datetime.fromtimestamp(item.get('date', 0)),
                            sentiment_score=sentiment_score,
                            impact_level='medium',
                            symbols_mentioned=[symbol]
                        )
                        
                        news_list.append(news_data)
        
        return news_list
    
    def _is_relevant_to_symbol(self, text: str, symbol: str) -> bool:
        """Kiểm tra tin tức có liên quan đến symbol không"""
        
        text_lower = text.lower()
        
        # Symbol keywords mapping
        symbol_keywords = {
            'EURUSD': ['eur', 'euro', 'dollar', 'fx', 'forex', 'currency'],
            'GBPUSD': ['gbp', 'pound', 'sterling', 'dollar', 'fx', 'forex'],
            'USDJPY': ['jpy', 'yen', 'japan', 'dollar', 'fx', 'forex'],
            'XAUUSD': ['gold', 'bullion', 'precious metal', 'xau', 'commodity'],
            'BTCUSD': ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'digital'],
            'NAS100': ['nasdaq', 'nasdaq100', 'stock', 'equity', 'market']
        }
        
        if symbol in symbol_keywords:
            keywords = symbol_keywords[symbol]
            return any(keyword in text_lower for keyword in keywords)
        
        return False
    
    def _remove_duplicate_news(self, news_list: List[NewsData]) -> List[NewsData]:
        """Loại bỏ tin tức trùng lặp"""
        
        unique_news = []
        seen_titles = set()
        
        for news in news_list:
            # Simple deduplication based on title similarity
            title_key = news.title.lower()
            
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_similarity(title_key, seen_title) > 0.8:
                    indicates_duplicate = True
                    break
            
            if not is_duplicate:
                unique_news.append(news)
                seen_titles.add(title_key)
        
        return unique_news
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Tính độ tương tự giữa hai văn bản (simplified)"""
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_cached_news(self, symbol: str) -> List[NewsData]:
        """Lấy tin tức đã cache"""
        
        cache_key = f"news_{symbol}_{datetime.datetime.now().strftime('%Y%m%d_%H')}"
        return self.news_cache.get(cache_key, [])
    
    def get_news_features(self, news_list: List[NewsData]) -> Dict[str, float]:
        """Trích xuất đặc trưng từ danh sách tin tức"""
        
        if not news_list:
            return {
                'news_sentiment_score': 0.0,
                'news_count': 0.0,
                'positive_news_ratio': 0.0,
                'negative_news_ratio': 0.0,
                'avg_news_impact': 0.0
            }
        
        # Calculate sentiment features
        sentiments = [news.sentiment_score for news in news_list]
        
        avg_sentiment = np.mean(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        
        impact_scores = [self._impact_level_to_score(news.impact_level) for news in news_list]
        avg_impact = np.mean(impact_scores)
        
        return {
            'news_sentiment_score': avg_sentiment,
            'news_count': len(news_list),
            'positive_news_ratio': positive_count / len(news_list),
            'negative_news_ratio': negative_count / len(news_list),
            'avg_news_impact': avg_impact
        }
    
    def _impact_level_to_score(self, impact_level: str) -> float:
        """Chuyển đổi impact level thành điểm số"""
        
        impact_map = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        return impact_map.get(impact_level, 0.6)

class NewsEconomicManager:
    """Quản lý tổng thể tin tức và lịch kinh tế"""
    
    def __init__(self):
        self.api_manager = None
        self.news_aggregator = None
        self.logger = log_manager.get_logger('NewsManager')
        self.last_update = {}
    
    async def initialize(self):
        """Khởi tạo News Manager"""
        
        self.api_manager = APIManager()
        self.news_aggregator = MultiSourceNewsAggregator(self.api_manager)
        
        async with self.api_manager:
            self.logger.info("✅ News Economic Manager khởi tạo thành công")
    
    async def get_enriched_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Lấy dữ liệu tin tức được làm giàu cho các symbols"""
        
        try:
            enriched_data = {}
            
            # Fetch news for all symbols
            news_data = await self.news_aggregator.fetch_all_news(symbols)
            
            # Fetch economic events
            economic_events = await self.news_aggregator.calendar_reader.get_economic_events(symbols)
            
            for symbol in symbols:
                symbol_news = news_data.get(symbol, [])
                news_features = self.news_aggregator.get_news_features(symbol_news)
                
                # Add economic calendar features
                econ_features = self._calculate_economic_features(economic_events, symbol)
                
                # Combine all features
                enriched_data[symbol] = {**news_features, **econ_features}
                
                self.last_update[symbol] = datetime.datetime.now()
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy dữ liệu tin tức: {str(e)}")
            return {symbol: {} for symbol in symbols}
    
    def _calculate_economic_features(self, events: List[dict], symbol: str) -> Dict[str, float]:
        """Tính toán đặc trưng từ lịch kinh tế"""
        
        symbol_events = []
        
        for event in events:
            if symbol in event.get('symbols_mentioned', []):
                symbol_events.append(event)
        
        if not symbol_events:
            return {
                'upcoming_high_impact_events': 0.0,
                'next_24h_event_count': 0.0,
                'economic_pressure_index': 0.0
            }
        
        now = datetime.datetime.now()
        
        # Count high impact events in next 7 days
        high_impact_count = sum(1 for event in symbol_events 
                              if event.get('impact_score', 0) > 0.8 and 
                              datetime.datetime.fromisoformat(event.get('date', '').replace('Z', '+00:00')) > now)
        
        # Count events in next 24 hours
        next_24h = now + datetime.timedelta(hours=24)
        events_24h = sum(1 for event in symbol_events 
                        if datetime.datetime.fromisoformat(event.get('date', '').replace('Z', '+00:00')) <= next_24h)
        
        # Economic pressure index (weighted sum of impacts)
        pressure_index = sum(event.get('impact_score', 0) for event in symbol_events 
                           if datetime.datetime.fromisoformat(event.get('date', '').replace('Z', '+00:00')) <= next_24h)
        
        return {
            'upcoming_high_impact_events': high_impact_count,
            'next_24h_event_count': events_24h,
            'economic_pressure_index': pressure_index
        }

# ========================================
# PART 9: MACHINE LEARNING SYSTEM
# ========================================

class PurgedGroupTimeSeriesSplit:
    """Time Series Cross Validation với purge period"""
    
    def __init__(self, n_splits=5, n_groups=None, t1=None):
        self.n_splits = n_splits
        self.n_groups = n_groups or max(100, n_splits)
        self.t1 = t1  # Purge threshold
    
    def split(self, X, y=None, groups=None):
        """Split data với purge period"""
        
        indices = np.arange(len(X))
        groups = groups if groups is not None else np.arange(len(X))
        
        unique_groups = np.unique(groups)
        
        if len(unique_groups) < self.n_splits:
            raise ValueError(f"Nhóm quá ít ({len(unique_groups)}) cho số fold ({self.n_splits})")
        
        group_indices = np.array_split(unique_groups, self.n_splits)
        
        for i in range(self.n_splits):
            test_groups = group_indices[i]
            
            # Trước test set
            train_groups = np.concatenate(group_indices[:i])
            
            # Sau test set  
            if i < self.n_splits - 1:
                train_groups = np.concatenate([train_groups, group_indices[i+1:]])
            
            # Purge period - loại bỏ các mẫu gần test set
            if self.t1 is not None:
                purge_threshold = np.percentile(group_indices[i], (100 - self.t1 * 100)) if len(group_indices[i]) > 1 else 0
                
                train_mask = groups <= purge_threshold
                train_groups = unique_groups[train_mask]
            
            train_indices = indices[np.isin(groups, train_groups)]
            test_indices = indices[np.isin(groups, test_groups)]
            
            yield train_indices, test_indices

class ConceptDriftDetector:
    """Phát hiện concept drift trong dữ liệu"""
    
    def __init__(self, drift_threshold: float = 0.1, window_size: int = 100):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.logger = log_manager.get_logger('EnsembleModel')
        self.baseline_distributions = {}
    
    def detect_drift(self, current_data: np.ndarray, symbol: str) -> bool:
        """Kiểm tra có drift hay không"""
        
        try:
            # Lấy phân bố baseline (nếu có)
            if symbol not in self.baseline_distributions:
                # Tính baseline từ data hiện tại
                self.baseline_distributions[symbol] = self._calculate_distribution(current_data)
                return False
            
            # Tính phân bố hiện tại
            current_dist = self._calculate_distribution(current_data)
            baseline_dist = self.baseline_distributions[symbol]
            
            # Tính khoảng cách phân bố (KL Divergence)
            kl_divergence = self._kl_divergence(current_dist, baseline_dist)
            
            drift_detected = kl_divergence > self.drift_threshold
            
            if drift_detected:
                self.logger.warning(f"🚨 Concept drift phát hiện cho {symbol}: KL={kl_divergence:.4f}")
                # Update baseline
                self.baseline_distributions[symbol] = current_dist
            
            return drift_detected
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi phát hiện drift cho {symbol}: {str(e)}")
            return False
    
    def _calculate_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Tính phân bố của data"""
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skew': np.mean((data - np.mean(data))**3) / (np.std(data)**3) if np.std(data) > 0 else 0,
            'kurtosis': np.mean((data - np.mean(data))**4) / (np.std(data)**4) if np.std(data) > 0 else 0
        }
    
    def _kl_divergence(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """Tính KL divergence giữa hai phân bố"""
        
        # Simplified KL divergence cho phân bố chuẩn giả
        score = 0.0
        
        # Mean difference
        mean_diff = abs(dist1['mean'] - dist2['mean']) / max(abs(dist2['mean']), 1e-8)
        score += mean_diff
        
        # Std difference  
        std_diff = abs(dist1['std'] - dist2['std']) / max(dist2['std'], 1e-8)
        score += std_diff
        
        return score / 2

class OnlineLearningManager:
    """Quản lý học online với River"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = log_manager.get_logger('EnsembleModel')
        
        # Online learning models
        self.online_scale = preprocessing.StandardScaler()
        self.online_model = ensemble.AdaptiveRandomForestClassifier(
            n_models=50,
            model_selector=lambda x, y: x,
            drift_detector=ensemble.ADWIN(),
            warning_detector=ensemble.ADWIN()
        )
        
        self.model_performance = []
        
    def update_model(self, features: np.ndarray, target: int, timestamp: datetime.datetime):
        """Cập nhật model với dữ liệu mới"""
        
        try:
            # Scale features
            scaled_features = self.online_scale.learn_one(features).transform_one(features)
            
            # Update model
            prediction = self.online_model.predict_one(scaled_features)
            self.online_model.learn_one(scaled_features, target)
            
            # Track performance
            accuracy = prediction == target if prediction is not None else 0
            self.model_performance.append({
                'timestamp': timestamp,
                'accuracy': accuracy,
                'prediction': prediction
            })
            
            # Keep only recent performance
            if len(self.model_performance) > 100:
                self.model_performance = self.model_performance[-100:]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi update online model cho {self.symbol}: {str(e)}")
            return None
    
    def get_model_stats(self) -> Dict[str, float]:
        """Lấy thống kê model"""
        
        if not self.model_performance:
            return {'accuracy': 0.0, 'recent_accuracy': 0.0, 'updates_count': 0}
        
        recent_performance = self.model_performance[-10:] if len(self.model_performance) >= 10 else self.model_performance
        
        accuracy = np.mean([p['accuracy'] for p in self.model_performance])
        recent_accuracy = np.mean([p['accuracy'] for p in recent_performance])
        
        return {
            'accuracy': accuracy,
            'recent_accuracy': recent_accuracy,
            'updates_count': len(self.model_performance)
        }

class UltraOverfittingPrevention:
    """Hệ thống chống overfitting mạnh mẽ"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = log_manager.get_logger('EnsembleModel')
        
    def apply_regularization_techniques(self, models: Dict, training_data: pd.DataFrame) -> Dict:
        """Áp dụng các kỹ thuật regularization"""
        
        try:
            regularized_models = {}
            
            for model_name, model in models.items():
                regularized_models[model_name] = self._regularize_model(model, model_name, training_data)
            
            self.logger.info(f"✅ Áp dụng regularization cho {len(regularized_models)} models {self.symbol}")
            
            return regularized_models
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi regularization: {str(e)}")
            return models
    
    def _regularize_model(self, model, model_name: str, training_data: pd.DataFrame):
        """Regularize một model cụ thể"""
        
        if 'XGB' in model_name.upper():
            # XGBoost regularization
            model.set_params(
                reg_alpha=0.1,
                reg_lambda=0.1,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8
            )
        
        elif 'LGBM' in model_name.upper() or 'LIGHTGBM' in model_name.upper():
            # LightGBM regularization
            model.set_params(
                reg_alpha=0.1,
                reg_lambda=0.1,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                feature_fraction=0.8
            )
        
        elif 'RF' in model_name.upper() or 'RANDOM' in model_name.upper():
            # Random Forest regularization
            model.set_params(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt'
            )
        
        elif hasattr(model, 'C'):
            # LogReg/SVM regularization
            model.set_params(C=0.1)
        
        # Apply dropout cho Neural Networks
        elif hasattr(model, 'dropout'):
            model.dropout = 0.3
        
        return model
    
    def cross_validate_with_purged_splits(self, model, X: pd.DataFrame, y: pd.Series,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Cross validation với purge period"""
        
        try:
            # Create time-based groups
            groups = np.arange(len(X)) // 100  # Group every 100 samples
            
            cv_splitter = PurgedGroupTimeSeriesSplit(n_splits=5, t1=0.01)
            
            scores = []
            fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
            
            for train_idx, val_idx in cv_splitter.split(X.values):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Preprocess
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                fold_scores['accuracy'].append(accuracy)
                fold_scores['precision'].append(precision)
                fold_scores['recall'].append(recall)
                fold_scores['f1'].append(f1)
                
                scores.append(accuracy)
            
            # Calculate final metrics
            final_scores = {
                'cv_accuracy': np.mean(fold_scores['accuracy']),
                'cv_std': np.std(scores),
                'cv_precision': np.mean(fold_scores['precision']),
                'cv_recall': np.mean(fold_scores['recall']),
                'cv_f1': np.mean(fold_scores['f1']),
                'overfitting_score': max(0, np.mean(scores) - np.min(scores))
            }
            
            return final_scores
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi cross validation: {str(e)}")
            return {'cv_accuracy': 0.0, 'cv_std': 1.0}

class LSTMModel:
    """Mạng LSTM nâng cao với Attention và Dropout"""
    
    def __init__(self, symbol: str, sequence_length: int = 60, n_features: int = 30):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.logger = log_manager.get_logger('EnsembleModel')
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self) -> tf.keras.Model:
        """Xây dựng kiến trúc LSTM với Attention"""
        
        try:
            # Input layer
            inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
            
            # First LSTM layer với Dropout
            lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
            lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
            
            # Second LSTM layer
            lstm2 = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)(lstm1)
            lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
            
            # Attention mechanism (simplified)
            attention_weights = tf.keras.layers.Dense(1, activation='softmax')(lstm2)
            attention_output = tf.keras.layers.Multiply()([lstm2, attention_weights])
            
            # Global Average Pooling thay vì flatten
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
            
            # Dense layers với dropout
            dense1 = tf.keras.layers.Dense(64, activation='relu')(pooled)
            dense1 = tf.keras.layers.Dropout(0.3)(dense1)
            dense1 = tf.keras.layers.BatchNormalization()(dense1)
        
            dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
            dense2 = tf.keras.layers.Dropout(0.3)(dense2)
            dense2 = tf.keras.layers.BatchNormalization()(dense2)
            
            # Output layer
            outputs = tf.keras.layers.Dense(3, activation='softmax')(dense2)  # BUY, SELL, HOLD
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile với optimizer nâng cao
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.logger.info(f"✅ Xây dựng LSTM model cho {self.symbol} thành công")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi xây dựng LSTM model: {str(e)}")
            return None
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị dữ liệu dạng sequence cho LSTM"""
        
        try:
            # Select features
            feature_cols = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            if len(feature_cols) > self.n_features:
                # Select top features nếu có quá nhiều
                feature_cols = feature_cols[:self.n_features]
            
            X = df[feature_cols].fillna(0).values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(self.sequence_length, len(df)):
                X_sequences.append(X_scaled[i-self.sequence_length:i])
                target = df.iloc[i][target_col]
                y_sequences.append(target)
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Convert targets to categorical
            y_categorical = tf.keras.utils.to_categorical(y_sequences, num_classes=3)
            
            self.logger.info(f"✅ Chuẩn bị {len(X_sequences)} sequences cho LSTM")
            
            return X_sequences, y_categorical
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi chuẩn bị sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, float]:
        """Huấn luyện LSTM model"""
        
        try:
            if self.model is None:
                self.build_model()
            
            # Prepare sequences
            X, y = self.prepare_sequences(df)
            
            if len(X) == 0:
                return {'accuracy': 0.0, 'loss': 1.0}
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/lstm_{self.symbol}.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            
            results = {
                'accuracy': val_accuracy,
                'loss': val_loss,
                'epochs_trained': len(history.history['loss'])
            }
            
            self.logger.info(f"✅ LSTM training hoàn thành cho {self.symbol}: Acc={val_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi training LSTM: {str(e)}")
            return {'accuracy': 0.0, 'loss': 1.0}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Dự đoán với LSTM model"""
        
        try:
            if self.model is None:
                return np.zeros(3)
            
            X, _ = self.prepare_sequences(df)
            
            if len(X) == 0:
                return np.zeros(3)
            
            # Predict
            predictions = self.model.predict(X[-1:], verbose=0)
            
            return predictions[0]
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi predict LSTM: {str(e)}")
            return np.zeros(3)

class EnsembleModel:
    """Hệ thống Ensemble model với XGBoost, LightGBM, Random Forest"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = log_manager.get_logger('EnsembleModel')
        
        # Base models
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        # Meta model (Stacking)
        self.meta_model = CalibratedClassifierCV(
            LogisticRegression(random_state=42),
            method='isotonic'
        )
        
        # Optuna study for hyperparameter optimization
        self.study = None
        
        # Prevention systems
        self.drift_detector = ConceptDriftDetector()
        self.online_learning = OnlineLearningManager(symbol)
        self.overfitting_prevention = UltraOverfittingPrevention(symbol)
        
        # Training stats
        self.training_stats = {}
        self.feature_importance = {}
        
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                trials: int = 50) -> Dict[str, Any]:
        """Tối ưu hóa siêu tham số với Optuna"""
        
        def objective(trial):
            model_name = trial.suggest_categorical('model_type', ['XGBoost', 'LightGBM'])
            
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
                }
                model = xgb.XGBClassifier(random_state=42, **params)
            
            else:  # LightGBM
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0)
                }
                model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
            
            # Cross validation với PurgedGroupTimeSeriesSplit
            cv_splitter = PurgedGroupTimeSeriesSplit(n_splits=3)
            groups = np.arange(len(X)) // 100
            
            scores = []
            for train_idx, val_idx in cv_splitter.split(X.values):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            return np.mean(scores)
        
        try:
            # Configure Optuna study
            self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            
            # Run optimization
            self.study.optimize(objective, n_trials=trials, show_progress_bar=False)
            
            # Get best parameters
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            self.logger.info(f"✅ Optuna optimization hoàn thành cho {self.symbol}: Score={best_score:.3f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': self.study
            }
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi Optuna optimization: {str(e)}")
            return {'best_params': {}, 'best_score': 0.0}
    
    def train_ensemble(self, df: pd.DataFrame, target_col: str = 'target',
                      optimize_hp: bool = True) -> Dict[str, Any]:
        """Huấn luyện toàn bộ ensemble"""
        
        try:
            # Prepare data
            feature_cols = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            X = df[feature_cols].fillna(0)
            y = df[target_col].astype(int)
            
            self.logger.info(f"🔧 Bắt đầu training ensemble cho {self.symbol}: {len(X)} mẫu")
            
            # Detect concept drift
            drift_detected = self.drift_detector.detect_drift(X.values, self.symbol)
            
            if drift_detected:
                self.logger.warning(f"🚨 Concept drift phát hiện cho {self.symbol}, áp dụng điều chỉnh")
            
            # Optimize hyperparameters if requested
            if optimize_hp and not self.study:
                self.optimize_hyperparameters(X, y, trials=30)
            
            # Apply regularization
            regularized_models = self.overfitting_prevention.apply_regularization_techniques(
                self.models, df
            )
            
            # Cross validation scores
            cv_results = {}
            
            for model_name, model in regularized_models.items():
                cv_score = self.overfitting_prevention.cross_validate_with_purged_splits(
                    model, X, y, feature_cols
                )
                cv_results[model_name] = cv_score
            
            # Train with best parameters from Optuna
            if self.study and self.study.best_params:
                best_model_type = self.study.best_params.get('model_type', 'XGBoost')
                if best_model_type in regularized_models:
                    best_params = {k: v for k, v in self.study.best_params.items() 
                                 if k != 'model_type'}
                    regularized_models[best_model_type].set_params(**best_params)
            
            # Train all models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            trained_models = {}
            
            for model_name, model in regularized_models.items():
                try:
                    model.fit(X_scaled, y)
                    trained_models[model_name] = model
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = dict(zip(feature_cols, model.feature_importances_))
                    
                    self.logger.info(f"✅ Training {model_name} hoàn thành")
                    
                except Exception as e:
                    self.logger.error(f"❌ Lỗi training {model_name}: {str(e)}")
            
            # Train meta model (stacking)
            if len(trained_models) > 1:
                try:
                    # Create meta features từ base models
                    meta_features = np.zeros((len(X), len(trained_models) * 3))  # 3 classes
                    
                    for i, (model_name, model) in enumerate(trained_models.items()):
                        if hasattr(model, 'predict_proba'):
                            probas = model.predict_proba(X_scaled)
                        else:
                            # Convert predictions to probabilities
                            preds = model.predict(X_scaled)
                            probas = np.zeros((len(X), 3))
                            for j in range(len(X)):
                                probas[j, preds[j]] = 1.0
                        
                        meta_features[:, i*3:(i+1)*3] = probas
                    
                    # Train meta model
                    self.meta_model.fit(meta_features, y)
                    
                    self.logger.info("✅ Meta model training hoàn thành")
                    
                except Exception as e:
                    self.logger.error(f"❌ Lỗi training meta model: {str(e)}")
            
            # Store training stats
            self.training_stats = {
                'training_samples': len(X),
                'features_count': len(feature_cols),
                'cv_results': cv_results,
                'models_trained': list(trained_models.keys()),
                'drift_detected': drift_detected,
                'optimization_score': self.study.best_value if self.study else None
            }
            
            # Update online learning
            online_stats = self.online_learning.get_model_stats()
            self.training_stats['online_performance'] = online_stats
            
            # Save models
            self.models = trained_models
            self.scaler = scaler
            
            self.logger.info(f"🎯 Ensemble training hoàn thành cho {self.symbol}")
            
            return self.training_stats
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi training ensemble: {str(e)}")
            return {}
    
    def predict_ensemble(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict[str, Any]]:
        """Dự đoán với ensemble"""
        
        try:
            if not self.models:
                return SignalType.HOLD, 0.0, {}
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            X = df[feature_cols].fillna(0).iloc[-1:].values
            
            if len(X) == 0:
                return SignalType.HOLD, 0.0, {}
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from base models
            base_predictions = {}
            base_probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_scaled)[0]
                        prediction = np.argmax(probabilities)
                    else:
                        prediction = model.predict(X_scaled)[0]
                        probabilities = np.zeros(3)
                        probabilities[prediction] = 1.0
                    
                    base_predictions[model_name] = prediction
                    base_probabilities[model_name] = probabilities
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Lỗi predict {model_name}: {str(e)}")
                    base_predictions[model_name] = 0  # HOLD
                    base_probabilities[model_name] = np.array([0.33, 0.33, 0.34])
            
            # Meta model prediction
            final_prediction = SignalType.HOLD
            final_confidence = 0.5
            
            if self.meta_model and len(base_probabilities) > 1:
                # Create meta features
                meta_features = np.concatenate([
                    np.zeros((1, len(self.models) * 3))
                ])
                
                for i, (model_name, probas) in enumerate(base_probabilities.items()):
                    meta_features[0, i*3:(i+1)*3] = probas
                
                # Meta prediction
                meta_probas = self.meta_model.predict_proba(meta_features)[0]
                final_prediction_idx = np.argmax(meta_probas)
                final_confidence = np.max(meta_probas)
                
                # Convert to SignalType
                signal_mapping = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
                final_prediction = signal_mapping[final_prediction_idx]
            
            else:
                # Simple voting nếu không có meta model
                predictions_list = list(base_predictions.values())
                votes = np.bincount(predictions_list, minlength=3)
                final_prediction_idx = np.argmax(votes)
                final_confidence = votes[final_prediction_idx] / sum(votes)
                
                signal_mapping = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
                final_prediction = signal_mapping[final_prediction_idx]
            
            # Prediction details
            prediction_details = {
                'base_predictions': base_predictions,
                'base_probabilities': base_probabilities,
                'final_confidence': final_confidence,
                'feature_importance': self.feature_importance,
                'training_stats': self.training_stats
            }
            
            return final_prediction, final_confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi predict ensemble: {str(e)}")
            return SignalType.HOLD, 0.0, {}
    
    def update_with_new_data(self, df: pd.DataFrame, target_col: str = 'target'):
        """Cập nhật model với dữ liệu mới (online learning)"""
        
        try:
            if df.empty or len(df) < 2:
                return
            
            # Get latest features and target
            latest_row = df.iloc[-1]
            feature_cols = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            features = latest_row[feature_cols].fillna(0).values
            target = int(latest_row[target_col]) if target_col in latest_row else 1  # Default to HOLD
            
            # Update online learning
            prediction = self.online_learning.update_model(features, target, datetime.datetime.now())
            
            # Periodic retraining if needed
            if len(self.online_learning.model_performance) % 100 == 0:
                self.logger.info(f"🔄 Tiến hành retrain ensemble cho {self.symbol}")
                self.train_ensemble(df.tail(1000), target_col)  # Use last 1000 samples
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi update với data mới: {str(e)}")

# ========================================
# PART 10: REINFORCEMENT LEARNING SYSTEM
# ========================================

class PortfolioState:
    """Class đại diện cho trạng thái portfolio"""
    
    def __init__(self, symbols: List[str], cash: float = 10000.0):
        self.symbols = symbols
        self.cash = cash
        self.positions = {symbol: 0.0 for symbol in symbols}  # Number of shares/units
        self.price_history = {symbol: [] for symbol in symbols}
        self.portfolio_values = []
        self.current_prices = {symbol: 0.0 for symbol in symbols}
        
        # Performance metrics
        self.sharpe_ratio = 0.0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        
        self.timestamp = datetime.datetime.now()
    
    def get_portfolio_value(self) -> float:
        """Compute total portfolio value"""
        positions_value = sum(self.positions[symbol] * self.current_prices[symbol] 
                            for symbol in self.symbols)
        return self.cash + positions_value
    
    def get_position_weights(self) -> Dict[str, float]:
        """Calculate position weights"""
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {symbol: 0.0 for symbol in self.symbols}
        
        return {
            symbol: (self.positions[symbol] * self.current_prices[symbol]) / total_value
            for symbol in self.symbols
        }
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all symbols"""
        self.current_prices.update(prices)
    
    def add_price_history(self, symbol: str, price: float):
        """Add price to history"""
        if symbol in self.price_history:
            self.price_history[symbol].append(price)
            
        # Keep only recent history for memory efficiency
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        self.sharpe_ratio = excess_return / np.std(returns) * np.sqrt(252)  # Annualized
        
        return self.sharpe_ratio
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        self.max_drawdown = max_dd
        return max_dd
    
    def update_performance_metrics(self):
        """Update all performance metrics"""
        current_value = self.get_portfolio_value()
        if not self.portfolio_values:
            initial_value = self.get_portfolio_value()
        else:
            initial_value = self.portfolio_values[0]
        
        self.portfolio_values.append(current_value)
        
        # Keep only recent values
        if len(self.portfolio_values) > 252:  # Keep last year
            self.portfolio_values = self.portfolio_values[:1] + self.portfolio_values[-252:]
        
        self.total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0.0
        self.calculate_sharpe_ratio()
        self.calculate_max_drawdown()
        
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            self.volatility = np.std(returns) * np.sqrt(252)  # Annualized

class PortfolioEnvironment(gym.Env):
    """RL Environment cho Portfolio Management"""
    
    def __init__(self, symbols: List[str], initial_cash: float = 10000.0,
                 transaction_cost: float = 0.001, max_position_size: float = 0.3):
        
        super(PortfolioEnvironment, self).__init__()
        
        self.symbols = symbols
        self.n_symbols = len(symbols)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        self.logger = log_manager.get_logger('RLAgent')
        
        # Action space: [buy_symbol1, sell_symbol1, hold_symbol1, ..., buy_symbolN, sell_symbolN, hold_symbolN]
        # HAY có thể là weights cho từng symbol: [weight_symbol1, weight_symbol2, ..., weight_symbolN]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_symbols,), dtype=np.float32
        )
        
        # State space: [price_features, portfolio_features, market_state] cho mỗi symbol
        # Giả sử mỗi symbol có 10 features, plus portfolio state (cash, positions, returns, etc.)
        state_size = self.n_symbols * 10 + 10  # Portfolio state features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Initialize state
        self.current_state = PortfolioState(symbols, initial_cash)
        self.data_manager = None
        self.feature_engineer = None
        
        # Reward tracking
        self.rewards_history = []
        self.episode_count = 0
        
        # Performance tracking
        self.best_sharpe = float('-inf')
        self.performance_history = []
    
    def set_data_providers(self, data_manager, feature_engineer):
        """Set access to data manager and feature engineer"""
        self.data_manager = data_manager
        self.feature_engineer = feature_engineer
    
    def reset(self):
        """Reset environment"""
        
        self.current_state = PortfolioState(self.symbols, self.initial_cash)
        self.rewards_history = []
        
        # Get initial market data
        observation = self._get_initial_observation()
        
        self.logger.info(f"🔄 Environment reset cho {len(self.symbols)} symbols")
        
        return observation
    
    def _get_initial_observation(self) -> np.ndarray:
        """Get initial observation from current market data"""
        
        try:
            observation = []
            
            # Get latest data for each symbol
            for symbol in self.symbols:
                if self.data_manager:
                    # Get latest features (simplified)
                    df_data = self.data_manager.get_real_time_data(symbol)
                    if df_data is not None and not df_data.empty:
                        features = self.feature_engineer.get_latest_features(df_data)
                    else:
                        features = {key: 0.0 for key in ['rsi_14', 'atr_14', 'macd', 'trend_strength', 
                                                          'volatility_20', 'momentum_5', 'bb_width', 
                                                          'market_regime', 'volume_spike', 'returns']}
                else:
                    # Default features if no data manager
                    features = {key: 0.0 for key in ['rsi_14', 'atr_14', 'macd', 'trend_strength', 
                                                      'volatility_20', 'momentum_5', 'bb_width', 
                                                      'market_regime', 'volume_spike', 'returns'])
                
                # Add symbol features to observation
                symbol_features = [features.get(key, 0.0) for key in 
                                  ['rsi_14', 'atr_14', 'macd', 'trend_strength', 'volatility_20', 
                                   'momentum_5', 'bb_width', 'market_regime', 'volume_spike', 'returns']]
                observation.extend(symbol_features)
            
            # Add portfolio state features
            portfolio_features = [
                self.current_state.cash / self.initial_cash,  # Normalized cash
                self.current_state.get_portfolio_value() / self.initial_cash,  # Portfolio value
                len(self.current_state.portfolio_values),  # Time steps
                self.current_state.total_return,
                self.current_state.sharpe_ratio,
                self.current_state.max_drawdown,
                self.current_state.volatility,
                sum(self.current_state.positions.values()),  # Total positions
                np.std(list(self.current_state.get_position_weights().values())),  # Position diversity
                len(self.current_state.portfolio_values) / 252  # Portfolio age in years
            ]
            
            observation.extend(portfolio_features)
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo observation: {str(e)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state"""
        
        try:
            # Parse action (weights for each symbol)
            action_weights = action.copy()
            
            # Normalize weights to sum to 1 (softmax-like)
            action_weights = np.exp(action_weights) / np.sum(np.exp(action_weights))
            
            # Clip weights to reasonable range
            action_weights = np.clip(action_weights, 0, self.max_position_size)
            action_weights = action_weights / np.sum(action_weights) * 0.95  # Reserve 5% cash
            
            # Execute rebalancing
            reward, transaction_cost = self._rebalance_portfolio(action_weights)
            
            # Get new observation
            observation = self._get_new_observation()
            
            # Update state
            self.current_state.update_performance_metrics()
            
            # Check termination conditions
            terminated = self._check_termination()
            
            info = {
                'portfolio_value': self.current_state.get_portfolio_value(),
                'transaction_cost': transaction_cost,
                'sharpe_ratio': self.current_state.sharpe_ratio,
                'max_drawdown': self.current_state.max_drawdown,
                'total_return': self.current_state.total_return
            }
            
            self.rewards_history.append(reward)
            
            return observation, reward, terminated, info
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi execute action: {str(e)}")
            return self._get_new_observation(), -1.0, True, {}
    
    def _rebalance_portfolio(self, target_weights: np.ndarray) -> Tuple[float, float]:
        """Rebalance portfolio according to target weights"""
        
        total_cost = 0.0
        
        try:
            current_weights = np.array(list(self.current_state.get_position_weights().values()))
            
            # Calculate required trades
            weight_diff = target_weights - current_weights
            
            # Execute trades
            for i, symbol in enumerate(self.symbols):
                if abs(weight_diff[i]) > 0.01:  # Minimum rebalancing threshold
                    portfolio_value = self.current_state.get_portfolio_value()
                    
                    # Calculate trade amount
                    target_value = target_weights[i] * portfolio_value
                    current_value = self.current_state.positions[symbol] * self.current_state.current_prices[symbol]
                    trade_value = target_value - current_value
                    
                    if abs(trade_value) > 10:  # Minimum trade size
                        # Execute trade
                        if trade_value > 0:  # Buy
                            shares_to_buy = trade_value / self.current_state.current_prices[symbol]
                            cost = trade_value * (1 + self.transaction_cost)
                            
                            if cost <= self.current_state.cash:
                                self.current_state.positions[symbol] += shares_to_buy
                                self.current_state.cash -= cost
                                total_cost += cost * self.transaction_cost
                        
                        else:  # Sell
                            shares_to_sell = abs(trade_value) / self.current_state.current_prices[symbol]
                            proceeds = trade_value * (1 - self.transaction_cost)
                            
                            if shares_to_sell <= self.current_state.positions[symbol]:
                                self.current_state.positions[symbol] -= abs(shares_to_sell)
                                self.current_state.cash += abs(proceeds)
                                total_cost += abs(proceeds * self.transaction_cost)
        
        except Exception as e:
            self.logger.error(f"❌ Lỗi rebalancing: {str(e)}")
        
        # Calculate reward
        reward = self._calculate_reward(total_cost)
        
        return reward, total_cost
    
    def _calculate_reward(self, transaction_cost: float) -> float:
        """Calculate reward for current state"""
        
        try:
            # Primary reward: Sharpe ratio
            sharpe_reward = self.current_state.sharpe_ratio * 0.5
            
            # Penalty for high drawdown
            drawdown_penalty = min(0, self.current_state.max_drawdown * 10)
            
            # Penalty for excessive transactions
            transaction_penalty = -transaction_cost / self.initial_cash * 100
            
            # Diversification reward
            weights = np.array(list(self.current_state.get_position_weights().values()))
            diversification_reward = -np.std(weights) * 2  # Penalty for concentrated positions
            
            # Portfolio growth reward
            portfolio_growth = self.current_state.total_return * 0.3
            
            # Combine rewards
            total_reward = (sharpe_reward + drawdown_penalty + transaction_penalty + 
                          diversification_reward + portfolio_growth)
            
            # Clip reward to reasonable range
            total_reward = np.clip(total_reward, -10, 10)
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tính reward: {str(e)}")
            return -1.0
    
    def _get_new_observation(self) -> np.ndarray:
        """Get updated observation"""
        return self._get_initial_observation()  # Simplified for now
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Terminate if drawdown too high
        if self.current_state.max_drawdown > 0.15:  # 15% max drawdown
            return True
        
        # Terminate if portfolio value drops too much
        if self.current_state.get_portfolio_value() < self.initial_cash * 0.8:  # 20% loss
            return True
        
        return False

class RLAgent:
    """RL Agent sử dụng PPO từ stable-baselines3"""
    
    def __init__(self, symbols: List[str], initial_cash: float = 10000.0):
        self.symbols = symbols
        self.logger = log_manager.get_logger('RLAgent')
        
        # Create environment
        self.env = PortfolioEnvironment(symbols, initial_cash)
        self.env = make_vec_env(lambda: self.env, n_envs=4)  # Vectorized environment
        
        # PP Agent với custom settings
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            tensorboard_log="./logs/rl_tensorboard/"
        )
        
        # Training statistics
        self.training_stats = {
            'episodes_trained': 0,
            'best_reward': float('-inf'),
            'last_mean_reward': 0.0,
            'training_time': 0.0
        }
        
        # Callbacks
        self.callbacks = []
        
    def set_data_providers(self, data_manager, feature_engineer):
        """Provide access to data managers"""
        self.env.envs[0].env.set_data_providers(data_manager, feature_engineer)
        for env in self.env.envs[1:]:
            env.env.set_data_providers(data_manager, feature_engineer)
    
    def add_custom_callbacks(self):
        """Add custom callbacks for training"""
        
        # Early stopping callback
        early_stopping = StopTrainingOnRewardThreshold(reward_threshold=5.0, verbose=1)
        
        # Evaluation callback
        eval_env = PortfolioEnvironment(self.symbols)
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path="./models/",
            log_path="./logs/rl_evaluation/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        self.callbacks = [early_stopping, eval_callback]
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train RL agent"""
        
        try:
            self.logger.info(f"🚀 Bắt đầu training RL Agent cho {len(self.symbols)} symbols")
            
            start_time = time.time()
            
            # Set callbacks
            self.add_custom_callbacks()
            
            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                tb_log_name="ppo_portfolio_trading"
            )
            
            training_time = time.time() - start_time
            
            # Update training stats
            self.training_stats.update({
                'episodes_trained': self.training_stats['episodes_trained'] + 1,
                'training_time': training_time,
                'total_timesteps': total_timesteps
            })
            
            # Save trained model
            model_path = f"models/rl_agent_{'_'.join(self.symbols)}_ppo"
            self.model.save(model_path)
            
            self.logger.info(f"✅ RL Agent training hoàn thành: {total_timesteps} timesteps trong {training_time:.2f}s")
            
            return self.training_stats
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi training RL Agent: {str(e)}")
            return {}
    
    def predict_action(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict action from observation"""
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action, np.array([0.8])  # Confidence placeholder
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi predict action: {str(e)}")
            return np.zeros(self.env.observation_space.shape), np.array([0.0])
    
    def evaluate_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        
        try:
            total_rewards = []
            episode_lengths = []
            sharpe_ratios = []
            max_drawdowns = []
            
            for episode in range(n_episodes):
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    action, _ = self.predict_action(obs)
                    obs, reward, terminated, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated:
                        break
                
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                sharpe_ratios.append(info.get('sharpe_ratio', 0.0))
                max_drawdowns.append(info.get('max_drawdown', 0.0))
            
            performance_metrics = {
                'mean_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'mean_episode_length': np.mean(episode_lengths),
                'mean_sharpe_ratio': np.mean(sharpe_ratios),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'win_rate': sum(r > 0 for r in total_rewards) / len(total_rewards)
            }
            
            self.logger.info(f"📊 RL Performance Evaluation: Sharpe={performance_metrics['mean_sharpe_ratio']:.3f}")
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi evaluate performance: {str(e)}")
            return {}

class AutoRetrainManager:
    """Quản lý tự động retrain định kỳ"""
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        self.logger = log_manager.get_logger('EnsembleModel')
        
        # Retrain schedules
        self.last_training = {}
        self.performance_history = {}
        
        # Automatic retraining triggers
        self.retrain_thresholds = {
            'accuracy_drop': 0.05,
            'drift_score': 0.1,
            'days_since_training': 7,
            'performance_degradation': 0.1
        }
        
    def should_retrain_model(self, symbol: str, model_type: str = 'ensemble') -> bool:
        """Kiểm tra có nên retrain model hay không"""
        
        try:
            now = datetime.datetime.now()
            last_train = self.last_training.get(f"{symbol}_{model_type}")
            
            # Check time-based retrain
            if last_train is None:
                return True
            
            days_since = (now - last_train).days
            if days_since >= self.retrain_thresholds['days_since_training']:
                self.logger.info(f"⏰ Retrain {symbol} {model_type}: {days_since} ngày đã qua")
                return True
            
            # Check performance degradation
            performance_key = f"{symbol}_{model_type}"
            if performance_key in self.performance_history:
                recent_performance = self.performance_history[performance_key][-10:]  # Last 10 evaluations
                if len(recent_performance) >= 5:
                    current_avg = np.mean(recent_performance[-5:])
                    historical_avg = np.mean(recent_performance[:-5])
                    
                    performance_drop = historical_avg - current_avg
                    if performance_drop > self.retrain_thresholds['performance_degradation']:
                        self.logger.warning(f"📉 Retrain {symbol} {model_type}: Performance giảm {"%.2f" % performance_drop}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi check retrain: {str(e)}")
            return False
    
    def schedule_retraining(self, symbol: str, model_type: str, priority: str = 'normal'):
        """Lên lịch retraining"""
        
        self.logger.info(f"📅 Scheduled {priority} retrain cho {symbol} {model_type}")
        
        # Store schedule for main training loop
        retrain_key = f"{symbol}_{model_type}_{priority}"
        
        # This would be picked up by the main training loop
        return retrain_key
    
    def track_performance(self, symbol: str, model_type: str, performance_score: float):
        """Track model performance over time"""
        
        try:
            performance_key = f"{symbol}_{model_type}"
            
            if performance_key not in self.performance_history:
                self.performance_history[performance_key] = []
            
            self.performance_history[performance_key].append(performance_score)
            
            # Keep only recent history
            if len(self.performance_history[performance_key]) > 100:
                self.performance_history[performance_key] = self.performance_history[performance_key][-100:]
            
            self.logger.debug(f"📈 Tracked performance {symbol} {model_type}: {performance_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi track performance: {str(e)}")

# ========================================
# MAIN EXECUTION
# ========================================

class TradingBot:
    """Bot giao dịch chính"""
    
    def __init__(self):
        self.logger = log_manager.get_logger('BotCore')
        self.is_running = True
    
    async def initialize(self):
        """Khởi tạo bot"""
        try:
            self.data_manager = EnhancedDataManager()
            await self.data_manager.initialize()
            
            self.feature_engineer = AdvancedFeatureEngineer()
            self.ensemble_models = {symbol: EnsembleModel(symbol) for symbol in SYMBOLS}
            
            self.logger.info("✅ Trading Bot khởi tạo thành công")
            return True
        except Exception as e:
            self.logger.error(f"❌ Lỗi khởi tạo: {str(e)}")
            return False
    
    async def run(self):
        """Chạy bot chính"""
        try:
            if not await self.initialize():
                return
            
            self.logger.info("🚀 Bot giao dịch đang chạy...")
            
            while self.is_running:
                await self.trading_cycle()
                await asyncio.sleep(3600)  # Sleep 1 hour
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Bot dừng")
        except Exception as e:
            self.logger.error(f"❌ Lỗi: {str(e)}")
    
    async def trading_cycle(self):
        """Một chu kỳ giao dịch"""
        try:
            for symbol in SYMBOLS:
                # Get data
                df_data = await self.data_manager.fetch_multiple_timeframes(symbol)
                
                if 'M15' in df_data and not df_data['M15'].empty:
                    # Engineer features
                    featured_data = self.feature_engineer.engineer_features(df_data['M15'], symbol)
                    
                    # Generate signal
                    signal, confidence, details = self.ensemble_models[symbol].predict_ensemble(featured_data)
                    
                    if confidence > 0.7:
                        self.logger.info(f"📈 {symbol}: {signal.value} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi trading cycle: {str(e)}")

async def main():
    """Hàm main"""
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🚀 AI TRADING BOT v1.0 🚀                    ║
    ║                                                              ║
    ║  Features:                                                   ║  
    ║  ✅ Advanced Machine Learning Ensemble (XGBoost, LightGBM)   ║
    ║  ✅ LSTM Neural Networks with Attention                      ║
    ║  ✅ Reinforcement Learning with PPO                          ║
    ║  ✅ Multi-source News Analysis with AI Sentiment              ║
    ║  ✅ Advanced Technical Indicators & Pattern Recognition      ║
    ║  ✅ Real-time Risk Management                                ║
    ║  ✅ Discord Integration                                      ║
    ║                                                              ║
    ║  Symbols: XAUUSD, EURUSD, NAS100, BTCUSD                     ║
    ║  Author: AI Trading System                                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        traceback.print_exc()