"""
Enhanced Data Management System with Multi-timeframe Support and Freshness Monitoring
Advanced data collection, validation, and feature engineering for trading models
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

from logger import data_logger, get_logger
from config import TIMEFRAMES, ASSET_TYPES, MARKET_HOURS
from api_manager import api_manager

class DataFreshnessManager:
    """Monitor and ensure data freshness across different assets"""
    
    def __init__(self):
        self.freshness_thresholds = {
            "forex": {"minutes": 15, "weekend_relax": True},
            "crypto": {"minutes": 5, "weekend_relax": True},
            "commodity": {"minutes": 15, "weekend_relax": False},
            "index": {"minutes": 30, "weekend_relax": False}
        }
        self.logger = get_logger("DataFreshnessManager")
    
    def check_freshness(self, symbol: str, last_update: datetime, timeframe: str) -> Tuple[bool, str]:
        """Check if data is fresh enough for trading"""
        asset_type = ASSET_TYPES.get(symbol, "forex")
        threshold_config = self.freshness_thresholds[asset_type]
        
        # Calculate time since last update
        time_since_update = datetime.now() - last_update
        
        # Apply weekend relaxation for applicable assets
        if threshold_config["weekend_relax"] and self._is_weekend():
            threshold_minutes = threshold_config["minutes"] * 2  # Double threshold on weekends
        else:
            threshold_minutes = threshold_config["minutes"]
        
        # Adjust threshold based on timeframe
        timeframe_multiplier = {
            "M15": 1,
            "H1": 2,
            "H4": 4,
            "D1": 8
        }.get(timeframe, 1)
        
        adjusted_threshold = threshold_minutes * timeframe_multiplier
        
        if time_since_update.total_seconds() > adjusted_threshold * 60:
            message = f"Data stale: {time_since_update.total_seconds()/60:.1f}min > {adjusted_threshold}min threshold"
            return False, message
        
        return True, f"Data fresh: {time_since_update.total_seconds()/60:.1f}min old"
    
    def _is_weekend(self) -> bool:
        """Check if current time is weekend"""
        now = datetime.now()
        return now.weekday() >= 5
    
    def get_refresh_priority(self, symbol: str, timeframes: List[str]) -> List[str]:
        """Get priority order for data refresh based on asset type and timeframes"""
        asset_type = ASSET_TYPES.get(symbol, "forex")
        
        # Priority: shorter timeframes first for most asset types
        priority_order = ["M15", "H1", "H4", "D1", "W1"]
        
        # Adjust for index/assets that update less frequently
        if asset_type in ["index"]:
            priority_order = ["H1", "H4", "D1", "M15", "W1"]
        
        # Filter to only available timeframes
        return [tf for tf in priority_order if tf in timeframes]

class EnhancedDataManager:
    """Advanced data management with multi-source integration and caching"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.freshness_manager = DataFreshnessManager()
        self.logger = data_logger
        self.cache_db = "data_cache.db"
        
        # Initialize databases
        self._init_databases()
        
        # Data storage
        self.price_data = {}
        self.news_data = {}
        self.economic_data = {}
        
    def _init_databases(self):
        """Initialize SQLite databases for data storage"""
        # Main trading database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                content TEXT,
                url TEXT,
                published_at DATETIME,
                source TEXT,
                sentiment_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                value REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.close()
        
        # Cache database
        cache_conn = sqlite3.connect(self.cache_db)
        cache_conn.execute("""
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cache_conn.close()
    
    async def get_price_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Get OHLCV price data with freshness check and auto-refresh"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            df = pd.read_json(cached_data, orient='records')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check freshness
            if not df.empty:
                last_update = df['timestamp'].max().to_pydatetime()
                is_fresh, message = self.freshness_manager.check_freshness(symbol, last_update, timeframe)
                
                if is_fresh:
                    self.logger.debug(f"✅ Using cached data for {symbol}_{timeframe}: {message}")
                    return df.set_index('timestamp')
                else:
                    self.logger.warning(f"⚠️ Cached data stale for {symbol}_{timeframe}: {message}")
        
        # Fetch fresh data
        df = await self._fetch_price_data(symbol, timeframe, limit)
        
        if not df.empty:
            # Save to cache
            self._save_to_cache(cache_key, df.to_json(orient='records'))
            
            # Save to database
            self._save_price_to_db(df, symbol, timeframe)
            
            return df.set_index('timestamp')
        
        return pd.DataFrame()
    
    async def _fetch_price_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch fresh price data from APIs"""
        try:
            async with api_manager as am:
                if symbol == "BTCUSD":
                    # Use different API for crypto
                    data = await am.get_price_data(symbol, "5min", "alpha_vantage")
                    df = self._parse_alpha_vantage_data(data, timeframe)
                else:
                    data = await am.get_price_data(symbol, timeframe.lower(), "alpha_vantage")
                    df = self._parse_alpha_vantage_data(data, timeframe)
                
                self.logger.info(f"📊 Fetched {len(df)} candles for {symbol}_{timeframe}")
                return df.tail(limit)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch price data for {symbol}_{timeframe}: {e}")
            return pd.DataFrame()
    
    def _parse_alpha_vantage_data(self, data: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Parse Alpha Vantage API response"""
        try:
            time_series_key = None
            
            # Find the correct time series key
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                self.logger.error(f"No time series data found in API response")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Parse data
            df_data = []
            for timestamp_str, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(float(values['5. volume']))
                })
            
            df = pd DataFrame(df_data)
            return df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    async def get_news_data(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """Get comprehensive news data"""
        cache_key = f"news_{symbol}_{days_back}"
        
        # Check cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            df = pd.read_json(cached_data, orient='records')
            df['published_at'] = pd.to_datetime(df['published_at'])
            self.logger.debug(f"✅ Using cached news data for {symbol}")
            return df
        
        # Fetch fresh news
        try:
            async with api_manager as am:
                news_list = await am.get_news_data(symbol, days_back)
                
                # Process and deduplicate news
                processed_news = self._process_news_data(news_list)
                df = pd.DataFrame(processed_news)
                
                if not df.empty:
                    self._save_to_cache(cache_key, df.to_json(orient='records'))
                    self._save_news_to_db(df, symbol)
                
                self.logger.info(f"📰 Fetched {len(df)} news articles for {symbol}")
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch news data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_news_data(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and deduplicate news data"""
        processed = []
        seen_titles = set()
        
        for news in news_list:
            title = news.get('title', '')
            
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            processed_news = {
                'title': title,
                'description': news.get('description', ''),
                'content': news.get('content', ''),
                'url': news.get('url', ''),
                'published_at': news.get('publishedAt', news.get('published_at')),
                'source': news.get('source', {}).get('name', 'unknown') if isinstance(news.get('source'), dict) else str(news.get('source', '')),
                'sentiment_score': 0.0  # Will be filled by sentiment analysis
            }
            
            processed.append(processed_news)
        
        return processed
    
    async def get_economic_calendar(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get economic calendar data from free sources"""
        try:
            # Use FRED API for economic data
            econ_data = []
            
            # Get major economic indicators for days ahead
            today = datetime.now()
            for i in range(days_ahead):
                date = today + timedelta(days=i)
                
                # Add placeholder economic events
                # In a real implementation, you would query FRED or other free sources
                econ_data.append({
                    'date': date,
                    'time': '00:00',
                    'currency': 'USD',
                    'event': 'Economic Event',
                    'importance': 'Medium',
                    'actual': None,
                    'forecast': None,
                    'previous': None
                })
            
            df = pd.DataFrame(econ_data)
            self.logger.info(f"📅 Retrieved economic calendar for {len(df)} events")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get economic calendar: {e}")
            return pd.DataFrame()
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get data from cache"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.execute(
                "SELECT data FROM data_cache WHERE cache_key = ? AND expires_at > ?",
                (cache_key, datetime.now())
            )
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: str, ttl_hours: int = 1):
        """Save data to cache"""
        try:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            conn = sqlite3.connect(self.cache_db)
            conn.execute(
                "INSERT OR REPLACE INTO data_cache (cache_key, data, expires_at) VALUES (?, ?, ?)",
                (cache_key, data, expires_at)
            )
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Cache save error: {e}")
    
    def _save_price_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save price data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df_db = df.reset_index().copy()
            df_db['symbol'] = symbol
            df_db['timeframe'] = timeframe
            
            df_db.to_sql('price_data', conn, if_exists='append', index=False, method='ignore')
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database save error: {e}")
    
    def _save_news_to_db(self, df: pd.DataFrame, symbol: str):
        """Save news data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df_db = df.copy()
            df_db['symbol'] = symbol
            
            df_db.to_sql('news_data', conn, if_exists='append', index=False, method='ignore')
            conn.close()
            
        except Exception as e:
            self.logger.error(f"News database save error: {e}")
    
    def get_latest_price(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get latest price for symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """SELECT timestamp, open, high, low, close, volume 
                   FROM price_data 
                   WHERE symbol = ? AND timeframe = ? 
                   ORDER BY timestamp DESC LIMIT 1""",
                (symbol, timeframe)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'timestamp': result[0],
                    'open': result[1],
                    'high': result[2],
                    'low': result[3],
                    'close': result[4],
                    'volume': result[5]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest price: {e}")
            return None
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for symbol"""
        asset_type = ASSET_TYPES.get(symbol, "forex")
        market_hours = MARKET_HOURS.get(asset_type, {"open": 0, "close": 24})
        
        now = datetime.now()
        current_hour = now.hour
        
        if asset_type == "crypto":
            return True  # Crypto markets are always open
        
        # Weekend check for Forex and Commodities
        if asset_type in ["forex", "commodity"]:
            if now.weekday() >= 5:  # Weekend
                return False
        
        # Regular market hours
        return market_hours["open"] <= current_hour < market_hours["close"]

# Global data manager instance
data_manager = EnhancedDataManager()