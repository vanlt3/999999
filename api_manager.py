"""
Advanced API Manager with Rate Limiting, Retry Logic, and Health Monitoring
Manages multiple data providers with intelligent fallback and monitoring
"""

import asyncio
import aiohttp
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json

from logger import api_logger, get_logger
from config import api_config

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    base_url: str
    rate_limit: int  # requests per minute
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    retry_attempts: int = 3
    timeout: int = 30

class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, rate_limit: int, burst_size: int = 10):
        self.rate_limit = rate_limit
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        
    async def acquire(self):
        """Wait for token to be available"""
        while True:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Refill tokens based on time passed

            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * self.rate_limit / 60
            )
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
                
            # Wait for next refill
            await asyncio.sleep(0.1)

class APIMonitoringSystem:
    """Monitor API health and performance"""
    
    def __init__(self):
        self.metrics = {}
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 5.0,  # 5 seconds
            "down_time": 300  # 5 minutes
        }
        
    def record_request(self, api_name: str, success: bool, response_time: float):
        """Record API request metrics"""
        if api_name not in self.metrics:
            self.metrics[api_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0,
                "last_successful": None,
                "last_failure": None,
                "recent_errors": deque(maxlen=10)
            }
        
        metrics = self.metrics[api_name]
        metrics["total_requests"] += 1
        metrics["total_response_time"] += response_time
        
        if success:
            metrics["successful_requests"] += 1
            metrics["last_successful"] = datetime.now()
            if "down" in metrics:
                del metrics["down"]
        else:
            metrics["failed_requests"] += 1
            metrics["last_failure"] = datetime.now()
            metrics["recent_errors"].append(datetime.now())
            
            # Check if API is down
            if len(metrics["recent_errors"]) >= 5:
                if not metrics.get("down"):
                    metrics["down"] = datetime.now()
                    api_logger.warning(f"🚨 API {api_name} appears to be DOWN!")
    
    def get_health_status(self, api_name: str) -> Dict[str, Any]:
        """Get health status of API"""
        if api_name not in self.metrics:
            return {"status": "unknown", "message": "No data"}
        
        metrics = self.metrics[api_name]
        
        # Calculate metrics
        error_rate = metrics["failed_requests"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0
        avg_response_time = metrics["total_response_time"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0
        
        # Determine status
        if metrics["down"]:
            down_time = (datetime.now() - metrics["down"]).total_seconds()
            return {
                "status": "down",
                "down_time": down_time,
                "message": f"API has been down for {down_time:.0f} seconds"
            }
        elif error_rate > self.alert_thresholds["error_rate"]:
            return {
                "status": "degraded",
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "message": f"High error rate: {error_rate:.1%}"
            }
        elif avg_response_time > self.alert_thresholds["response_time"]:
            return {
                "status": "slow",
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "message": f"Slow response time: {avg_response_time:.1f}s"
            }
        else:
            return {
                "status": "healthy",
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "message": "API is healthy"
            }

class APIManager:
    """Centralized API management system"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.monitoring = APIMonitoringSystem()
        self.endpoints = self._setup_endpoints()
        self.logger = api_logger
        
    def _setup_endpoints(self) -> Dict[str, APIEndpoint]:
        """Setup API endpoints"""
        endpoints = {
            "alpha_vantage": APIEndpoint(
                name="alpha_vantage",
                base_url="https://www.alphavantage.co/query",
                rate_limit=500,
                api_key=api_config.vantage_api_key
            ),
            "finnhub": APIEndpoint(
                name="finnhub",
                base_url="https://finnhub.io/api/v1",
                rate_limit=60,
                api_key=api_config.finnhub_api_key
            ),
            "marketaux": APIEndpoint(
                name="marketaux",
                base_url="https://api.marketaux.com/v1",
                rate_limit=100,
                api_key=api_config.marketaux_api_key
            ),
            "newsapi": APIEndpoint(
                name="newsapi",
                base_url="https://newsapi.org/v2",
                rate_limit=1000,
                api_key=api_config.newsapi_key
            ),
            "eodhd": APIEndpoint(
                name="eodhd", 
                base_url="https://eodhistoricaldata.com/api",
                rate_limit=1000,
                api_key=api_config.eodhd_api_key
            )
        }
        
        # Initialize rate limiters
        for name, endpoint in endpoints.items():
            self.rate_limiters[name] = RateLimiter(endpoint.rate_limit)
            
        return endpoints
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "TradingBot/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, api_name: str, url: str, params: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Make rate-limited API request with retry logic"""
        if api_name not in self.endpoints:
            raise ValueError(f"Unknown API: {api_name}")
        
        endpoint = self.endpoints[api_name]
        
        # Wait for rate limit
        await self.rate_limiters[api_name].acquire()
        
        # Prepare request parameters
        if endpoint.api_key:
            params = params or {}
            params["apikey"] = endpoint.api_key
        
        request_headers = endpoint.headers.copy() if endpoint.headers else {}
        if headers:
            request_headers.update(headers)
        
        # Retry logic
        last_exception = None
        for attempt in range(endpoint.retry_attempts):
            start_time = time.time()
            
            try:
                async with self.session.get(url, params=params, headers=request_headers) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        self.monitoring.record_request(api_name, True, response_time)
                        data = await response.json()
                        
                        # Check for API-specific error responses
                        if self._check_api_error(data, api_name):
                            raise aiohttp.ClientError(f"API error in response: {data}")
                        
                        self.logger.debug(f"✅ {api_name} request successful ({response_time:.2f}s)")
                        return data
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                        
            except Exception as e:
                response_time = time.time() - start_time
                self.monitoring.record_request(api_name, False, response_time)
                last_exception = e
                
                self.logger.warning(f"⚠️ {api_name} request failed (attempt {attempt + 1}): {e}")
                
                if attempt < endpoint.retry_attempts - 1:
                    # Exponential backoff with jitter
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(min(backoff_time, 10))
        
        # All retries failed
        self.logger.error(f"❌ {api_name} request failed after {endpoint.retry_attempts} attempts")
        raise last_exception
    
    def _check_api_error(self, data: Dict[str la, Any], api_name: str) -> bool:
        """Check for API-specific error indicators"""
        if api_name == "alpha_vantage" and "Error Message" in data:
            return True
        elif api_name == "finnhub" and "error" in data:
            return True
        elif api_name == "marketaux" and data.get("status") == "error":
            return True
        elif api_name == "newsapi" and data.get("status") != "ok":
            return True
        
        return False
    
    # Specific API methods
    async def get_price_data(self, symbol: str, timeframe: str, api_name: str = "alpha_vantage") -> Dict[str, Any]:
        """Get OHLCV price data"""
        if api_name == "alpha_vantage":
            url = f"{self.endpoints[api_name].base_url}"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": timeframe,
                "outputsize": "compact",
                "datatype": "json"
            }
            return await self._make_request(api_name, url, params)
        
        elif api_name == "eodhd":
            url = f"{self.endpoints[api_name].base_url}/intraday/{symbol}.US"
            params = {
                "interval": timeframe,
                "fmt": "json",
                "order": "a"
            }
            return await self._make_request(api_name, url, params)
    
    async def get_news_data(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get news data from multiple sources"""
        all_news = []
        
        # NewsAPI
        try:
            url = f"{self.endpoints['newsapi'].base_url}/everything"
            params = {
                "q": symbol,
                "from": (datetime.now() - timedelta(days=days_back)).date(),
                "sortBy": "publishedAt",
                "language": "en"
            }
            newsapi_data = await self._make_request("newsapi", url, params)
            all_news.extend(newsapi_data.get("articles", []))
        except Exception as e:
            self.logger.warning(f"Failed to get NewsAPI data: {e}")
        
        # Marketaux
        try:
            url = f"{self.endpoints['marketaux'].base_url}/news/all"
            params = {
                "symbols": symbol,
                "limit": 50,
                "published_on_from": (datetime.now() - timedelta(days=days_back)).date(),
                "published_on_to": datetime.now().date()
            }
            marketaux_data = await self._make_request("marketaux", url, params)
            all_news.extend(marketaux_data.get("data", []))
        except Exception as e:
            self.logger.warning(f"Failed to get Marketaux data: {e}")
        
        # Finnhub
        try:
            url = f"{self.endpoints['finnhub'].base_url}/company-news"
            params = {
                "symbol": symbol,
                "from": (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                "to": datetime.now().strftime('%Y-%m-%d')
            }
            finnhub_data = await self._make_request("finnhub", url, params)
            all_news.extend(finnhub_data or [])
        except Exception as e:
            self.logger.warning(f"Failed to get Finnhub data: {e}")
        
        return all_news
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information"""
        url = f"{self.endpoints['finnhub'].base_url}/stock/profile2"
        params = {"symbol": symbol}
        return await self._make_request("finnhub", url, params)
    
    async def get_technical_indicators(self, symbol: str, indicator: str) -> Dict[str, Any]:
        """Get technical indicators from Alpha Vantage"""
        url = f"{self.endpoints['alpha_vantage'].base_url}"
        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": "daily",
            "time_period": "20",
            "series_type": "close"
        }
        return await self._make_request("alpha_vantage", url, params)
    
    def get_api_health_status(self) -> Dict[str, Any]:
        """Get health status of all APIs"""
        health_status = {}
        for api_name in self.endpoints.keys():
            health_status[api_name] = self.monitoring.get_health_status(api_name)
        
        return health_status
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        stats = {}
        for api_name, metrics in self.monitoring.metrics.items():
            stats[api_name] = {
                "total_requests": metrics["total_requests"],
                "success_rate": metrics["successful_requests"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0,
                "avg_response_time": metrics["total_response_time"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0,
                "last_successful": metrics["last_successful"],
                "status": self.monitoring.get_health_status(api_name)["status"]
            }
        
        return stats

# Global API manager instance
api_manager = APIManager()