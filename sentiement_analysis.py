"""
AI-Powered Sentiment Analysis using Large Language Models
News sentiment analysis, economic calendar processing, and LLM integration
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import google.generativeai as genai

from logger import get_logger
from config import api_config

class LLMSentimentAnalyzer:
    """Advanced sentiment analysis using Google Gemini LLM"""
    
    def __init__(self):
        self.logger = get_logger("LLMSentimentAnalyzer")
        
        # Configure Gemini API
        genai.configure(api_key="YOUR_GEMINI_API_KEY")  # You need to add your actual API key
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Sentiment prompt template
        self.sentiment_prompt = """
        Analyze the sentiment of the following financial news article for trading purposes.
        
        Article Title: {title}
        Article Content: {content}
        
        Please provide:
        1. Sentiment Score (-1 to +1, where -1 is very bearish, 0 is neutral, +1 is very bullish)
        2. Confidence Level (0 to 1, where 0 is no confidence, 1 is very confident)
        3. Market Impact Level (Low, Medium, High)
        4. Key Themes (comma-separated list)
        5. Brief Reasoning (1-2 sentences)
        
        Respond in JSON format:
        {{
            "sentiment_score": score,
            "confidence": confidence_level,
            "market_impact": "impact_level",
            "key_themes": ["theme1", "theme2"],
            "reasoning": "brief explanation"
        }}
        """
        
        # Sentiment aggregation weights
        self.impact_weights = {
            "High": 0.5,
            "Medium": 0.3,
            "Low": 0.1
        }
    
    async def analyze_article(self, title: str, content: str, source: str = None) -> Dict[str, Any]:
        """Analyze single article sentiment using LLM"""
        
        try:
            prompt = self.sentiment_prompt.format(title=title, content=content)
            
            response = await self._call_gemini_api(prompt)
            
            if response:
                # Parse JSON response
                sentiment_data = json.loads(response)
                
                # Validate sentiment score
                sentiment_score = float(sentiment_data.get('sentiment_score', 0))
                sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
                
                result = {
                    'sentiment_score': sentiment_score,
                    'confidence': float(sentiment_data.get('confidence', 0.5)),
                    'market_impact': sentiment_data.get('market_impact', 'Low'),
                    'key_themes': sentiment_data.get('key_themes', []),
                    'reasoning': sentiment_data.get('reasoning', ''),
                    'source': source,
                    'timestamp': datetime.now()
                }
                
                self.logger.debug(f"📊 Analyzed article: sentiment={sentiment_score:.2f}, "
                                f"confidence={result['confidence']:.2f}")
                
                return result
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis failed: {e}")
            
            # Fallback to rule-based sentiment
            return self.get_rule_based_sentiment(title, content, source)
    
    async def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API for sentiment analysis"""
        try:
            # For demonstration, return mock response
            # In production, uncomment the actual API call below
            
            # response = self.model.generate_content(prompt)
            # return response.text
            
            # Mock response for demonstration
            mock_response = {
                "sentiment_score": np.random.uniform(-0.8, 0.8),
                "confidence": np.random.uniform(0.6, 0.9),
                "market_impact": np.random.choice(["Low", "Medium", "High"]),
                "key_themes": ["earnings", "market_volatility"],
                "reasoning": "Based on technical and fundamental analysis"
            }
            
            return json.dumps(mock_response)
            
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return None
    
    def get_rule_based_sentiment(self, title: str, content: str, source: str = None) -> Dict[str, Any]:
        """Fallback rule-based sentiment analysis"""
        
        text = f"{title} {content}".lower()
        
        # Positive keywords
        positive_words = [
            'bullish', 'rise', 'gain', 'profit', 'growth', 'positive', 'strong',
            'outperform', 'beat', 'exceed', 'surge', 'rally', 'optimism'
        ]
        
        # Negative keywords  
        negative_words = [
            'bearish', 'fall', 'drop', 'loss', 'decline', 'negative', 'weak',
            'underperform', 'miss', 'disappoint', 'crash', 'selloff', 'pessimism'
        ]
        
        # Calculate sentiment score
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_words = len(text.split())
        sentiment_score = (positive_count - negative_count) / max(total_words, 1) * 10
        sentiment_score = max(-1, min(1, sentiment_score))
        
        # Determine confidence based on keyword density
        keyword_density = (positive_count + negative_count) / max(total_words, 1)
        confidence = min(0.8, keyword_density * 2)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'market_impact': 'Low',
            'key_themes': [],
            'reasoning': 'Rule-based sentiment analysis',
            'source': source
            
            'timestamp': datetime.now()
        }
    
    async def analyze_batch(self, articles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Analyze multiple articles in batch"""
        
        self.logger.info(f"📰 Analyzing sentiment for {len(articles)} articles")
        
        tasks = []
        for article in articles:
            task = self.analyze_article(
                article['title'],
                article.get('content', ''),
                article.get('source', '')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        
        self.logger.info(f"✅ Processed {len(valid_results)}/{len(articles)} articles")
        
        return valid_results
    
    def aggregate_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment analysis results"""
        
        if not sentiment_results:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'market_impact': 'Low',
                'article_count': 0,
                'themes': [],
                'timestamp': datetime.now()
            }
        
        # Weight by confidence and market impact
        weighted_scores = []
        theme_counts = {}
        impact_scores = []
        
        for result in sentiment_results:
            confidence = result['confidence']
            impact = result['market_impact']
            score = result['sentiment_score']
            
            # Apply weighting
            weight = confidence * self.impact_weights.get(impact, 0.1)
            weighted_scores.append(score * weight)
            impact_scores.append(weight)
            
            # Count themes
            for theme in result['key_themes']:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Calculate weighted average sentiment
        if weighted_scores and impact_scores:
            total_weight = sum(impact_scores)
            overall_sentiment = sum(weighted_scores) / total_weight
            
            # Calculate overall confidence
            avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
            
            # Determine dominant market impact
            impact_level = max(self.impact_weights.keys(), 
                             key=lambda x: sum(self.impact_weights[x] * 
                                             len([r for r in sentiment_results if r['market_impact'] == x])))
        else:
            overall_sentiment = 0.0
            avg_confidence = 0.0
            impact_level = 'Low'
        
        # Get top themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'market_impact': impact_level,
            'article_count': len(sentiment_results),
            'themes': [theme for theme, count in top_themes],
            'individual_results': sentiment_results,
            'timestamp': datetime.now()
        }

class EconomicCalendarProcessor:
    """Process economic calendar data for trading decisions"""
    
    def __init__(self):
        self.logger = get_logger("EconomicCalendarProcessor")
        
        # Major economic events and their market impact
        self.event_impact = {
            'nonfarm payroll': 'High',
            'fomc': 'High',
            'cpi': 'High',
            'ppi': 'Medium',
            'retail sales': 'Medium',
            'gdp': 'High',
            'rate decision': 'High',
            'employment': 'Medium',
            'inflation': 'High',
            'manufacturing': 'Medium',
            'consumer confidence': 'Medium',
            'trade balance': 'Low',
            'balance': 'Low'
        }
        
        # Currency sensitive events
        self.currency_events = {
            'EUR': ['ecb', 'final manufacturing', 'retail sales', 'inflation'],
            'GBP': ['boe', 'retail sales', 'inflation', 'manufacturing'],
            'JPY': ['boj', 'core cpi', 'trade balance', 'ppi'],
            'AUD': ['rba', 'employment', 'retail sales', 'trade'],
            'USD': ['fomc', 'nonfarm payroll', 'cpi', 'ppi', 'gdp']
        }
    
    def categorize_event(self, event_name: str, event_desc: str) -> Dict[str, Any]:
        """Categorize economic event"""
        
        name_lower = event_name.lower()
        desc_lower = event_desc.lower()
        
        # Detect impact level
        impact = 'Low'
        for keyword, level in self.event_impact.items():
            if keyword in name_lower or keyword in desc_lower:
                impact = level
                break
        
        # Detect affected currencies/assets
        affected_assets = []
        text_to_check = f"{name_lower} {desc_lower}"
        
        for currency, keywords in self.currency_events.items():
            if any(keyword in text_to_check for keyword in keywords):
                affected_assets.append(f"{currency}USD")
        
        # Special handling for different asset types
        if 'gold' in text_to_check or 'xau' in text_to_check:
            affected_assets.append('XAUUSD')
        
        return {
            'event_name': event_name,
            'event_desc': event_desc,
            'impact': impact,
            'affected_assets': affected_assets,
            'keywords': self.extract_keywords(text_to_check)
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from event text"""
        
        keywords = []
        
        # Time keywords
        time_words = ['today', 'tomorrow', 'this week', 'monthly', 'quarterly', 'annual']
        
        # Economic keywords
        econ_words = ['expectation', 'forecast', 'previous', 'revised', 'preliminary', 'final']
        
        all_keywords = time_words + econ_words + list(self.event_impact.keys())
        
        for keyword in all_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def should_gate_trading(self, events: List[Dict[str, Any]], 
                           current_time: datetime, symbol: str) -> Tuple[bool, str]:
        """Determine if trading should be gated due to major events"""
        
        gating_reason = None
        
        for event in events:
            event_time = event.get('time', current_time)
            time_diff_hours = abs((current_time - event_time).total_seconds() / 3600)
            
            # Check for high-impact events within gating period
            if event.get('impact') == 'High' and time_diff_hours <= 2:
                
                # Check if symbol is affected
                affected_assets = event.get('affected_assets', [])
                
                if symbol in affected_assets or not affected_assets:
                    gating_reason = (f"High impact event: {event.get('event_name')} "
                                   f"in {time_diff_hours:.1f}h")
                    return True, gating_reason
        
        return False, "No gating events"
    
    def prioritize_events(self, events: List[Dict[str, Any]], 
                          current_time: datetime) -> List[Dict[str, Any]]:
        """Prioritize events by impact and timing"""
        
        for event in events:
            event_time = event.get('time', current_time)
            time_diff_hours = (event_time - current_time).total_seconds() / 3600
            
            # Priority calculation
            impact_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(event.get('impact'), 0)
            urgency_score = max(0, 3 - abs(time_diff_hours))  # Higher for closer events
            
            priority = impact_score + urgency_score
            event['priority'] = priority
            event['hours_until'] = time_diff_hours
        
        # Sort by priority
        return sorted(events, key=lambda x: x['priority'], reverse=True)

class NewsEconomicManager:
    """Integration manager for news and economic data"""
    
    def __init__(self):
        self.logger = get_logger("NewsEconomicManager")
        
        # Initialize components
        self.sentiment_analyzer = LLMSentimentAnalyzer()
        self.economic_processor = EconomicCalendarProcessor()
        
        # Aggregated data cache
        self.news_cache = {}
        self.economic_cache = {}
    
    async def process_symbol_news(self, symbol: str, 
                                news_data: pd.DataFrame) -> Dict[str, Any]:
        """Process news data for a symbol"""
        
        self.logger.info(f"📰 Processing news for {symbol}")
        
        if news_data.empty:
            return self.get_empty_sentiment_result()
        
        # Convert DataFrame to article list
        articles = []
        for idx, row in news_data.iterrows():
            article = {
                'title': str(row.get('title', '')),
                'content': str(row.get('content', row.get('description', ''))),
                'source': str(row.get('source', 'unknown')),
                'published_at': row.get('published_at')
            }
            articles.append(article)
        
        # Analyze sentiment
        sentiment_results = await self.sentiment_analyzer.analyze_batch(articles)
        
        # Aggregate results
        aggregated_sentiment = self.sentiment_analyzer.aggregate_sentiment(sentiment_results)
        
        # Cache results
        self.news_cache[symbol] = {
            'sentiment_data': aggregated_sentiment,
            'individual_articles': sentiment_results,
            'timestamp': datetime.now()
        }
        
        # Log results
        self.logger.info(f"✅ {symbol} sentiment: {aggregated_sentiment['overall_sentiment']:.2f} "
                        f"(confidence: {aggregated_sentiment['confidence']:.2f})")
        
        return aggregated_sentiment
    
    def get_empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result"""
        return {
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'market_impact': 'Low',
            'article_count': 0,
            'themes': [],
            'timestamp': datetime.now()
        }
    
    async def process_economic_calendar(self, calendar_data: pd.DataFrame) -> Dict[str, Any]:
        """Process economic calendar data"""
        
        self.logger.info(f"📅 Processing economic calendar")
        
        if calendar_data.empty:
            return {'events': [], 'trading_gates': {}}
        
        processed_events = []
        current_time = datetime.now()
        
        # Process each event
        for idx, row in calendar_data.iterrows():
            event = {
                'event_name': str(row.get('event', 'Unknown Event')),
                'event_desc': str(row.get('description', '')),
                'time': row.get('time', current_time),
                'importance': str(row.get('importance', 'Low'))
            }
            
            # Categorize event
            categorized_event = self.economic_processor.categorize_event(
                event['event_name'], event['event_desc']
            )
            event.update(categorized_event)
            
            processed_events.append(event)
        
        # Prioritize events
        prioritized_events = self.economic_processor.prioritize_events(
            processed_events, current_time
        )
        
        # Generate trading gates
        trading_gates = {}
        major_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'NASDAQ100']
        
        for symbol in major_symbols:
            should_gate, reason = self.economic_processor.should_gate_trading(
                processed_events, current_time, symbol
            )
            trading_gates[symbol] = {
                'should_gate',
                'reason': reason
            }
        
        result = {
            'events': prioritized_events,
            'trading_gates': trading_gates,
            'high_impact_count': len([e for e in processed_events if e.get('impact') == 'High']),
            'timestamp': datetime.now()
        }
        
        # Cache results
        self.economic_cache = result
        
        self.logger.info(f"📅 Processed {len(processed_events)} economic events")
        
        return result
    
    def get_news_features(self, symbol: str) -> Dict[str, float]:
        """Extract news-based features for ML models"""
        
        if symbol not in self.news_cache:
            return {
                'news_sentiment': 0.0,
                'news_confidence': 0.0,
                'news_volume': 0.0,
                'news_impact_score': 0.0,
                'news_diversity': 0.0
            }
        
        sentiment_data = self.news_cache[symbol]['sentiment_data']
        articles = self.news_cache[symbol]['individual_articles']
        
        # Calculate features
        sentiment = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        volume = sentiment_data['article_count']
        
        # Impact score
        impact_scores = {'High': 1.0, 'Medium': 0.5, 'Low': 0.1}
        impact_score = np.mean([
            impact_scores.get(article['market_impact'], 0.1) * article['confidence']
            for article in articles
        ]) if articles else 0.0
        
        # Diversity (number of unique themes)
        themes_count = len(sentiment_data.get('themes', []))
        diversity = min(1.0, themes_count / 5.0)  # Normalize to max 5 themes
        
        return {
            'news_sentiment': sentiment,
            'news_confidence': confidence,
            'news_volume': volume,
            'news_impact_score': impact_score,
            'news_diversity': diversity
        }
    
    def get_trading_gates(self, symbol: str) -> Tuple[bool, str]:
        """Get current trading gates for symbol"""
        
        if 'trading_gates' not in self.economic_cache:
            return False, "No economic events"
        
        symbol_gates = self.economic_cache['trading_gates'].get(symbol, {})
        
        return (
            symbol_gates.get('should_gate', False),
            symbol_gates.get('reason', 'Clear to trade')
        )
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics"""
        
        return {
            'news_cache_size': len(self.news_cache),
            'economic_cache_exists': bool(self.economic_cache),
            'cache_timestamps': {
                symbol: data['timestamp'].isoformat()
                for symbol, data in self.news_cache.items()
            },
            'economic_cache_time': self.economic_cache.get('timestamp').isoformat() 
                                 if self.economic_cache else None
        }

# Global news economic manager instance
news_manager = NewsEconomicManager()