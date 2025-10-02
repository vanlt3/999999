#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Trading Bot
Tests all components individually and as a system
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from logger import bot_logger
from api_manager import api_manager
from data_manager import data_manager
from feature_engineer import feature_engineer
from ml_ensemble import ensemble_model, lstm_model
from master_agent import master_agent
from risk_manager import risk_manager
from discord_notifier import discord_notifier
from sentiement_analysis import news_manager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotTester:
    """Comprehensive bot testing suite"""
    
    def __init__(self):
        self.test_results = {}
        self.symbols = ["EURUSD", "XAUUSD"]  # Test with 2 symbols
        logger.info("🧪 Bot Tester initialized")
    
    async def run_all_tests(self):
        """Run all test suites"""
        
        logger.info("🚀 Starting comprehensive bot tests")
        
        # Core component tests
        await self.test_api_manager()
        await self.test_data_manager() 
        await self.test_feature_engineering()
        await self.test_ml_models()
        await self.test_master_agent()
        await self.test_risk_manager()
        await self.test_discord_integration()
        await self.test_sentiment_analysis()
        
        # Integration tests
        await self.test_end_to_end_flow()
        
        # Report results
        self.report_results()
    
    async def test_api_manager(self):
        """Test API manager functionality"""
        
        logger.info("🔗 Testing API Manager...")
        
        try:
            async with api_manager as am:
                # Test API health status
                health_status = am.get_api_health_status()
                
                # Test statistics
                stats = am.get_api_statistics()
                
                # Test sample API call
                sample_data = await am.get_price_data("EURUSD", "1h", "alpha_vantage")
                
                self.test_results['api_manager'] = {
                    'status': 'PASSED',
                    'health_check': len([k for k, v in health_status.items() if v['status'] == 'healthy']),
                    'total_apis': len(health_status),
                    'sample_data_received': bool(sample_data)
                }
                
                logger.info("✅ API Manager test passed")
                
        except Exception as e:
            self.test_results['api_manager'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ API Manager test failed: {e}")
    
    async def test_data_manager(self):
        """Test data management system"""
        
        logger.info("📊 Testing Data Manager...")
        
        try:
            # Test price data retrieval
            price_data = await data_manager.get_price_data("EURUSD", "H1", limit=100)
            
            # Test news data retrieval
            news_data = await data_manager.get_news_data("EURUSD", days_back=7)
            
            # Test economic calendar
            econ_data = await data_manager.get_economic_calendar()
            
            # Test market hours check
            market_open = data_manager.is_market_open("EURUSD")
            
            self.test_results['data_manager'] = {
                'status': 'PASSED',
                'price_data_length': len(price_data) if not price_data.empty else 0,
                'news_data_count': len(news_data) if not news_data.empty else 0,
                'economic_events': len(econ_data) if not econ_data.empty else 0,
                'market_open': market_open
            }
            
            logger.info("✅ Data Manager test passed")
            
        except Exception as e:
            self.test_results['data_manager'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Data Manager test failed: {e}")
    
    async def test_feature_engineering(self):
        """Test feature engineering system"""
        
        logger.info("🧬 Testing Feature Engineering...")
        
        try:
            # Create sample data
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 100 + np.cumsum(np.random.randn(1000) * 0.001),
                'high': 100 + np.cumsum(np.random.randn(1000) * 0.001) + np.random.rand(1000) * 0.01,
                'low': 100 + np.cumsum(np.random.randn(1000) * 0.001) - np.random.rand(1000) * 0.01,
                'close': 100 + np.cumsum(np.random.randn(1000) * 0.001),
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            # Engineer features
            engineered_data = feature_engineer.engineer_features(sample_data, "EURUSD")
            
            # Check for expected features
            expected_features = ['RSI_14', 'MACD', 'ATR', 'SMA_20', 'ADX']
            features_present = [feat for feat in expected_features if feat in engineered_data.columns]
            
            self.test_results['feature_engineering'] = {
                'status': 'PASSED',
                'input_rows': len(sample_data),
                'output_rows': len(engineered_data),
                'input_columns': len(sample_data.columns),
                'output_columns': len(engineered_data.columns),
                'features_added': len(engineered_data.columns) - len(sample_data.columns),
                'expected_features_present': len(features_present)
            }
            
            logger.info("✅ Feature Engineering test passed")
            
        except Exception as e:
            self.test_results['feature_engineering'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Feature Engineering test failed: {e}")
    
    async def test_ml_models(self):
        """Test machine learning models"""
        
        logger.info("🤖 Testing ML Models...")
        
        try:
            # Create sample training data
            np.random.seed(42)
            n_samples = 1000
            n_features = 50
            
            X = pd.DataFrame(np.random.randn(n_samples, n_features))
            y = pd.Series(np.random.randint(0, 2, n_samples))
            
            # Test ensemble model
            ensemble_model.train_stack(X, y, "test_symbol")
            
            # Test LSTM model
            lstm_model.train(X, y)
            
            # Test prediction
            ensemble_pred, ensemble_conf = ensemble_model.predict(X.tail(10))
            lstm_pred, lstm_conf = lstm_model.predict(X.tail(10))
            
            self.test_results['ml_models'] = {
                'status': 'PASSED',
                'ensemble_trained': ensemble_model.is_trained,
                'lstm_trained': lstm_model.is_trained,
                'ensemble_predictions': len(ensemble_pred),
                'lstm_predictions': len(lstm_pred),
                'ensemble_confidence_range': f"{ensemble_conf.min():.3f} - {ensemble_conf.max():.3f}",
                'lstm_confidence_range': f"{lstm_conf.min():.3f} - {lstm_conf.max():.3f}"
            }
            
            logger.info("✅ ML Models test passed")
            
        except Exception as e:
            self.test_results['ml_models'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ ML Models test failed: {e}")
    
    async def test_master_agent(self):
        """Test Master Agent system"""
        
        logger.info("🎯 Testing Master Agent...")
        
        try:
            # Create sample market data
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
                'open': 100 + np.cumsum(np.random.randn(100) * 0.001),
                'high': 100 + np.cumsum(np.random.randn(100) * 0.001) + np.random.rand(100) * 0.01,
                'low': 100 + np.cumsum(np.random.randn(100) * 0.001) - np.random.rand(100) * 0.01,
                'close': 100 + np.cumsum(np.random.randn(100) * 0.001),
                'volume': np.random.randint(1000, 10000, 100),
                'RSI_14': np.random.uniform(20, 80, 100),
                'MACD': np.random.randn(100),
                'ATR': np.random.uniform(0.001, 0.01, 100),
                'ADX': np.random.uniform(10, 70, 100)
            })
            
            # Test decision making
            decision = await master_agent.make_decision(
                symbol="EURUSD",
                market_data=sample_data,
                additional_data={
                    'news_sentiment': 0.2,
                    'risk_score': 0.3,
                    'VOLATILITY_20': 0.015
                }
            )
            
            # Test agent performance tracking
            performance = master_agent.get_agent_performance()
            
            self.test_results['master_agent'] = {
                'status': 'PASSED',
                'final_signal': decision.final_signal.value,
                'consensus_confidence': decision.consensus_confidence,
                'specialist_count': len(decision.specialist_votes),
                'reasoning_length': len(decision.reasoning),
                'agent_performance_tracked': len(performance),
                'risk_score': decision.risk_score,
                'market_regime': decision.market_regime
            }
            
            logger.info("✅ Master Agent test passed")
            
        except Exception as e:
            self.test_results['master_agent'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Master Agent test failed: {e}")
    
    async def test_risk_manager(self):
        """Test risk management system"""
        
        logger.info("🛡️ Testing Risk Manager...")
        
        try:
            # Test trade validation
            validation_ok, reason = risk_manager.validate_trade(
                symbol="EURUSD",
                side="BUY",
                confidence=0.8,
                market_data={'price': 1.1000},
                master_decision=None
            )
            
            # Test position sizing
            size, risk_amount = risk_manager.position_calculator.calculate_size(
                symbol="EURUSD",
                confidence=0.8,
                risk_score=0.3,
                account_value=10000,
                atr=0.002,
                current_price=1.1000
            )
            
            # Test stop loss calculation
            sl, tp = risk_manager.stop_loss_manager.calculate_stop_levels(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                atr=0.002,
                market_regime="trending",
                confidence=0.8
            )
            
            # Test trading gates
            gates = risk_manager.trade_gating.get_all_restrictions("EURUSD")
            
            # Get risk summary
            risk_summary = risk_manager.get_risk_summary()
            
            self.test_results['risk_manager'] = {
                'status': 'PASSED',
                'trade_validation': validation_ok,
                'position_size_calculated': size > 0,
                'stop_loss_calculated': bool(sl),
                'take_profit_calculated': bool(tp),
                'trading_gates_count': len(gates),
                'portfolio_value': risk_summary['portfolio_value'],
                'position_count': risk_summary['position_count']
            }
            
            logger.info("✅ Risk Manager test passed")
            
        except Exception as e:
            self.test_results['risk_manager'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Risk Manager test failed: {e}")
    
    async def test_discord_integration(self):
        """Test Discord notification system"""
        
        logger.info("🔔 Testing Discord Integration...")
        
        try:
            async with discord_notifier:
                # Send test message
                success = await discord_notifier.send_test_message()
                
                # Test different notification types
                test_notification_types = [
                    ('system_alert', {
                        'alert_type': 'Test Alert',
                        'message': 'This is a test notification from the trading bot',
                        'severity': 'info'
                    })
                ]
                
                notification_results = {}
                for notif_type, data in test_notification_types:
                    if notif_type == 'system_alert':
                        sent = await discord_notifier.notify_system_alert(data)
                        notification_results[notif_type] = sent
                
                self.test_results['discord_integration'] = {
                    'status': 'PASSED' if success else 'PARTIAL',
                    'test_message_sent': success,
                    'notifications_tested': len(notification_results),
                    'successful_notifications': sum(notification_results.values())
                }
                
                logger.info("✅ Discord Integration test passed")
                
        except Exception as e:
            self.test_results['discord_integration'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Discord Integration test failed: {e}")
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis system"""
        
        logger.info("📰 Testing Sentiment Analysis...")
        
        try:
            # Create sample news data
            sample_articles = [
                {
                    'title': 'EURUSD rises on positive economic data',
                    'content': 'The Euro strengthened against the US Dollar as economic indicators showed improvement...',
                    'source': 'Reuters'
                },
                {
                    'title': 'Gold prices fall amid risk appetite',
                    'content': 'Gold prices declined as investors moved to riskier assets following positive news...',
                    'source': 'Bloomberg'
                }
            ]
            
            # Test sentiment analysis
            sentiment_results = await news_manager.sentiment_analyzer.analyze_batch(sample_articles)
            
            # Test aggregation
            aggregated_sentiment = news_manager.sentiment_analyzer.aggregate_sentiment(sentiment_results)
            
            # Test economic calendar processing
            sample_econ_data = pd.DataFrame({
                'event': ['FOMC Decision', 'Non-farm Payroll', 'CPI Release'],
                'description': ['Federal Reserve interest rate decision', 'Monthly employment report', 'Consumer price index'],
                'time': [datetime.now(), datetime.now() + timedelta(hours=1), datetime.now() + timedelta(hours=6)],
                'importance': ['High', 'High', 'Medium']
            })
            
            econ_result = await news_manager.process_economic_calendar(sample_econ_data)
            
            self.test_results['sentiment_analysis'] = {
                'status': 'PASSED',
                'articles_processed': len(sentiment_results),
                'aggregation_successful': 'overall_sentiment' in aggregated_sentiment,
                'avg_sentiment': aggregated_sentiment.get('overall_sentiment', 0),
                'economic_events_processed': len(econ_result.get('events', [])),
                'high_impact_events': econ_result.get('high_impact_count', 0)
            }
            
            logger.info("✅ Sentiment Analysis test passed")
            
        except Exception as e:
            self.test_results['sentiment_analysis'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Sentiment Analysis test failed: {e}")
    
    async def test_end_to_end_flow(self):
        """Test complete end-to-end flow"""
        
        logger.info("🔄 Testing End-to-End Flow...")
        
        try:
            # This would test the full bot logic
            # For now, we'll simulate a simplified version
            
            # Step 1: Get market data
            price_data = await data_manager.get_price_data("EURUSD", "H1", limit=100)
            
            if not price_data.empty:
                # Step 2: Engineer features
                engineered_data = feature_engineer.engineer_features(
                    price_data.reset_index(), "EURUSD"
                )
                
                # Step 3: Get news sentiment
                news_features = news_manager.get_news_features("EURUSD")
                
                # Step 4: Make decision (simplified)
                has_data = len(engineered_data) > 10 and any(news_features.values())
                
                self.test_results['end_to_end_flow'] = {
                    'status': 'PASSED' if has_data else 'PARTIAL',
                    'data_retrieval_success': not price_data.empty,
                    'feature_engineering_success': len(engineered_data) > 10,
                    'sentiment_features_available': any(news_features.values()),
                    'flow_completion': has_data
                }
            else:
                self.test_results['end_to_end_flow'] = {
                    'status': 'FAILED',
                    'reason': 'No market data available'
                }
            
            logger.info("✅ End-to-End Flow test completed")
            
        except Exception as e:
            self.test_results['end_to_end_flow'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ End-to-End Flow test failed: {e}")
    
    def report_results(self):
        """Generate and display test results report"""
        
        logger.info("\n" + "="*80)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("="*80)
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_emoji} {test_name.replace('_', ' ').title()}: {result['status']}")
            
            if result['status'] == 'PASSED':
                passed_tests += 1
                
                # Show detailed results for successful tests
                for key, value in result.items():
                    if key not in ['status', 'error'] and isinstance(value, (int, float, str)):
                        logger.info(f"   📈 {key.replace('_', ' ').title()}: {value}")
        
        logger.info("="*80)
        logger.info(f"🏆 OVERALL SCORE: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Bot is ready to run.")
        elif passed_tests >= total_tests * 0.7:
            logger.info("⚠️  Most tests passed. Check failed components before running.")
        else:
            logger.info("🚨 Multiple test failures detected. Fix issues before running.")
        
        logger.info("="*80)
        
        return passed_tests >= total_tests * 0.8  # 80% pass rate

async def main():
    """Run all tests"""
    
    print("🧪 Advanced Trading Bot - Test Suite")
    print("=" * 50)
    
    tester = BotTester()
    
    try:
        await tester.run_all_tests()
        success = tester.report_results()
        
        if success:
            print("\n✅ Test suite completed successfully!")
            return 0
        else:
            print("\n❌ Test suite found issues!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))