#!/usr/bin/env python3
"""
Test script cho Trading Bot
===========================

Script đơn giản để test các components chính của bot mà không cần chạy full trading loop.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import các components từ trading_bot
try:
    from trading_bot import (
        AdvancedFeatureEngineer,
        TechnicalIndicators,
        WyckoffAnalyzer,
        SupplyDemandAnalyzer,
        RSIDivergenceDetector,
        MarketRegimeDetector,
        EnsembleModel,
        LSTMModel,
        log_manager
    )
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    print("Vui lòng đảm bảo trading_bot.py đã được tạo đầy đủ")
    sys.exit(1)

def create_sample_data():
    """Tạo dữ liệu mẫu để test"""
    print("🔧 Tạo dữ liệu mẫu...")
    
    # Generate random OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    
    base_price = 2000.0
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"✅ Tạo được {len(df)} mẫu dữ liệu từ {df.index[0]} đến {df.index[-1]}")
    return df

async def test_feature_engineering():
    """Test Advanced Feature Engineering"""
    print("\n🔧 Testing Advanced Feature Engineering...")
    
    try:
        # Create sample data
        df = create_sample_data()
        
        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer()
        
        # Engineer features
        featured_df = feature_engineer.engineer_features(df, 'XAUUSD')
        
        print(f"✅ Features engineered: {len(featured_df.columns)} columns")
        print(f"📊 Sample features:")
        
        # Show some sample features
        feature_cols = [col for col in featured_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        sample_features = feature_cols[:10]  # Show first 10 features
        
        for feature in sample_features:
            if feature in featured_df.columns:
                value = featured_df[feature].iloc[-1]
                print(f"   {feature}: {value:.4f}")
        
        return featured_df
        
    except Exception as e:
        print(f"❌ Lỗi feature engineering: {e}")
        return None

async def test_technical_indicators():
    """Test Technical Indicators calculation"""
    print("\n📊 Testing Technical Indicators...")
    
    try:
        df = create_sample_data()
        
        # Test basic indicators
        indicators = TechnicalIndicators()
        
        # Basic indicators
        result_df = indicators.calculate_basic_indicators(df)
        print(f"✅ Basic indicators calculated: {len([col for col in result_df.columns if col.startswith(('rsi', 'atr', 'adx', 'macd'))])}")
        
        # Statistical features
        result_df = indicators.calculate_statistical_features(result_df)
        print(f"✅ Statistical features added: {len([col for col in result_df.columns if col.startswith(('returns', 'volatility', 'momentum'))])}")
        
        # Candlestick patterns
        result_df = indicators.detect_candlestick_patterns(result_df)
        pattern_cols = [col for col in result_df.columns if col in ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing']]
        print(f"✅ Candlestick patterns detected: {len(pattern_cols)}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Lỗi technical indicators: {e}")
        return None

async def test_wyckoff_analysis():
    """Test Wyckoff Analysis"""
    print("\n🔍 Testing Wyckoff Analysis...")
    
    try:
        df = create_sample_data()
        
        # Add volatility column for Wyckoff analysis
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        
        wyckoff = WyckoffAnalyzer()
        result_df = wyckoff.identify_wyckoff_signals(df)
        
        wyckoff_cols = [col for col in result_df.columns if 'wyckoff|spring|upthrust|accumulation|distribution' in col.lower()]
        print(f"✅ Wyckoff signals calculated: {len(wyckoff_cols)}")
        
        # Check for any detected signals
        spring_count = result_df['spring_detected'].sum() if 'spring_detected' in result_df.columns else 0
        upthrust_count = result_df['upthrust_detected'].sum() if 'upthrust_detected' in result_df.columns else 0
        
        print(f"📈 Springs detected: {spring_count}")
        print(f"📉 Upthrusts detected: {upthrust_count}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Lỗi Wyckoff analysis: {e}")
        return None

async def test_supply_demand():
    """Test Supply/Demand Zone Analysis"""
    print("\n🎯 Testing Supply/Demand Analysis...")
    
    try:
        df = create_sample_data()
        
        sda = SupplyDemandAnalyzer()
        result_df = sda.find_supply_demand_zones(df)
        
        sd_cols = [col for col in result_df.columns if 'supply|demand' in col]
        print(f"✅ Supply/Demand zones calculated: {len(sd_cols)}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Lỗi Supply/Demand analysis: {e}")
        return None

async def test_market_regime():
    """Test Market Regime Detection"""
    print("\n🎪 Testing Market Regime Detection...")
    
    try:
        df = create_sample_data()
        
        # Add required volatility column
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        
        mrd = MarketRegimeDetector()
        result_df = mrd.classify_market_regime(df)
        
        if 'market_regime' in result_df.columns:
            regime_counts = result_df['market_regime'].value_counts()
            print(f"✅ Market regimes detected:")
            for regime, count in regime_counts.items():
                print(f"   {regime}: {count} periods")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Lỗi market regime detection: {e}")
        return None

async def test_ensemble_model():
    """Test Ensemble Model (simplified)"""
    print("\n🤖 Testing Ensemble Model...")
    
    try:
        # Get feature-engineered data
        featured_df = await test_feature_engineering()
        
        if featured_df is None or featured_df.empty:
            print("⚠️ Không có dữ liệu để test ensemble model")
            return None
        
        # Create simple targets
        featured_df['target'] = (featured_df['close'].shift(-1) > featured_df['close']).astype(int)
        featured_df = featured_df.dropna()
        
        if len(featured_df) < 100:
            print("⚠️ Không đủ dữ liệu để train model")
            return None
        
        # Initialize ensemble model
        ensemble = EnsembleModel('XAUUSD')
        
        # Train ensemble (simplified - use small sample)
        train_data = featured_df.tail(500)  # Use last 500 samples
        
        print(f"📊 Training với {len(train_data)} mẫu...")
        
        # Train without hyperparameter optimization for speed
        training_stats = ensemble.train_ensemble(train_data, optimize_hp=False)
        
        if training_stats:
            print("✅ Ensemble model training completed")
            print(f"   Models trained: {training_stats.get('models_trained', [])}")
            
            # Test prediction
            if ensemble.models:
                signal, confidence, details = ensemble.predict_ensemble(train_data.tail(1))
                print(f"🎯 Sample prediction: {signal.value} (confidence: {confidence:.3f})")
        else:
            print("⚠️ Training không thành công")
        
        return ensemble
        
    except Exception as e:
        print(f"❌ Lỗi ensemble model: {e}")
        return None

async def test_rsi_divergence():
    """Test RSI Divergence Detection"""
    print("\n🔺 Testing RSI Divergence Detection...")
    
    try:
        df = create_sample_data()
        
        # Add RSI column
        from ta.momentum import RSIIndicator
        rsi = RSIIndicator(df['close'], window=14)
        df['rsi_14'] = rsi.rsi()
        
        rdd = RSIDivergenceDetector()
        result_df = rdd.detect_rsi_divergence(df)
        
        divergence_cols = [col for col in result_df.columns if 'divergence' in col]
        print(f"✅ RSI divergences detected: {len(divergence_cols)}")
        
        # Count divergences
        bullish_div = result_df['rsi_bullish_divergence'].sum() if 'rsi_bullish_divergence' in result_df.columns else 0
        bearish_div = result_df['rsi_bearish_divergence'].sum() if 'rsi_bearish_divergence' in result_df.columns else 0
        
        print(f"📈 Bullish divergences: {bullish_div}")
        print(f"📉 Bearish divergences: {bearish_div}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Lỗi RSI divergence detection: {e}")
        return None

async def run_all_tests():
    """Chạy tất cả tests"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  🧪 AI TRADING BOT TESTS v1.0 🧪             ║
    ║                                                              ║
    ║  Testing:                                                    ║  
    ║  ✅ Advanced Feature Engineering                            ║
    ║  ✅ Technical Indicators & Pattern Recognition              ║
    ║  ✅ Wyckoff Analysis                                        ║
    ║  ✅ Supply/Demand Zone Detection                            ║
    ║  ✅ Market Regime Classification                             ║
    ║  ✅ RSI Divergence Detection                                 ║
    ║  ✅ Machine Learning Models                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    test_results = {}
    
    # Run individual tests
    test_functions = [
        ("Feature Engineering", test_feature_engineering),
        ("Technical Indicators", test_technical_indicators),
        ("Wyckoff Analysis", test_wyckoff_analysis),
        ("Supply/Demand", test_supply_demand),
        ("Market Regime", test_market_regime),
        ("RSI Divergence", test_rsi_divergence),
        ("Ensemble Model", test_ensemble_model),
    ]
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*60}")
            result = await test_func()
            test_results[test_name] = "✅ PASSED" if result is not None else "❌ FAILED"
        except Exception as e:
            print(f"💥 CRITICAL ERROR in {test_name}: {e}")
            test_results[test_name] = "💥 ERROR"
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY:")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, status in test_results.items():
        print(f"{test_name:.<25} {status}")
        if status == "✅ PASSED":
            passed += 1
    
    print(f"{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Tất cả tests đều PASSED! Bot sẵn sàng để sử dụng.")
    elif passed > total // 2:
        print("⚠️ Một số tests failed nhưng core functionality hoạt động.")
    else:
        print("❌ Nhiều tests failed. Kiểm tra dependencies và configuration.")
    
    return passed == total

async def main():
    """Main test function"""
    try:
        await run_all_tests()
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical test error: {e}")

if __name__ == "__main__":
    # Run tests
    print("Starting automated tests...")
    asyncio.run(main())