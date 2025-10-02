"""
Advanced Feature Engineering System with Technical Indicators and Market Structure Analysis
Comprehensive feature engineering for trading models including Wyckoff methodology and advanced technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from logger import get_logger
from config import ASSET_TYPES

class TechnicalIndicators:
    """Comprehensive technical indicator calculations"""
    
    def __init__(self):
        self.logger = get_logger("TechnicalIndicators")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if len(df) < 50:
            self.logger.warning("Insufficient data for indicator calculation")
            return df.copy()
        
        result_df = df.copy()
        
        # Price-based indicators
        result_df = self._add_trend_indicators(result_df)
        result_df = self._add_oscillator_indicators(result_df)
        result_df = self._add_volatility_indicators(result_df)
        result_df = self._add_volume_indicators(result_df)
        result_df = self._add_fibonacci_levels(result_df)
        
        return result_df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        # Moving Averages
        df['SMA_5'] = talib.SMA(df['close'], timeperiod=5)
        df['SMA_10'] = talib.SMA(df['close'], timeperiod=10)
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)
        
        df['EMA_5'] = talib.EMA(df['close'], timeperiod=5)
        df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(df['high'], df['low'])
        
        return df
    
    def _add_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add oscillating indicators"""
        # RSI
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
        df['RSI_7'] = talib.RSI(df['close'], timeperiod=7)
        df['RSI_21'] = talib.RSI(df['close'], timeperiod=21)
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Money Flow Index
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # ROC (Rate of Change)
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # ATR (Average True Range)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['close'], timeperiod=20)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
        df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        
        # Keltner Channels
        df['KC_UPPER'] = df['EMA_20'] + (2 * df['ATR'])
        df['KC_MIDDLE'] = df['EMA_20']
        df['KC_LOWER'] = df['EMA_20'] - (2 * df['ATR'])
        
        # Volatility
        df['VOLATILITY'] = df['close'].pct_change().rolling(20).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        # On Balance Volume
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        
        # Average Directional Movement Index
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume SMA
        df['VOL_SMA_10'] = talib.SMA(df['volume'], timeperiod=10)
        df['VOL_SMA_20'] = talib.SMA(df['volume'], timeperiod=20)
        
        # Volume Rate of Change
        df['VOL_ROC'] = talib.ROC(df['volume'], timeperiod=10)
        
        return df
    
    def _add_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement levels"""
        # Calculate swing high/low for Fibonacci
        lookback = 50
        
        df['SWING_HIGH'] = df['high'].rolling(window=lookback*2+1, center=True).max()
        df['SWING_LOW'] = df['low'].rolling(window=lookback*2+1, center=True).min()
        
        # Fibonacci retracement levels from recent swing
        recent_high = df['high'].rolling(lookback).max()
        recent_low = df['low'].rolling(lookback).min()
        
        fib_range = recent_high - recent_low
        
        df['FIB_0_236'] = recent_low + 0.236 * fib_range
        df['FIB_0_382'] = recent_low + 0.382 * fib_range
        df['FIB_0_500'] = recent_low + 0.500 * fib_range
        df['FIB_0_618'] = recent_low + 0.618 * fib_range
        df['FIB_0_764'] = recent_low + 0.764 * fib_range
        
        return df

class WyckoffAnalysis:
    """Wyckoff method analysis for market structure"""
    
    def __init__(self):
        self.logger = get_logger("WyckoffAnalysis")
    
    def analyze_market_structure(self, df: pd.DataFrame, volume_threshold: float = 1.5) -> pd.DataFrame:
        """Analyze Wyckoff market structure phases"""
        result_df = df.copy()
        
        # Identify Wyckoff phases
        result_df['WYCKOFF_PHASE'] = self._identify_wyckoff_phases(df, volume_threshold)
        
        # Identify Springs and Upthrusts
        result_df['SPRING'] = self._identify_springs(df)
        result_df['UPTHRUST'] = self._identify_upthrusts(df)
        
        # Supply and Demand zones
        result_df = self._identify_supply_demand_zones(result_df)
        
        return result_df
    
    def _identify_wyckoff_phases(self, df: pd.DataFrame, volume_threshold: float) -> pd.Series:
        """Identify Wyckoff phases"""
        phases = pd.Series('None', index=df.index)
        
        # Accumulation (1-3): Low prices, low volume, consolidation
        if len(df) > 20:
            prices_near_low = df['low'].rolling(20).min() * 1.02 >= df['close']
            low_volatility = df['close'].rolling(20).std() / df['close'].rolling(20).mean() < 0.05
            
            phases[(prices_near_low) & (low_volatility)] = 'Accumulation'
        
        # Distribution (4-5): High prices, high volume, volatility increase
        if len(df) > 20:
            prices_near_high = df['close'] >= df['high'].rolling(20).max() * 0.98
            high_volume = df['volume'].rolling(5).mean() > df['volume'].rolling(20).mean() * volume_threshold
            
            phases[(prices_near_high) & (high_volume)] = 'Distribution'
        
        # Markup (uptrend) and Markdown (downtrend)
        if len(df) > 50:
            price_sma_20 = df['close'].rolling(20).mean()
            price_sma_50 = df['close'].rolling(50).mean()
            
            uptrend = (df['close'] > price_sma_20) & (price_sma_20 > price_sma_50)
            phases[uptrend & (phases == 'None')] = 'Markup'
            
            downtrend = (df['close'] < price_sma_20) & (price_sma_20 < price_sma_50)
            phases[downtrend & (phases == 'None')] = 'Markdown'
        
        return phases
    
    def _identify_springs(self, df: pd.DataFrame) -> pd.Series:
        """Identify Spring patterns (false breakdowns)"""
        springs = pd.Series(False, index=df.index)
        
        if len(df) < 10:
            return springs
        
        # Look for price breaking below support but quickly recovering
        for i in range(10, len(df)):
            # Previous 10 periods as support level
            support_level = df['low'].iloc[i-10:i].min()
            
            # Check if current low breaks support but closes above
            if (df['low'].iloc[i] < support_level * 0.995 and 
                df['close'].iloc[i] > support_level and
                df['close'].iloc[i] > df['close'].iloc[i-1]):
                springs.iloc[i] = True
        
        return springs
    
    def _identify_upthrusts(self, df: pd.DataFrame) -> pd.Series:
        """Identify Upthrust patterns (false breakouts)"""
        upthrusts = pd.Series(False, index=df.index)
        
        if len(df) < 10:
            return upthrusts
        
        # Look for price breaking above resistance but quickly falling back
        for i in range(10, len(df)):
            # Previous 10 periods as resistance level
            resistance_level = df['high'].iloc[i-10:i].max()
            
            # Check if current high breaks resistance but closes below
            if (df['high'].iloc[i] > resistance_level * 1.005 and 
                df['close'].iloc[i] < resistance_level and
                df['close'].iloc[i] < df['close'].iloc[i-1]):
                upthrusts.iloc[i] = True
        
        return upthrusts
    
    def _identify_supply_demand_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify supply and demand zones"""
        result = df.copy()
        
        # Supply zones (resistance areas)
        result['SUPPLY_ZONE'] = self._find_supply_zones(df)
        
        # Demand zones (support areas)
        result['DEMAND_ZONE'] = self._find_demand_zones(df)
        
        # Distance to nearest zones
        result['DISTANCE_TO_SUPPLY'] = self._calculate_distance_to_zones(df, result['SUPPLY_ZONE'])
        result['DISTANCE_TO_DEMAND'] = self._calculate_distance_to_zones(df, result['DEMAND_ZONE'])
        
        return result
    
    def _find_supply_zones(self, df: pd.DataFrame) -> pd.Series:
        """Find supply (resistance) zones"""
        zones = pd.Series(0, index=df.index)
        
        if len(df) < 20:
            return zones
        
        # Look for areas where price rejected significantly from highs
        recent_highs = df['high'].rolling(10, center=True).max()
        rejection_above_highs = (df['high'] >= recent_highs * 1.001) & (df['close'] < recent_highs * 0.998)
        
        zones[rejection_above_highs] = 1
        
        return zones
    
    def _find_demand_zones(self, df: pd.DataFrame) -> pd.Series:
        """Find demand (support) zones"""
        zones = pd.Series(0, index=df.index)
        
        if len(df) < 20:
            return zones
        
        # Look for areas where price bounced significantly from lows
        recent_lows = df['low'].rolling(10, center=True).min()
        bounce_above_lows = (df['low'] <= recent_lows * 0.999) & (df['close']: recent_lows * 1.002)
        
        zones[bounce_above_lows] = 1
        
        return zones
    
    def _calculate_distance_to_zones(self, df: pd.DataFrame, zones: pd.Series) -> pd.Series:
        """Calculate distance to nearest zones"""
        distances = pd.Series(np.nan, index=df.index)
        
        current_price = df['close']
        
        for i in range(len(df)):
            if zones.iloc[i] == 1:
                # Calculate distance from future prices
                if i > 0:
                    distances.iloc[i-1] = abs(current_price.iloc[i-1] - df['close'].iloc[i]) / df['close'].iloc[i]
        
        return distances.fillna(method='ffill').fillna(1.0)

class AdvancedFeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self):
        self.logger = get_logger("AdvancedFeatureEngineer")
        self.tech_indicators = TechnicalIndicators()
        self.technical_indicators = TechnicalIndicators()
        self.wyckoff_analysis = WyckoffAnalysis()
        self.scaler = RobustScaler()
    
    def engineer_features(self, df: pd.DataFrame, symbol: str, 
                         include_news: bool = True, 
                         include_market_structure: bool = True) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        
        if len(df) < 50:
            self.logger.warning(f"Insufficient data for feature engineering: {len(df)} rows")
            return df.copy()
        
        result_df = df.copy()
        
        # Technical indicators
        result_df = self.technical_indicators.calculate_all_indicators(result_df)
        
        # Market structure analysis
        if include_market_structure:
            result_df = self.wyckoff_analysis.analyze_market_structure(result_df)
        
        # Candlestick patterns
        result_df = self._add_candlestick_patterns(result_df)
        
        # Statistical features
        result_df = self._add_statistical_features(result_df)
        
        # Market regime detection
        result_df = self._detect_market_regime(result_df)
        
        # RSI divergence
        result_df = self._detect_rsi_divergence(result_df)
        
        # Price action patterns
        result_df = self._add_price_action_patterns(result_df)
        
        # Feature transformations
        result_df = self._apply_feature_transformations(result_df)
        
        # Feature interactions
        result_df = self._create_feature_interactions(result_df)
        
        self.logger.info(f"🎯 Engineered {len(result_df.columns)} features for {symbol}")
        
        return result_df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        result = df.copy()
        
        # Single candlestick patterns
        result['DOJI'] = abs((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) < 0.1
        result['HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']) / 100
        result['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']) / 100
        
        # Multi-candlestick patterns
        result['ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) / 100
        result['HARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']) / 100
        result['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']) / 100
        result['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']) / 100
        
        return result
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        result = df.copy()
        
        # Returns and volatility
        result['RETURN_1'] = df['close'].pct_change(1)
        result['RETURN_5'] = df['close'].pct_change(5)
        result['RETURN_10'] = df['close'].pct_change(10)
        
        # Volatility measures
        result['VOLATILITY_5'] = df['close'].rolling(5).std()
        result['VOLATILITY_20'] = df['close'].rolling(20).std()
        
        # Skewness and Kurtosis
        result['SKEWNESS_20'] = df['close'].rolling(20).apply(lambda x: skew(x.dropna()), raw=False)
        result['KURTOSIS_20'] = df['close'].rolling(20).apply(lambda x: kurtosis(x.dropna()), raw=False)
        
        # Price position within recent range
        result['PRICE_POSITION_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        # Volume analysis
        result['VOLUME_RATIO'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return result
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime (trending vs ranging)"""
        result = df.copy()
        
        if len(df) < 50:
            result['MARKET_REGIME'] = 'Sideways'
            return result
        
        # Calculate trend strength using ADX
        adx_threshold = 25
        
        # Simple range detection using price action
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        range_size = high_20 - low_20
        
        # Market regime classification
        conditions = [
            (df['ADX'].fillna(20) > adx_threshold),
            (df['ADX'].fillna(20) <= adx_threshold)
        ]
        
        choices = ['Trending', 'Sideways']
        result['MARKET_REGIME'] = np.select(conditions, choices, default='Sideways')
        
        # Regime strength (0-1)
        result['TREND_STRENGTH'] = np.clip(df['ADX'] / 50, 0, 1)
        
        return result
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect RSI divergence patterns"""
        result = df.copy()
        
        if len(df) < 20:
            result['RSI_DIVERGENCE'] = 'None'
            return result
        
        # Look for divergence over last 10 periods
        lookback = 10
        price_slope = df['close'].rolling(lookback).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
        rsi_slope = df['RSI_14'].rolling(lookback).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
        
        # Bullish divergence: price falling, RSI rising
        bullish_div = (price_slope < 0) & (rsi_slope > 0) & (df['RSI_14'] < 50)
        
        # Bearish divergence: price rising, RSI falling
        bearish_div = (price_slope > 0) & (rsi_slope < 0) & (df['RSI_14'] > 50)
        
        result['RSI_DIVERGENCE'] = np.where(bearish_div, 'Bearish', np.where(bullish_div, 'Bullish', 'None'))
        
        return result
    
    def _add_price_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action pattern recognition"""
        result = df.copy()
        
        # Gap detection
        result['GAP_UP'] = df['open'] > df['high'].shift(1) * 1.005
        result['GAP_DOWN'] = df['open'] < df['low'].shift(1) * 0.995
        
        # Breakout pattern
        result['BREAKOUT_HIGH'] = df['close'] > df['high'].rolling(20).max().shift(1)
        result['BREAKOUT_LOW'] = df['close'] < df['low'].rolling(20).min().shift(1)
        
        # Inside Bar
        result['INSIDE_BAR'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
        
        # Outside Bar (Engulfing)
        result['OUTSIDE_BAR'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        
        return result
    
    def _apply_feature_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations"""
        result = df.copy()
        
        # Log transformations for volatile features
        volume_cols = ['volume', 'OBV', 'AD']
        for col in volume_cols:
            if col in result.columns:
                result[f'{col}_LOG'] = np.log1p(result[col])
        
        # Square root transformations for moderately skewed features
        price_cols = ['ATR', 'VOLATILITY_5', 'VOLATILITY_20']
        for col in price_cols:
            if col in result.columns:
                result[f'{col}_SQRT'] = np.sqrt(result[col])
        
        # Lag features for temporal relationships
        lag_cols = ['RSI_14', 'MACD', 'volume']
        for col in lag_cols:
            if col in result.columns:
                for lag in [1, 2, 3]:
                    result[f'{col}_LAG_{lag}'] = result[col].shift(lag)
        
        return result
    
    def _create_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature interaction terms"""
        result = df.copy()
        
        # RSI and MACD interaction
        if 'RSI_14' in df.columns and 'MACD' in df.columns:
            result['RSI_MACD_INTERACTION'] = df['RSI_14'] * df['MACD']
        
        # Price and volume interaction
        if 'volume' in df.columns:
            result['PRICE_VOLUME_INTERACTION'] = df['close'] * df['volume']
        
        # Volatility and momentum interaction
        if 'VOLATILITY_20' in df.columns and 'ROC' in df.columns:
            result['VOLATILITY_MOMENTUM'] = df['VOLATILITY_20'] * abs(df['ROC'])
        
        return result
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            else:
                self.logger.warning("Model does not support feature importance")
                return {}
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {e}")
            return {}
    
    def select_features(self, df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """Select most relevant features for training"""
        
        # Remove columns with too many NaN values
        nan_threshold = 0.5
        valid_cols = df.columns[df.isnull().mean() < nan_threshold]
        
        # Remove constant columns
        constant_cols = df[valid_cols].columns[df[valid_cols].nunique() <= 1]
        valid_cols = [col for col in valid_cols if col not in constant_cols]
        
        # Focus on feature columns (exclude basic OHLCV)
        feature_cols = [col for col in valid_cols if not col in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        # Select subset if too many features
        if len(feature_cols) > max_features:
            # Use correlation to select diverse features
            correlation_matrix = df[feature_cols].corr().abs()
            selected_features = []
            
            for feature in feature_cols:
                if all(correlation_matrix.loc[feature, selected_features].max() < 0.95 for _ in [None]):
                    selected_features.append(feature)
                    if len(selected_features) >= max_features:
                        break
            
            feature_cols = selected_features
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume'] + feature_cols]

# Global feature engineer instance
feature_engineer = AdvancedFeatureEngineer()