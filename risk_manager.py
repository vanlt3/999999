"""
Advanced Multi-layered Risk Management System with Intelligent Stop Loss and Portfolio Protection
Comprehensive risk management with correlation checks, position sizing, and automated protection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import json

from logger import get_logger
from config import trading_config, SECURITY_CONFIG, ASSET_TYPES

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    side: str  # "BUY" or "SELL"
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    entry_reason: str = ""
    pnl: float = 0.0
    risk_amount: float = 0.0
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        
        self.pnl = self._calculate_pnl()

    def _calculate_pnl(self) -> float:
        """Calculate unrealized PnL"""
        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size
    
    def update_price(self, new_price: float):
        """Update current price and PnL"""
        self.current_price = new_price
        self.pnl = self._calculate_pnl()

@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_value: float
    cash: float
    total_exposure: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    positions: Dict[str, Position]
    max_drawdown: float
    correlation_score: float
    risk_score: float
    margin_used: float = 0.0

class CorrelationManager:
    """Manage correlation between positions and prevent over-concentration"""
    
    def __init__(self):
        self.logger = get_logger("CorrelationManager")
        self.correlation_threshold = trading_config.correlation_threshold
        self.correlation_data = {}
        self.correlation_patterns = {
            # Forex correlations
            'EURUSD-GBPUSD': 0.7,
            'EURUSD-USDCHF': -0.9,
            'AUDUSD-NZDUSD': 0.8,
            'GBPJPY-EURJPY': 0.9,
            
            # Commodity correlations
            'XAUUSD-XAGUSD': 0.8,
            'XAUUSD-USD': -0.9,
            'XAUUSD-URAUSD': -0.7,
            
            # Crypto correlations
            'BTCUSD-ETHUSD': 0.8,
            
            # Index correlations
            'NASDAQ100-SPY': 0.8,
        }
    
    def calculate_correlation(self, symbol1: str, symbol2: str, 
                           price_data: pd.DataFrame) -> float:
        """Calculate correlation between two symbols"""
        try:
            # If both symbols are in the data
            if symbol1 in price_data.columns and symbol2 in price_data.columns:
                returns1 = price_data[symbol1].pct_change().dropna()
                returns2 = price_data[symbol2].pct_change().dropna()
                
                # Align the returns
                returns_data = pd.DataFrame({
                    symbol1: returns1,
                    symbol2: returns2
                }).dropna()
                
                if len(returns_data) > 20:
                    correlation = returns_data[symbol1].corr(returns_data[symbol2])
                    return correlation
            
            # Use static patterns if data unavailable
            pattern_key = f"{symbol1}-{symbol2}"
            reverse_key = f"{symbol2}-{symbol1}"
            
            return self.correlation_patterns.get(pattern_key, 
                                                self.correlation_patterns.get(reverse_key, 0))
            
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed {symbol1}-{symbol2}: {e}")
            return 0
    
    def check_correlation_risk(self, new_symbol: str, 
                             existing_positions: Dict[str, Position]) -> Tuple[bool, List[str]]:
        """Check if new position violates correlation constraints"""
        
        high_correlation_positions = []
        
        for existing_symbol, position in existing_positions.items():
            correlation = self.calculate_correlation(new_symbol, existing_symbol, None)
            
            if correlation >= self.correlation_threshold:
                high_correlation_positions.append(existing_symbol)
        
        violates_risk = len(high_correlation_positions) >= 2  # Max 2 highly correlated positions
        
        return violates_risk, high_correlation_positions

class PositionSizeCalculator:
    """Dynamic position sizing based on confidence and risk"""
    
    def __init__(self):
        self.logger = get_logger("PositionSizeCalculator")
        self.base_size = trading_config.max_position_size
    
    def calculate_size(self, symbol: str, confidence: float, risk_score: float,
                      account_value: float, atr: float, current_price: float,
                      risk_per_trade: float = 0.02) -> Tuple[float, float]:
        """Calculate optimal position size"""
        
        # Account percentage method
        account_percentage = self.base_size * confidence
        account_size = account_value * account_percentage
        
        # Risk-based method (fixed risk amount)
        risk_amount = account_value * risk_per_trade
        stop_distance = atr * 2  # 2x ATR stop
        
        if stop_distance > 0:
            risk_size = risk_amount / stop_distance
        else:
            risk_size = account_size / current_price
        
        # Kelly criterion (simplified)
        win_rate = confidence
        avg_win_avg_loss = 1.5  # Assume 1.5:1 risk-reward
        kelly_percentage = (win_rate * avg_win_avg_loss - (1 - win_rate)) / avg_win_avg_loss
        kelly_percentage = max(0, min(0.25, kelly_percentage))  # Cap at 25%
        
        kelly_size = account_value * kelly_percentage / current_price
        
        # Choose the most conservative size
        sizes = [
            account_size / current_price,
            risk_size,
            kelly_size
        ]
        
        optimal_size = min(sizes)
        
        # Apply risk score adjustment
        risk_adjustment = max(0.5, 1.0 - risk_score)
        final_size = optimal_size * risk_adjustment
        
        # Ensure size is within limits
        final_size = min(final_size, 
                        account_value * trading_config.max_position_size / current_price)
        
        # Calculate actual risk amount
        actual_risk = stop_distance * final_size
        
        self.logger.debug(f"💼 Position size for {symbol}: "
                         f"{final_size:.2f} units, Risk: ${actual_risk:.2f}")
        
        return final_size, actual_risk

class StopLossManager:
    """Intelligent stop loss management with Master Agent input"""
    
    def __init__(self):
        self.logger = get_logger("StopLossManager")
        self.default_sl_multiplier = trading_config.default_sl_multiplier
        self.default_tp_multiplier = trading_config.default_tp_multiplier
    
    def calculate_stop_levels(self, symbol: str, side: str, entry_price: float,
                            atr: float, market_regime: str, confidence: float,
                            support_resistance_levels: Dict[str, float] = None) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels using Master Agent logic"""
        
        # Base SL/TP based on ATR
        base_sl_distance = atr * self.default_sl_multiplier
        base_tp_distance = atr * self.default_tp_multiplier
        
        # Adjust based on market régime
        régime_multipliers = {
            "trending": {"sl": 1.2, "tp": 1.5},     # Wider SL/TP in trends
            "sideways": {"sl": 0.8, "tp": 1.0},     # Tighter SL/TP in ranges
            "volatile": {"sl": 1.5, "tp": 2.0}      # Much wider in volatile markets
        }
        
        multiplier = régime_multipliers.get(market_regime, {"sl": 1.0, "tp": 1.0})
        
        sl_distance = base_sl_distance * multiplier["sl"]
        tp_distance = base_tp_distance * multiplier["tp"]
        
        # Adjust based on confidence
        confidence_multiplier = 0.8 if confidence > 0.8 else 1.2  # Tighter SL at high confidence
        sl_distance *= confidence_multiplier
        tp_distance *= (1 / confidence_multiplier)  # Wider TP at high confidence
        
        # S/R levels override
        if support_resistance_levels:
            support = support_resistance_levels.get("support")
            resistance = support_resistance_levels.get("resistance")
            
            if side == "BUY" and support and support < entry_price:
                sl_level = support * 0.995  # Just below support
                tp_level = resistance * 0.995 if resistance else entry_price + tp_distance
            elif side == "SELL" and resistance and resistance > entry_price:
                sl_level = resistance * 1.005  # Just above resistance
                tp_level = support * 1.005 if support else entry_price - tp_distance
            else:
                if side == "BUY":
                    sl_level = entry_price - sl_distance
                    tp_level = entry_price + tp_distance
                else:
                    sl_level = entry_price + sl_distance
                    tp_level = entry_price - tp_distance
        else:
            if side == "BUY":
                sl_level = entry_price - sl_distance
                tp_level = entry_price + tp_distance
            else:
                sl_level = entry_price + sl_distance
                tp_level = entry_price - tp_distance
        
        self.logger.debug(f"🔒 Stop levels for {symbol}: "
                         f"SL=${sl_level:.2f}, TP=${tp_level:.2f}")
        
        return sl_level, tp_level
    
    def check_stop_loss_hit(self, position: Position, current_price: float,
                          high_price: float, low_price: float, volume: float) -> Tuple[bool, bool]:
        """Check if stop loss should be triggered"""
        
        triggered = False
        false_break = False
        
        if position.side == "BUY":
            # Check if price went below stop loss
            if low_price <= position.stop_loss:
                # Check for false break (wick detection)
                if closes >= position.ср и volume > 1.5 * average_volume:
                    false_break = True
                else:
                    triggered = True
        
        else:  # SELL position
            if high_price >= position.stop_loss:
                if closes >= position.ср и volume > 1.5 * average_volume:
                    false_break = True
                else:
                    triggered = True
        
        return triggered, false_break
    
    def update_trailing_stop(self, position: Position, current_price: float,
                           atr: float, master_agent_decision: str = None) -> Optional[float]:
        """Update trailing stop using Master Agent decision"""
        
        if position.trailing_stop is None:
            # Initialize trailing stop when profitable
            profit_threshold = trading_config.trailing_stop_min_profit
            current_profit = position.pnl / (position.entry_price * position.size)
            
            if current_profit >= profit_threshold:
                trail_distance = atr * trading_config.trailing_stop_atr_multiplier
                
                if position.side == "BUY":
                    new_trailing_stop = current_price - trail_distance
                else:
                    new_trailing_stop = current_price + trail_distance
                
                position.trailing_stop = new_trailing_stop
                
                self.logger.info(f"🎯 Trailing stop activated for {position.symbol}: ${new_trailing_stop:.2f}")
                
                return new_trailing_stop
        
        else:
            # Update existing trailing stop
            trail_distance = atr * trading_config.trailing_stop_atr_multiplier
            
            if position.side == "BUY":
                new_trailing_stop = current_price - trail_distance
                if new_trailing_stop > position.trailing_stop:
                    old_stop = position.trailing_stop
                    position.trailing_stop = new_trailing_stop
                    
                    self.logger.info(f"📈 Trailing stop updated for {position.symbol}: "
                                    f"${old_stop:.2f} → ${new_trailing_stop:.2f}")
                    
                    return new_trailing_stop
            
            else:
                new_trailing_stop = current_price + trail_distance
                if new_trailing_stop < position.trailing_stop:
                    old_stop = position.trailing_stop
                    position.trailing_stop = new_trailing_stop
                    
                    self.logger.info(f"📉 Trailing stop updated for {position.symbol}: "
                                    f"${old_stop:.2f} → ${new_trailing_stop:.2f}")
                    
                    return new_trailing_stop
        
        return None

class TradeGating:
    """Trade gating system to prevent trading during adverse conditions"""
    
    def __init__(self):
        self.logger = get_logger("TradeGating")
        self.news_gating_hours = SECURITY_CONFIG.news_gating_hours
        self.major_events = SECURITY_CONFIG.news_gating_events
        
        # Track economic events
        self.economic_calendar = {}
        
    def check_news_gates(self, symbol: str, current_time: datetime = None) -> Tuple[bool, str]:
        """Check if trading should be paused due to major news events"""
        
        if current_time is None:
            current_time = datetime.now()
        
        # Check for upcoming economic events
        for event_name in self.major_events:
            upcoming_events = self._get_upcoming_events(event_name, current_time, hours_ahead=24)
            
            for event in upcoming_events:
                event_time = event['time']
                time_diff = abs((current_time - event_time).total_seconds() / 3600)
                
                if time_diff <= self.news_gating_hours:
                    reason = f"Major event: {event_name} at {event_time.strftime('%H:%M')}"
                    self.logger.warning(f"🚫 Trade gate activated: {reason}")
                    return False, reason
        
        return True, "Clear to trade"
    
    def check_weekend_trading(self, symbol: str) -> Tuple[bool, str]:
        """Check if weekend trading is allowed for symbol"""
        
        asset_type = ASSET_TYPES.get(symbol, "forex")
        
        if asset_type == "crypto":
            return True, "Crypto trading allowed"
        
        now = datetime.now()
        
        # Weekend trading rules
        if now.weekday() >= 5:  # Saturday or Sunday
            if SECURITY_CONFIG.allow_weekend_trading_crypto and asset_type == "crypto":
                return True, "Weekend crypto trading"
            else:
                return False, "Weekend trading not allowed for this asset type"
        
        return True, "Regular trading hours"
    
    def check_market_hours(self, symbol: str) -> Tuple[bool, str]:
        """Check if market is open for symbol"""
        
        asset_type = ASSET_TYPES.get(symbol, "forex")
        
        # Cryptocurrencies trade 24/7
        if asset_type == "crypto":
            return True, "Crypto markets open"
        
        # Forex trades 24/5 (except weekend)
        if asset_type == "forex":
            now = datetime.now()
            if now.weekday() >= 5:  # Weekend
                return False, "Forex markets closed weekends"
            else:
                return True, "Forex markets open"
        
        # For other assets, check specific market hours
        return True, "Market hours check passed"
    
    def _get_upcoming_events(self, event_name: str, current_time: datetime, 
                           hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming economic events"""
        
        # This would integrate with an economic calendar API
        # For now, return empty list
        return []
    
    def get_all_restrictions(self, symbol: str) -> List[str]:
        """Get all current trade restrictions"""
        
        restrictions = []
        
        # Check news gates
        news_ok, news_reason = self.check_news_gates(symbol)
        if not news_ok:
            restrictions.append(news_reason)
        
        # Check weekend trading
        weekend_ok, weekend_reason = self.check_weekend_trading(symbol)
        if not weekend_ok:
            restrictions.append(weekend_reason)
        
        # Check market hours
        market_ok, market_reason = self.check_market_hours(symbol)
        if not market_ok:
            restrictions.append(market_reason)
        
        return restrictions

class RiskManager:
    """Main risk management orchestrator"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.logger = get_logger("RiskManager")
        
        # Component managers
        self.correlation_manager = CorrelationManager()
        self.position_calculator = PositionSizeCalculator()
        self.stop_loss_manager = StopLossManager()
        self.trade_gating = TradeGating()
        
        # Portfolio state
        self.portfolio_state = PortfolioState(
            total_value=initial_capital,
            cash=initial_capital,
            total_exposure=0.0,
            total_pnl=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            positions={},
            max_drawdown=0.0,
            correlation_score=0.0,
            risk_score=0.0
        )
        
        # Risk monitoring
        self.daily_pnl = []
        self.drawdown_history = []
        
        # Database connection
        self.db_conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize risk management database"""
        self.db_conn = sqlite3.connect('risk_management.db')
        self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                total_exposure REAL NOT NULL,
                total_pnl REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                risk_score REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.db_conn.commit()
    
    def validate_trade(self, symbol: str, side: str, confidence: float, 
                      market_data: Dict[str, Any], master_decision: Any) -> Tuple[bool, str]:
        """Validate trade against all risk criteria"""
        
        self.logger.info(f"🔍 Validating trade: {symbol} {side}")
        
        # 1. Trade gating checks
        restrictions = self.trade_gating.get_all_restrictions(symbol)
        if restrictions:
            reason = f"Trade restrictions: {restrictions[0]}"
            self.logger.warning(f"❌ Trade validation failed: {reason}")
            return False, reason
        
        # 2. Correlation risk check
        violates_correlation, correlated_positions = self.correlation_manager.check_correlation_risk(
            symbol, self.portfolio_state.positions
        )
        
        if violates_correlation:
            reason = f"Too many correlated positions: {correlated_positions}")
            self.logger.warning(f"❌ Trade validation failed: {reason}")
            return False, reason
        
        # 3. Portfolio exposure check
        current_exposure = self.portfolio_state.total_exposure / self.portfolio_state.total_value
        if current_exposure >= trading_config.max_total_exposure:
            reason = f"Maximum exposure reached: {current_exposure:.1%}"
            self.logger.warning(f"❌ Trade validation failed: {reason}")
            return False, reason
        
        # 4. Minimum confidence check
        if confidence < 0.5:
            reason = f"Low confidence signal: {confidence:.2f}"
            self.logger.warning(f"❌ Trade validation failed: {reason}")
            return False, reason
        
        # 5. Drawdown protection
        if self.portfolio_state.max_drawdown > 0.15:  # 15% max drawdown
            reason = f"Maximum drawdown exceeded: {self.portfolio_state.max_drawdown:.1%}"
            self.logger.warning(f"❌ Trade validation failed: {reason}")
            return False, reason
        
        self.logger.info(f"✅ Trade validation passed for {symbol}")
        return True, "All risk checks passed"
    
    def calculate_position_parameters(self, symbol: str, side: str, entry_price: float,
                                      confidence: float, market_data: Dict[str, Any],
                                      master_decision: Any) -> Dict[str, Any]:
        """Calculate all position parameters"""
        
        # Calculate position size
        atr = market_data.get('ATR', entry_price * 0.02)  # Default 2% volatility
        
        position_size, risk_amount = self.position_calculator.calculate_size(
            symbol=symbol,
            confidence=confidence,
            risk_score=master_decision.risk_score,
            account_value=self.portfolio_state.total_value,
            atr=atr,
            current_price=entry_price
        )
        
        # Calculate stop levels
        stop_loss, take_profit = self.stop_loss_manager.calculate_stop_levels(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            atr=atr,
            market_regime=master_decision.market_regime,
            confidence=confidence
        )
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'atr': atr
        }
    
    def open_position(self, symbol: str, side: str, size: float, entry_price: float,
                     stop_loss: float, take_profit: float, reason: str) -> Position:
        """Open new position with risk tracking"""
        
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=reason
        )
        
        # Update portfolio state
        self.portfolio_state.positions[symbol] = position
        self.portfolio_state.cash -= size * entry_price
        self.portfolio_state.total_exposure += size * entry_price
        
        self.logger.info(f"📈 Opened position: {symbol} {side} "
                        f"Size: {size:.2f} @ ${entry_price:.2f}")
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> Dict[str, Any]:
        """Close position and update portfolio"""
        
        if symbol not in self.portfolio_state.positions:
            return {"success": False, "reason": "Position not found"}
        
        position = self.portfolio_state.positions[symbol]
        position.exit_time = datetime.now()
        position.update_price(exit_price)
        
        # Calculate final PnL
        pnl = position.pnl
        
        # Update portfolio state
        self.portfolio_state.cash += position.size * exit_price
        self.portfolio_state.total_exposure -= position.size * position.entry_price
        self.portfolio_state.realized_pnl += pnl
        
        # Remove position
        del self.portfolio_state.positions[symbol]
        
        self.logger.info(f"📉 Closed position: {symbol} "
                        f"PnL: ${pnl:.2f} @ ${exit_price:.2f} - {reason}")
        
        return {
            "success": True,
            "pnl": pnl,
            "position": position
        }
    
    def update_portfolio(self, current_prices: Dict[str, float]):
        """Update portfolio with current prices"""
        
        total_value = self.portfolio_state.cash
        unrealized_pnl = 0
        
        for symbol, position in self.portfolio_state.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
                position_value = position.size * current_prices[symbol]
                total_value += position_value
                unrealized_pnl += position.pnl
        
        # Update total portfolio metrics
        old_total_value = self.portfolio_state.total_value
        self.portfolio_state.total_value = total_value
        self.portfolio_state.unrealized_pnl = unrealized_pnl
        
        # Update total PnL
        total_pnl = self.portfolio_state.realized_pnl + unrealized_pnl
        self.portfolio_state.total_pnl = total_pnl
        
        # Calculate drawdown
        peak_value = max(self.initial_capital, *self.drawdown_history) if self.drawdown_history else self.initial_capital
        current_drawdown = max(0, (peak_value - total_value) / peak_value)
        self.portfolio_state.max_drawdown = max(self.portfolio_state.max_drawdown, current_drawdown)
        
        self.drawdown_history.append(total_value)
        if len(self.drawdown_history) > 100:
            self.drawdown_history.pop(0)
        
        # Calculate correlation score
        self.portfolio_state.correlation_score = self._calculate_portfolio_correlation()
        
        # Calculate overall risk score
        self.portfolio_state.risk_score = self._calculate_portfolio_risk()
        
        # Log update
        if old_total_value != total_value:
            change = total_value - old_total_value
            self.logger.debug(f"📊 Portfolio updated: "
                            f"Value: ${total_value:.2f} ({change:+.2f}), "
                            f"PnL: ${total_pnl:+.2f}")
    
    def _calculate_portfolio_correlation(self) -> float:
        """Calculate portfolio correlation score"""
        if len(self.portfolio_state.positions) < 2:
            return 0.0
        
        symbols = list(self.portfolio_state.positions.keys())
        total_correlation = 0
        pairs_count = 0
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                correlation = self.correlation_manager.calculate_correlation(
                    symbols[i], symbols[j], None
                )
                total_correlation += abs(correlation)
                pairs_count += 1
        
        return total_correlation / pairs_count if pairs_count > 0 else 0.0
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate overall portfolio risk score"""
        risk_score = 0
        
        # Drawdown risk
        risk_score += self.portfolio_state.max_drawdown * 0.3
        
        # Exposure risk
        exposure_ratio = self.portfolio_state.total_exposure / self.portfolio_state.total_value
        risk_score += min(exposure_ratio / trading_config.max_total_exposure, 1.0) * 0.2
        
        # Concentration risk (number of positions)
        position_count = len(self.portfolio_state.positions)
        if position_count > 5:
            risk_score += 0.1 * (position_count - 5) / 5
        
        # Correlation risk
        risk_score += self.portfolio_state.correlation_score * 0.2
        
        # PnL volatility risk (simplified)
        if len(self.daily_pnl) > 5:
            pnl_volatility = np.std(self.daily_pnl[-10:])
            risk_score += min(pnl_volatility / (self.portfolio_state.total_value * 0.01), 1.0) * 0.3
        
        return min(1.0, risk_score)
    
    def should_close_all_positions(self) -> Tuple[bool, str]:
        """Check if all positions should be closed (emergency stop)"""
        
        # Emergency conditions
        if self.portfolio_state.max_drawdown >= 0.20:  # 20% drawdown
            return True, f"Emergency stop: Drawdown {self.portfolio_state.max_drawdown:.1%}"
        
        if self.portfolio_state.total_value <= self.initial_capital * 0.80:  # 20% loss
            return True, f"Emergency stop: Portfolio down 20%"
        
        if len(self.daily_pnl) >= 10 and np.mean(self.daily_pnl[-10:]) < -0.05:  # Poor recent performance
            return True, "Emergency stop: Poor performance streak"
        
        return False, "Normal risk levels"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        return {
            'portfolio_value': self.portfolio_state.total_value,
            'cash': self.portfolio_state.cash,
            'total_exposure': self.portfolio_state.total_exposure,
            'total_pnl': self.portfolio_state.total_pnl,
            'unrealized_pnl': self.portfolio_state.unrealized_pnl,
            'realized_pnl': self.portfolio_state.realized_pnl,
            'max_drawdown': self.portfolio_state.max_drawdown,
            'correlation_score': self.portfolio_state.correlation_score,
            'risk_score': self.portfolio_state.risk_score,
            'position_count': len(self.portfolio_state.positions),
            'positions': {
                symbol: {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for symbol, pos in self.portfolio_state.positions.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def save_risk_metrics(self):
        """Save current risk metrics to database"""
        cursor = self.db_conn.execute(
            "INSERT INTO risk_metrics (timestamp, total_value, cash, total_exposure, total_pnl, max_drawdown, risk_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(), self.portfolio_state.total_value, self.portfolio_state.cash,
             self.portfolio_state.total_exposure, self.portfolio_state.total_pnl,
             self.portfolio_state.max_drawdown, self.portfolio_state.risk_score)
        )
        self.db_conn.commit()

# Global risk manager instance
risk_manager = RiskManager()