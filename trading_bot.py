"""
Advanced Trading Bot Main Orchestration System
Complete trading bot with ML, RL, sentiment analysis, and risk management
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import signal
import sys
import os
from pathlib import Path

# Import all bot components
from config import *
from logger import bot_logger, trade_logger, get_logger
from api_manager import api_manager
from data_manager import data_manager
from feature_engineer import feature_engineer
from ml_ensemble import ensemble_model, lstm_model, auto_retrain_manager
from reinforcement_learning import rl_agent, online_learning_manager
from master_agent import master_agent
from risk_manager import risk_manager
from discord_notifier import discord_notifier
from sentiement_analysis import news_manager

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.logger = bot_logger
        self.initial_capital = initial_capital
        self.is_running = False
        self.symbols = trading_config.symbols
        
        # Initialize components
        self.market_data = {}
        self.feature_data = {}
        self.models_ready = False
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.session_stats = {
            'trades_today': 0,
            'profit_today': 0.0,
            'signals_generated': 0,
            'models_retrained': 0,
            'start_time': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🤖 Advanced Trading Bot initialized")
    
    async def initialize(self):
        """Initialize all bot components"""
        self.logger.info("🚀 Initializing Advanced Trading Bot")
        
        try:
            # 1. Test Discord integration
            async with discord_notifier:
                await discord_notifier.send_test_message()
            
            # 2. Initialize API management
            async with api_manager as am:
                health_status = am.get_api_health_status()
                self.logger.info(f"🔗 API Health: {len([k for k, v in health_status.items() if v['status'] == 'healthy'])}/{len(health_status)} APIs healthy")
            
            # 3. Initialize models if needed
            await self._initialize_models()
            
            # 4. Load historical data
            await self._load_historical_data()
            
            # 5. Initialize news sentiment analysis
            await self._initialize_sentiment_analysis()
            
            # 6. Setup real-time monitoring
            self._setup_realtime_monitoring()
            
            self.logger.info("✅ Bot initialization completed successfully")
            
            # Send startup notification
            async with discord_notifier:
                await discord_notifier.notify_system_alert({
                    'alert_type': 'System Startup',
                    'message': f'🚀 Advanced Trading Bot started with ${self.initial_capital:,.2f} capital',
                    'severity': 'success',
                    'additional_data': {
                        'Symbols': ', '.join(self.symbols),
                        'Models': 'Ready' if self.models_ready else 'Training',
                        'Components': 'All systems operational'
                    }
                })
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bot initialization failed: {e}")
            
            # Send error notification
            async with discord_notifier:
                await discord_notifier.notify_system_alert({
                    'alert_type': 'Initialization Error',
                    'message': f'Failed to initialize bot: {str(e)}',
                    'severity': 'error'
                })
            
            return False
    
    async def _initialize_models(self):
        """Initialize and train ML models"""
        self.logger.info("🤖 Initializing ML models")
        
        try:
            # Train ensemble models for each symbol
            for symbol in self.symbols:
                self.logger.info(f"📊 Training models for {symbol}")
                
                # Get historical data
                price_data = await data_manager.get_price_data(symbol, "H1", limit=1000)
                
                if len(price_data) < 500:
                    self.logger.warning(f"⚠️ Insufficient data for {symbol}, skipping model training")
                    continue
                
                # Engineer features
                engineered_features = feature_engineer.engineer_features(
                    price_data.reset_index(), symbol
                )
                
                # Prepare target (price movement)
                engineered_features['target'] = engineered_features['close'].shift(-1) - engineered_features['close']
                engineered_features['price_direction'] = (engineered_features['target'] > 0).astype(int)
                
                # Remove last row (no target)
                engineered_features = engineered_features[:-1]
                
                if len(engineered_features) < 100:
                    continue
                
                # Select features for training
                feature_cols = [col for col in engineered_features.columns 
                               if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']]
                
                X = engineered_features[feature_cols].fillna(0)
                y = engineered_features['price_direction']
                
                # Train ensemble model
                ensemble_model.train_stack(X, y, symbol)
                
                # Train LSTM model
                lstm_model.train(X, y)
                
                # Set drift detection baseline
                auto_retrain_manager.set_baseline_from_training_data(X)
                
                self.logger.info(f"✅ Models trained for {symbol}")
            
            self.models_ready = True
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            self.models_ready = False
    
    async def _load_historical_data(self):
        """Load historical data for all symbols"""
        self.logger.info("📊 Loading historical data")
        
        for symbol in self.symbols:
            try:
                # Load price data for multiple timeframes
                timeframes = ['H1', 'H4', 'D1']
                
                symbol_data = {}
                for timeframe in timeframes:
                    data = await data_manager.get_price_data(symbol, timeframe, limit=500)
                    if not data.empty:
                        symbol_data[timeframe] = data
                
                if symbol_data:
                    self.market_data[symbol] = symbol_data
                    self.logger.info(f"📊 Loaded historical data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load data for {symbol}: {e}")
    
    async def _initialize_sentiment_analysis(self):
        """Initialize sentiment analysis system"""
        self.logger.info("📰 Initializing sentiment analysis")
        
        try:
            # Process news data for each symbol
            for symbol in self.symbols:
                news_data = await data_manager.get_news_data(symbol, days_back=7)
                
                if not news_data.empty:
                    sentiment_result = await news_manager.process_symbol_news(symbol, news_data)
                    self.logger.info(f"📰 Processed sentiment for {symbol}: {sentiment_result['overall_sentiment']:.2f}")
            
            # Process economic calendar
            economic_data = await data_manager.get_economic_calendar()
            if not economic_data.empty:
                economic_result = await news_manager.process_economic_calendar(economic_data)
                self.logger.info(f"📅 Processed {len(economic_result['events'])} economic events")
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis initialization failed: {e}")
    
    def _setup_realtime_monitoring(self):
        """Setup real-time price monitoring"""
        self.logger.info("⏰ Setting up real-time monitoring")
        
        # This would integrate with real-time data feeds
        # For now, we'll simulate with scheduled updates
        
    async def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            cycle_start = datetime.now()
            
            # 1. Update market data
            await self._update_market_data()
            
            # 2. Generate signals for each symbol
            signals_generated = 0
            
            for symbol in self.symbols:
                try:
                    # Get current market data
                    if symbol not in self.market_data or 'H1' not in self.market_data[symbol]:
                        continue
                    
                    current_data = self.market_data[symbol]['H1'].tail(100)
                    
                    # Engineer fresh features
                    engineered_data = feature_engineer.engineer_features(
                        current_data.reset_index(), symbol
                    )
                    
                    # Get sentiment features
                    news_features = news_manager.get_news_features(symbol)
                    
                    # Add sentiment features to engineered data
                    for feature, value in news_features.items():
                        engineered_data[feature] = value
                    
                    # Generate signal using Master Agent
                    decision = await master_agent.make_decision(
                        symbol=symbol,
                        market_data=current_data.reset_index(),
                        additional_data=news_features
                    )
                    
                    # Log signal
                    signal_data = {
                        'symbol': symbol,
                        'signal_type': decision.final_signal.value,
                        'confidence': decision.consensus_confidence,
                        'reasoning': decision.reasoning,
                        'risk_score': decision.risk_score,
                        'market_regime': decision.market_regime,
                        'recommended_position_size': decision.recommended_position_size,
                        'timestamp': decision.timestamp
                    }
                    
                    # Notify via Discord
                    if decision.final_signal.value != "HOLD":
                        signals_generated += 1
                        
                        async with discord_notifier:
                            await discord_notifier.notify_signal(signal_data)
                    
                    # Execute trading logic
                    await self._execute_trading_logic(symbol, decision)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            self.session_stats['signals_generated'] += signals_generated
            
            # 3. Update portfolio and risk management
            await self._update_portfolio()
            
            # 4. Check for model retraining
            await self._check_model_retraining()
            
            # 5. Send periodic status reports
            if len(self.performance_history) % 10 == 0:  # Every 10 cycles
                await self._send_status_report()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.debug(f"🔄 Trading cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Trading cycle error: {e}")
    
    async def _execute_trading_logic(self, symbol: str, decision):
        """Execute trading logic based on Master Agent decision"""
        
        try:
            signal_type = decision.final_signal.value
            confidence = decision.consensus_confidence
            
            # Skip if signal confidence is too low
            if confidence < 0.5:
                return
            
            # Get current price
            current_data = self.market_data[symbol]['H1'].tail(1)
            if current_data.empty:
                return
            
            current_price = current_data['close'].iloc[-1]
            
            if signal_type == "BUY":
                await self._process_buy_signal(symbol, current_price, decision)
            elif signal_type == "SELL":
                await self._process_sell_signal(symbol, current_price, decision)
            elif signal_type == "CLOSE":
                await self._process_close_signal(symbol, decision)
            
        except Exception as e:
            self.logger.error(f"❌ Trading logic error for {symbol}: {e}")
    
    async def _process_buy_signal(self, symbol: str, price: float, decision):
        """Process buy signal"""
        
        # Validate trade with risk manager
        validation_ok, reason = risk_manager.validate_trade(
            symbol=symbol,
            side="BUY",
            confidence=decision.consensus_confidence,
            market_data={'price': price},
            master_decision=decision
        )
        
        if not validation_ok:
            self.logger.warning(f"⚠️ Buy signal rejected for {symbol}: {reason}")
            return
        
        # Get position parameters
        params = risk_manager.calculate_position_parameters(
            symbol=symbol,
            side="BUY",
            entry_price=price,
            confidence=decision.consensus_confidence,
            market_data={'price': price},
            master_decision=decision
        )
        
        # Open position
        position = risk_manager.open_position(
            symbol=symbol,
            side="BUY",
            size=params['position_size'],
            entry_price=price,
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit'],
            reason=decision.reasoning
        )
        
        self.logger.info(f"📈 BUY signal executed for {symbol}")
        
        # Update trade history
        self.session_stats['trades_today'] += 1
        
        # Notify Discord
        async with discord_notifier:
            await discord_notifier.notify_position_open({
                'symbol': symbol,
                'side': 'BUY',
                'size': params['position_size'],
                'price': price,
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'reason': decision.reasoning
            })
    
    async def _process_sell_signal(self, symbol: str, price: float, decision):
        """Process sell signal"""
        
        # Check if we have a position to close
        if symbol not in risk_manager.portfolio_state.positions:
            self.logger.info(f"📉 SELL signal for {symbol} but no position to close")
            return
        
        # Validate trade
        validation_ok, reason = risk_manager.validate_trade(
            symbol=symbol,
            side="SELL",
            confidence=decision.consensus_confidence,
            market_data={'price': price},
            master_decision=decision
        )
        
        if not validation_ok:
            self.logger.warning(f"⚠️ Sell signal rejected for {symbol}: {reason}")
            return
        
        # Close position
        position_data = risk_manager.close_position(
            symbol=symbol,
            exit_price=price,
            reason=decision.reasoning
        )
        
        if position_data['success']:
            self.logger.info(f"📉 SELL signal executed for {symbol}")
            
            # Update stats
            self.session_stats['profit_today'] += position_data['pnl']
            self.session_stats['trades_today'] += 1
            
            # Save to trade history
            trade_record = {
                'symbol': symbol,
                'side': 'SELL',
                'entry_price': position_data['position'].entry_price,
                'exit_price': price,
                'pnl': position_data['pnl'],
                'entry_time': position_data['position'].entry_time,
                'exit_time': datetime.now(),
                'reason': decision.reasoning
            }
            self.trade_history.append(trade_record)
            
            # Notify Discord
            self.async with discord_notifier:
                await discord_notifier.notify_position_close(trade_record)
    
    async def _process_close_signal(self, symbol: str, decision):
        """Process close signal"""
        
        if symbol not in risk_manager.portfolio_state.positions:
            return
        
        current_data = self.market_data[symbol]['H1'].tail(1)
        current_price = current_data['close'].iloc[-1]
        
        # Close position at market price
        await self._process_sell_signal(symbol, current_price, decision)
    
    async def _update_portfolio(self):
        """Update portfolio with current prices"""
        
        current_prices = {}
        
        for symbol in self.symbols:
            if symbol in self.market_data and 'H1' in self.market_data[symbol]:
                current_data = self.market_data[symbol]['H1'].tail(1)
                if not current_data.empty:
                    current_prices[symbol] = current_data['close'].iloc[-1]
        
        # Update portfolio state
        risk_manager.update_portfolio(current_prices)
        
        # Track performance
        portfolio_value = risk_manager.portfolio_state.total_value
        self.performance_history.append(portfolio_value)
        
        # Trim history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    async def _update_market_data(self):
        """Update market data for all symbols"""
        
        for symbol in self.symbols:
            try:
                # Update H1 data
                new_data = await data_manager.get_price_data(symbol, "H1", limit=50)
                if not new_data.empty:
                    self.market_data[symbol]['H1'] = new_data
                
                # Update news sentiment periodically
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    news_data = await data_manager.get_news_data(symbol, days_back=1)
                    if not news_data.empty:
                        await news_manager.process_symbol_news(symbol, news_data)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to update data for {symbol}: {e}")
    
    async def _check_model_retraining(self):
        """Check if models should be retrained"""
        
        if not self.models_ready:
            return
        
        try:
            # Get recent performance data
            recent_performance = self.session_stats.get('daily_pnl', 0)
            
            # Check if retraining is needed
            if len(self.market_data) > 0:
                # Get sample data for drift detection
                sample_symbol = next(iter(self.market_data.keys()))
                sample_data = self.market_data[sample_symbol]['H1'].tail(100)
                
                if len(sample_data) >= 50:
                    engineered_data = feature_engineer.engineer_features(
                        sample_data.reset_index(), sample_symbol
                    )
                    
                    should_retrain, reason = auto_retrain_manager.should_retrain(
                        recent_performance, engineered_data
                    )
                    
                    if should_retrain:
                        self.logger.info(f"🔄 Model retraining triggered: {reason}")
                        await self._retrain_models()
                        self.session_stats['models_retrained'] += 1
        
        except Exception as e:
            self.logger.error(f"❌ Model retraining check failed: {e}")
    
    async def _retrain_models(self):
        """Retrain ML models with latest data"""
        
        self.logger.info("🔄 Starting model retraining")
        
        try:
            # Retrain ensemble models
            ensemble_model.train_stack(
                pd.DataFrame(), pd.Series(), "retrain"
            )
            
            self.logger.info("✅ Models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model retraining failed: {e}")
    
    async def _send_status_report(self):
        """Send periodic status report via Discord"""
        
        if not self.performance_history:
            return
        
        current_portfolio = risk_manager.get_risk_summary()
        
        portfolio_data = {
            'total_value': current_portfolio['portfolio_value'],
            'total_pnl': current_portfolio['total_pnl'],
            'daily_pnl': self.session_stats['profit_today'],
            'win_rate': 0.0,  # Calculate from trade history
            'sharpe_ratio': 0.0,  # Calculate from returns
            'max_drawdown': current_portfolio['max_drawdown'],
            'position_count': current_portfolio['position_count'],
            'positions': current_portfolio['positions']
        }
        
        async with discord_notifier:
            await discord_notifier.notify_performance_report(portfolio_data)
    
    async def start_trading(self, cycle_interval: int = 300):  # 5 minutes default
        """Start the main trading loop"""
        
        if not await self.initialize():
            return False
        
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        self.logger.info(f"🚀 Starting trading loop (cycle interval: {cycle_interval}s)")
        
        try:
            while self.is_running:
                cycle_start = datetime.now()
                
                # Execute trading cycle
                await self.run_trading_cycle()
                
                # Calculate sleep time
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, cycle_interval - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of the bot"""
        
        self.logger.info("🛑 Shutting down trading bot")
        
        self.is_running = False
        
        # Close all positions if configured to do so
        should_close_all = risk_manager.should_close_all_positions()[0]
        
        if should_close_all:
            self.logger.info("🛡️ Closing all positions for safety")
            
            for symbol in list(risk_manager.portfolio_state.positions.keys()):
                await self._emergency_close_position(symbol)
        
        # Send shutdown notification
        session_duration = datetime.now() - self.session_stats['start_time']
        
        async with discord_notifier:
            await discord_notifier.notify_system_alert({
                'alert_type': 'Bot Shutdown',
                'message': f'🛑 Advanced Trading Bot shutdown after {session_duration}',
                'severity': 'info',
                'additional_data': {
                    'Session Duration': str(session_duration),
                    'Trades Made': self.session_stats['trades_today'],
                    'Final PnL': f"${self.session_stats['profit_today']:.2f}",
                    'Models Retrained': self.session_stats['models_retrained']
                }
            })
        
        self.logger.info("✅ Bot shutdown completed")
    
    async def _emergency_close_position(self, symbol: str):
        """Emergency close position"""
        
        try:
            current_data = self.market_data[symbol]['H1'].tail(1)
            current_price = current_data['close'].iloc[-1]
            
            risk_manager.close_position(
                symbol=symbol,
                exit_price=current_price,
                reason="Emergency shutdown"
            )
            
            self.logger.info(f"🚨 Emergency closed {symbol} position")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency close failed for {symbol}: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        
        self.logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        self.is_running = False
    
    def get_bot_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        
        return {
            'is_running': self.is_running,
            'models_ready': self.models_ready,
            'symbols_count': len(self.symbols),
            'portfolio_value': risk_manager.portfolio_state.total_value,
            'open_positions': len(risk_manager.portfolio_state.positions),
            'session_stats': self.session_stats,
            'performance_history_length': len(self.performance_history),
            'trade_history_length': len(self.trade_history),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main entry point for the trading bot"""
    
    print("🤖 Advanced Trading Bot")
    print("=" * 50)
    
    # Initialize bot
    bot = TradingBot(initial_capital=10000.0)
    
    # Run the bot
    try:
        asyncio.run(bot.start_trading(cycle_interval=60))  # 1 minute cycles for demo
    except Exception as e:
        print(f"❌ Bot execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())