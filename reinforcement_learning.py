"""
Reinforcement Learning Agent with Portfolio Environment
Advanced RL implementation using PPO from Stable Baselines3 for portfolio management
"""

import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# RL Libraries
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Online Learning
import river
from river import ensemble, linear_model, preprocessing, optim

from logger import rl_logger, get_logger
from config import trading_config, ASSET_TYPES

@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    portfolio_value: float
    total_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    sharpe_ratio: float
    max_drawdown: float

class PortfolioEnvironment(gym.Env):
    """Custom gym environment for portfolio trading"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 10000, 
                 transaction_cost: float = 0.001, max_position_size: float = 0.1):
        super(PortfolioEnvironment, self).__init__()
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Environment parameters
        self.current_step = 0
        self.n_steps = 0
        self.data = None
        self.reward_history = []
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.portfolio_history = []
        
        # Action space: [action_per_symbol, hold_action] for each symbol
        n_actions_per_symbol = 3  # Buy, Sell, Hold
        self.action_space = gym.spaces.MultiDiscrete([3] * len(symbols))
        
        # Observation space: portfolio + market data features
        n_features = 50  # Price, technical indicators, etc.
        n_portfolio_features = 10  # Cash ratio, position ratios, etc.
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(symbols) * n_features + n_portfolio_features,),
            dtype=np.float32
        )
        
        self.logger = get_logger("PortfolioEnvironment")
        
    def set_data(self, data: Dict[str, pd.DataFrame]):
        """Set market data for the environment"""
        self.data = data
        
        # Ensure all datasets have the same length
        min_length = min(len(df) for df in data.values())
        self.n_steps = min_length - 50  # Leave buffer for technical indicators
        
        self.logger.info(f"📊 Set data for environment: {self.n_steps} steps")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.reward_history = []
        self.portfolio_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= self.n_steps:
            return self._get_observation(), 0, True, {}
        
        # Execute actions
        reward = self._execute_actions(action)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.n_steps
        
        # Log portfolio state
        if self.current_step % 100 == 0:
            portfolio_value = self._calculate_portfolio_value()
            self.logger.debug(f"Step {self.current_step}: Portfolio Value: ${portfolio_value:.2f}")
        
        observation = self._get_observation() if not done else None
        info = self._get_info()
        
        return observation, reward, done, info
    
    def _execute_actions(self, action: np.ndarray) -> float:
        """Execute trading actions and calculate reward"""
        total_reward = 0
        
        for symbol_idx, action_value in enumerate(action):
            symbol = self.symbols[symbol_idx]
            
            if symbol not in self.data:
                continue
            
            # Get current market data
            market_data = self.data[symbol].iloc[self.current_step]
            current_price = market_data['close']
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value()
            
            # Asset allocation constraint
            max_allocation = portfolio_value * self.max_position_size
            
            if action_value == 0:  # Buy
                if self.cash >= current_price:
                    # Calculate position size (fractional shares for simplicity)
                    shares_to_buy = min(
                        self.cash // (current_price * (1 + self.transaction_cost)),
                        max_allocation // (current_price * (1 + self.transaction_cost))
                    )
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                        self.cash -= cost
                        self.positions[symbol] += shares_to_buy
                        
                        # Small positive reward for successful trade
                        total_reward += 0.01
            
            elif action_value == 1:  # Sell
                if self.positions[symbol] > 0:
                    # Calculate number of shares to sell (can be partial)
                    shares_to_sell = self.positions[symbol]
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                    
                    self.cash += proceeds
                    self.positions[symbol] = 0
                    
                    # Calculate profit/loss
                    if shares_to_sell > 0:
                        # This is simplified - in reality you'd track entry prices
                        total_reward += 0.01
            
            # Hold action gets small negative reward to encourage trading
        
        # Calculate overall portfolio reward
        portfolio_reward = self._calculate_portfolio_reward()
        total_reward += portfolio_reward
        
        self.reward_history.append(total_reward)
        self.portfolio_history.append(self._get_portfolio_state())
        
        return total_reward
    
    def _calculate_portfolio_reward(self) -> float:
        """Calculate reward based on Sharpe ratio and risk-adjusted returns"""
        if len(self.reward_history) < 10:
            return 0
        
        recent_rewards = self.reward_history[-10:]
        returns = np.diff(recent_rewards) if len(recent_rewards) > 1 else [0]
        
        if len(returns) == 0:
            return 0
        
        # Sharpe ratio based reward
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe_ratio = mean_return / std_return
            # Scale Sharpe ratio for reward
            reward = sharpe_ratio * 0.1
        else:
            reward = 0
        
        # Penalty for excessive risk
        current_portfolio_value = self._calculate_portfolio_value()
        drawdown = (self.initial_capital - current_portfolio_value) / self.initial_capital
        
        if drawdown > 0.1:  # 10% drawdown
            reward -= 0.5
        
        # Bonus for profitable trades
        total_pnl = current_portfolio_value - self.initial_capital
        if total_pnl > 0:
            reward += total_pnl / self.initial_capital * 0.1
        
        return reward
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in self.data:
                current_price = self.data[symbol].iloc[self.current_step]['close']
                total_value += quantity * current_price
        
        return total_value
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.data is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Market data features
        market_features = []
        
        for symbol in self.symbols:
            if symbol in self.data and len(self.data[symbol]) > self.current_step:
                # Get current and recent price data
                current_data = self.data[symbol].iloc[max(0, self.current_step-20):self.current_step+1]
                
                # Price-based features
                price_features = [
                    current_data['close'].iloc[-1] if len(current_data) : 0,
                    current_data['volume'].iloc[-1] if len(current_data) else 0,
                    np.std(current_data['close']) if len(current_data) > 1 else 0,
                    np.mean(current_data['close']) if len(current_data) > 0 else 0
                ]
                
                # Technical indicators (simplified)
                if len(current_data) > 10:
                    sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
                    rsi = self._calculate_rsi(current_data['close'].values)[-1]
                else:
                    sma_10, rsi = 0, 50
                
                price_features.extend([sma_10, rsi])
                
                # Pad features to consistent length
                while len(price_features) < 8:
                    price_features.append(0)
                
                market_features.extend(price_features)
            else:
                # Pad with zeros if no data
                market_features.extend([0] * 8)
        
        # Portfolio features
        portfolio_value = self._calculate_portfolio_value()
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1
        
        # Position ratios
        position_ratios = []
        for symbol in self.symbols:
            quantity = self.positions[symbol]
            if quantity > 0 and symbol in self.data and portfolio_value > 0:
                current_price = self.data[symbol].iloc[self.current_step]['close']
                position_value = quantity * current_price
                position_ratio = position_value / portfolio_value
            else:
                position_ratio = 0
            position_ratios.append(position_ratio)
        
        portfolio_features = [
            cash_ratio,
            np.sum(position_ratios),  # Total exposure
            np.max(position_ratios),  # Max position
            np.std(position_ratios),  # Position diversification
            len([r for r in position_ratios if r > 0]),  # Number of positions
            self.current_step / self.n_steps,  # Progress
            len(self.reward_history),  # Number of trades
            np.mean(self.reward_history[-10:]) if self.reward_history else 0,  # Recent performance
            np.std(self.re the_history[-10:]) if len(self.reward_history) >= 10 else 0,  # Recent volatility
            portfolio_value / self.initial_capital  # Portfolio growth
        ]
        
        all_features = market_features + portfolio_features
        
        return np.array(all_features, dtype=np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        portfolio_value = self._calculate_portfolio_value()
        
        total_exposure = sum(
            self.positions[symbol] * self.data[symbol].iloc[self.current_step]['close']
            for symbol in self.positions.keys()
            if symbol in self.data and self.positions[symbol] > 0
        )
        
        unrealized_pnl = 0  # Simplified - would need entry prices
        realized_pnl = portfolio_value - self.initial_capital
        
        # Calculate Sharpe ratio
        if len(self.reward_history) > 10:
            returns = np.diff(self.reward_history[-10:])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        max_drawdown = max(0, (self.initial_capital - portfolio_value) / self.initial_capital)
        
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information"""
        portfolio_state = self._get_portfolio_state()
        
        return {
            'portfolio_value': portfolio_state.portfolio_value,
            'cash_ratio': self.cash / portfolio_state.portfolio_value,
            'total_exposure': portfolio_state.total_exposure,
            'sharpe_ratio': portfolio_state.sharpe_ratio,
            'max_drawdown': portfolio_state.max_drawdown,
            'step': self.current_step,
            'total_steps': self.n_steps
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if self.current_step % 50 == 0:  # Log every 50 steps
            portfolio_state = self._get_portfolio_state()
            
            self.logger.info(f"""
🎯 Portfolio Environment Step {self.current_step}:
💰 Portfolio Value: ${portfolio_state.portfolio_value:.2f}
💵 Cash: ${portfolio_state.cash:.2f}
📊 Total Exposure: ${portfolio_state.total_exposure:.2f}
📈 Sharpe Ratio: {portfolio_state.sharpe_ratio:.3f}
📉 Max Drawdown: {portfolio_state.max_drawdown:.2%}
""")

class EarlyStoppingCallback(BaseCallback):
    """Custom callback to stop training early if performance is poor"""
    
    def __init__(self, check_freq: int, min_performance: float = -0.1, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.min_performance = min_performance
        self.best_performance = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current performance
            if len(self.model.ep_info_buffer) > 0:
                recent_performance = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
                if self.verbose > 0:
                    print(f"Recent performance: {recent_performance:.3f}")
                
                # Stop if performance is too poor
                if recent_performance < self.min_performance:
                    if self.verbose > 0:
                        print(f"Stopping training due to poor performance: {recent_performance:.3f}")
                    return False
                
                # Update best performance
                self.best_performance = max(self.best_performance, recent_performance)
        
        return True

class RLAgent:
    """Main RL Agent using PPO algorithm"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 10000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.logger = rl_logger
        
        # Environment
        self.env = None
        self.model = None
        
        # Online learning components
        self.online_models = {}
        self._init_online_models()
    
    def _init_online_models(self):
        """Initialize online learning models"""
        for symbol in self.symbols:
            # Online logistic regression for quick adaptation
            self.online_models[symbol] = river.compose.Pipeline(
                preprocessing.StandardScaler(),
                ensemble.AdaptiveRandomForestRegressor(n_models=20, seed=42)
            )
    
    def create_environment(self, data: Dict[str, pd.DataFrame]) -> PortfolioEnvironment:
        """Create and configure trading environment"""
        env = PortfolioEnvironment(
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            max_position_size=trading_config.max_position_size
        )
        
        env.set_data(data)
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        self.env = env
        
        self.logger.info(f"🎮 Created portfolio environment for {len(self.symbols)} symbols")
        return env
    
    def train(self, data: Dict[str, pd.DataFrame], total_timesteps: int = 10000) -> Dict[str, Any]:
        """Train RL agent using PPO"""
        self.logger.info("🚂 Starting RL agent training")
        
        # Create environment
        if self.env is None:
            self.create_environment(data)
        
        # Create vectorized environment
        def make_env():
            env = PortfolioEnvironment(
                symbols=self.symbols,
                initial_capital=self.initial_capital,
                max_position_size=trading_config.max_position_size
            )
            env.set_data(data)
            return Monitor(env)
        
        vec_env = make_vec_env(make_env, n_envs=4)
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./logs/rl_trading/"
        )
        
        # Setup callbacks
        callbacks = [EarlyStoppingCallback(check_freq=1000, min_performance=-0.2)]
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_portfolio_trading"
        )
        
        self.logger.info("✅ RL agent training completed")
        
        # Evaluate performance
        performance = self.evaluate()
        
        return performance
    
    def evaluate(self, n_episodes: int = 5) -> Dict[str, Any]:
        """Evaluate trained agent performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        episode_rewards = []
        episode_sharpe_ratios = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_sharpe_ratios.append(info.get('sharpe_ratio', 0))
                    break
        
        performance = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_sharpe': np.mean(episode_sharpe_ratios),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        self.logger.info(f"📊 RL Agent Performance: {performance}")
        
        return performance
    
    def predict_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """Predict trading action"""
        if self.model is None:
            # Default action (hold all positions)
            return np.zeros(len(self.symbols)), 0.0
        
        action, _state = self.model.predict(observation, deterministic=True)
        confidence = self._calculate_confidence(action, observation)
        
        return action, confidence
    
    def _calculate_confidence(self, action: np.ndarray, observation: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # This is simplified - in reality you'd use the model's uncertainty
        # For now, return a placeholder based on action diversity
        unique_actions = len(set(action))
        confidence = unique_actions / len(self.symbols)
        
        return confidence
    
    def online_update(self, symbol: str, market_data: pd.DataFrame, action: int):
        """Update online learning models"""
        if symbol in self.online_models:
            # Prepare features from market data
            features = self._extract_features(market_data)
            
            # Train online model
            for i, (idx, row) in enumerate(market_data.iterrows()):
                # Convert action to target (simplified)
                target = action if i == len(market_data) - 1 else market_data['close'].iloc[i]
                
                self.online_models[symbol].learn_one(features[i], target)
    
    def _extract_features(self, market_data: pd.DataFrame) -> List[List[float]]:
        """Extract features from market data for online learning"""
        features = []
        
        for idx, row in market_data.iterrows():
            feature_vector = [
                row['open'], row['high'],Row['low'], row['close'],
                row['volume'], 
                row.get('RSI_14', 50),
                row.get('MACD', 0),
                row.get('ATR', 0)
            ]
            features.append(feature_vector)
        
        return features
    
    def save_model(self, filepath: str):
        """Save trained RL model"""
        if self.model:
            self.model.save(filepath)
            
            # Save online models
            online_path = filepath.replace('.zip', '_online.pkl')
            import joblib
            joblib.dump(self.online_models, online_path)
            
            self.logger.info(f"💾 Saved RL model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained RL model"""
        self.model = PPO.load(filepath)
        
        # Load online models
        online_path = filepath.replace('.zip', '_online.pkl')
        import joblib
        self.online_models = joblib.load(online_path)
        
        self.logger.info(f"📂 Loaded RL model from {filepath}")

class OnlineLearningManager:
    """Manage online learning for continuous adaptation"""
    
    def __init__(self):
        self.logger = get_logger("OnlineLearningManager")
        self.models = {}
        
    def bootstrap_models(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]):
        """Bootstrap online models with historical data"""
        self.logger.info("🚀 Bootstrapping online learning models")
        
        for symbol in symbols:
            if symbol in historical_data:
                # Initialize model with historical data
                data = historical_data[symbol]
                
                # Create online ensemble model
                model = river.compose.Pipeline(
                    preprocessing.StandardScaler(),
                    ensemble.AdaBoostRegressor(n_models=10, learning_rate=0.1)
                )
                
                # Train on historical data
                for idx, row in data.iterrows():
                    features = [
                        row['open'], row['high'], row['low'], row['close'],
                        row['volume']
                    ]
                    target = row['close']  # Predict next close price
                    model.learn_one(features, target)
                
                self.models[symbol] = model
                
                self.logger.info(f"✅ Bootstrapped model for {symbol}")
    
    def update_model(self, symbol: str, features: List[float], target: float):
        """Update online model with new data point"""
        if symbol in self.models:
            self.models[symbol].learn_one(features, target)
    
    def predict(self, symbol: str, features: List[float]) -> Tuple[float, float]:
        """Make prediction using online model"""
        if symbol not in self.models:
            return 0.0, 0.0
        
        prediction = self.models[symbol].predict_one(features)
        confidence = 0.8  # Placeholder confidence
        
        return prediction, confidence
    
    def get_model_performance(self, symbol: str) -> Dict[str, float]:
        """Get online model performance metrics"""
        if symbol not in self.models:
            return {'status': 'no_model'}
        
        # This would track actual performance metrics
        # For now, return placeholder
        return {
            'accuracy': 0.75,
            'last_update': datetime.now(),
            'total_updates': 1000,
            'status': 'active'
        }

# Global RL components
rl_agent = RLAgent(symbols=trading_config.symbols)
online_learning_manager = OnlineLearningManager()