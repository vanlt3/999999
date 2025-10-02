# 🚀 AI Trading Bot v1.0

Bot giao dịch tự động hoàn chỉnh với tích hợp AI/ML tiên tiến, phân tích tin tức cảm tính và quản lý rủi ro đa tầng.

## ✨ Tính Năng Chính

### 🤖 Machine Learning & AI
- **Ensemble Models**: XGBoost, LightGBM, Random Forest với Stacking
- **LSTM Networks**: Với Attention mechanism và Batch Normalization
- **Reinforcement Learning**: PPO agent cho portfolio management
- **Concept Drift Detection**: Tự động phát hiện và retrain models
- **Hyperparameter Optimization**: Optuna integration

### 📊 Phân Tích Kỹ Thuật Nâng Cao
- **Technical Indicators**: ATR, ADX, RSI, MACD, Bollinger Bands, Stochastic
- **Pattern Recognition**: Candlestick patterns, Wyckoff analysis
- **Supply/Demand Zones**: Automatic detection và monitoring
- **RSI Divergence**: Phân tích phân kỳ tự động
- **Market Regime**: Trending/Sideways classification

### 📰 Phân Tích Tin Tức & Cảm Tính
- **Multi-source News**: Finnhub, Marketaux, NewsAPI, EODHD
- **AI Sentiment Analysis**: Gemini 1.5 Flash integration
- **Economic Calendar**: Automatic event monitoring
- **News Features**: Sentiment score, impact level, symbol relevance

### ⚡ Hiệu Suất & Monitoring
- **Async Architecture**: Non-blocking data collection
- **Real-time Monitoring**: Discord webhook notifications
- **Performance Tracking**: Sharpe ratio, max drawdown, win rate
- **Risk Management**: Multi-level risk controls

## 🎯 Symbols Hỗ Trợ

- **XAUUSD** (Gold vs USD)
- **EURUSD** (Euro vs USD)  
- **NAS100** (Nasdaq 100 Index)
- **BTCUSD** (Bitcoin vs USD)

## 🛠️ Cài Đặt & Chạy

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu Hình API Keys

File `trading_bot.py` đã có sẵn các API keys demo:
- Alpha Vantage: `FK3YQ1IKSC4E1AL5`
- Finnhub：`d1b3ichr01qjhvtsbj8g`
- Marketaux：`CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql`
- NewsAPI：`abd8f43b808f42fdb8d28fb1c429af72`
- EODHD：`68bafd7d44a7f0.25202650`
- OANDA：`814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d`

### 3. Cấu Hình Discord

Thay webhook URL trong file:
```python
DISCORD_WEBHOOK = "YOUR_DISCORD_WEBHOOK_URL"
```

### 4. Chạy Bot

```bash
python trading_bot.py
```

## 🏗️ Kiến Trúc Hệ Thống

```
📁 Trading Bot Architecture
├── 🔧 Data Management Layer
│   ├── EnhancedDataManager (OANDA API)
│   ├── DataFreshnessManager
│   └── Multi-timeframe Processing
├── 🧠 AI/ML Layer
│   ├── EnsembleModel (XGBoost + LightGBM + RF)
│   ├── LSTMModel (Neural Networks)
│   ├── RLAgent (PPO Reinforcement Learning)
│   └── AutoRetrainManager
├── 📊 Feature Engineering
│   ├── TechnicalIndicators
│   ├── WyckoffAnalyzer
│   ├── SupplyDemandAnalyzer
│   └── MarketRegimeDetector
├── 📰 News & Economic Data
│   ├── MultiSourceNewsAggregator
│   ├── GeminiSentimentAnalyzer
│   └── FinancialCalendarReader
├── 🎛️ Trading Execution
│   ├── TradingSignal Generation
│   ├── Position Management
│   └── Risk Management
└── 📱 Monitoring & Reporting
    ├── Discord Notifications
    ├── Performance Tracking
    └── Log Management
```

## 📈 Cách Thức Hoạt Động

### 1. Data Collection
- Thu thập dữ liệu đa khung thời gian (M15, H1, H4, D1)
- Cache thông minh với freshness checking
- Real-time price updates từ OANDA API

### 2. Feature Engineering
- Tính toán 50+ technical indicators
- Pattern recognition và market structure analysis
- Wyckoff signals và supply/demand zones
- Market regime classification

### 3. Signal Generation
- Ensemble prediction từ multiple models
- Confidence scoring và model weighting
- Market regime-based model selection
- RL agent portfolio recommendations

### 4. Risk Management
- Position sizing dựa trên confidence và ATR
- Dynamic stop loss/take profit levels
- Correlation checking between positions
- Drawdown monitoring và emergency stops

### 5. Execution & Monitoring
- Real-time position monitoring
- Trailing stops và profit management
- Performance tracking và reporting
- Discord notifications cho tất cả events

## 🔧 Configuration

### Risk Parameters
```python
MAX_POSITIONS = 4          # Maximum concurrent positions
MAX_DAILY_TRADES = 20      # Daily trading limit
MAX_DRAWDOWN = 0.05        # 5% maximum drawdown
PORTFOLIO_VALUE = 10000.0  # Starting capital
```

### Model Training
```python
MODEL_RETRAIN_FREQUENCY = 24    # Hours between retraining
CONCEPT_DRIFT_THRESHOLD = 0.1   # Drift detection sensitivity
OVERFIT_CONFIDENCE_THRESHOLD = 0.95  # Overfitting prevention
```

## 📊 Monitoring Dashboard

Bot tự động gửi thông báo Discord bao gồm:
- ✅ Tín hiệu giao dịch mới
- 💰 Mở/đóng vị thế với P&L
- 📈 Báo cáo hiệu suất định kỳ
- ⚠️ Cảnh báo rủi ro và lỗi hệ thống

## 🚨 Cảnh Báo Rủi Ro

⚠️ **QUAN TRỌNG**: Đây là bot demo chỉ để học tập và research. Không sử dụng với tiền thật mà không hiểu rủi ro.

- Bot có thể thua lỗ nguồn vốn
- Thị trường forex có rủi ro cao
- Luôn test thoroughly trước khi trade real
- Kiểm tra API limits và costs

## 🔧 Troubleshooting

### Lỗi TensorFlow/CUDA
```bash
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
```

### Lỗi API Rate Limits
Bot tự động handle rate limits và retry logic. Nếu gặp vấn đề:
1. Kiểm tra API keys validity
2. Verify network connection  
3. Check API quota limits

### Performance Issues
- Reduce number of symbols traded
- Lower frequency của trading cycles
- Optimize model complexity cho hardware

## 📝 Logs & Debugging

Logs được lưu trong `logs/trading_bot.log` với:
- Colored console output với emojis
- Module-specific logging levels
- Performance metrics tracking
- Error stack traces

## 🤝 Contributing

Để contribute vào project:
1. Fork repository
2. Create feature branch
3. Add comprehensive tests
4. Submit pull request với detailed description

## 📄 License

MIT License - Xem file LICENSE cho chi tiết.

## 🎓 Educational Use

Bot này được thiết kế cho:
- Learning về algorithmic trading
- Research về ML applications trong finance
- Understanding market microstructure
- Practicing với modern AI techniques

---

**⚡ Happy Trading với AI! ⚡**

Được phát triển với ❤️ và nhiều ☕