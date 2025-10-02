# 🤖 Advanced Trading Bot với AI/ML

Một trading bot tinh vi sử dụng trí tuệ nhân tạo, machine learning, reinforcementlearning và sentiment analysis để tự động giao dịch trên thị trường tài chính.

## 🌟 Tính năng chính

### 🧠 Lõi AI & Machine Learning
- **Ensemble Models**: XGBoost, LightGBM, Random Forest với stacking và calibration
- **Deep Learning**: LSTM với Attention mechanism và Batch Normalization
- **Reinforcement Learning**: PPO Agent cho quản lý portfolio thông minh
- **Hyperparameter Optimization**: Tối ưu hóa tự động với Optuna
- **Auto Retraining**: Huấn luyện lại mô hình dựa trên concept drift và performance

### 🧬 Kỹ thuật Đặc trưng Nâng cao
- **Technical Indicators**: ATR, ADX, RSI, MACD, Bollinger Bands, và nhiều hơn
- **Wyckoff Analysis**: Phương pháp Wyckoff cho phân tích cấu trúc thị trường
- **Supply/Demand Zones**: Tự động nhận diện vùng cung cầu quan trọng
- **RSI Divergence**: Phát hiện tín hiệu phân kỳ RSI tự động
- **Market Regime Detection**: Phân loại thị trường "Trending" vs "Sideways"

### 🎯 Master Agent System
- **Hierarchical AI**: Master Agent điều phối các specialist agents
- **Specialist Agents**: Trend Analysis, Risk Management, Sentiment Analysis, News Analysis, Volatility Prediction
- **Consensus Decision Making**: Ra quyết định dựa trên sự đồng thuận của AI agents

### 💰 Quản lý Rủi ro Đa tầng
- **Dynamic Position Sizing**: Tính toán kích thước vị thế linh hoạt
- **Smart Stop Loss**: SL/TP thông minh do Master Agent quyết định
- **Trailing Stop**: Trailing stop động với khoảng cách linh hoạt
- **Correlation Management**: Kiểm tra tương quan và ngăn chặn over-concentration
- **Trade Gating**: Tạm dừng giao dịch quanh các sự kiện kinh tế lớn

### 📊 Quản lý Dữ liệu Nâng cao
- **Multi-timeframe Analysis**: M15, H1, H4, D1, W1
- **Data Freshness Monitoring**: Tự động kiểm tra và refresh dữ liệu
- **Multi-source News**: Tích hợp NewsAPI, Finnhub, Marketaux, EODHD
- **Economic Calendar**: Lịch kinh tế miễn phí từ FRED và ForexFactory
- **AI Sentiment Analysis**: Gemini 1.5 Flash để phân tích cảm tính tin tức

### 🔔 Discord Integration
- **Rich Notifications**: Thông báo embed với emoji và formatting
- **Real-time Alerts**: Tín hiệu, mở/đóng lệnh, cập nhật SL, cảnh báo hệ thống
- **Performance Reports**: Báo cáo hiệu suất danh mục định kỳ
- **Trade Logging**: Log giao dịch chi tiết với metadata

## 🚀 Cài đặt và Sử dụng

### Prerequisites
```bash
Python 3.9+
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd advanced-trading-bot

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp config.example.py config.py
# Edit config.py với các API keys của bạn
```

### Configuration
Chỉnh sửa `config.py` với các thông tin API của bạn:

```python
# Trading APIs
vantage_api_key: str = "FK3YQ1IKSC4E1AL5"
finnhub_api_key: str = "d1b3ichr01qjhvtsbj8g"
marketaux_api_key: str = "CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql"
newsapi_key: str = "abd8f43b808f42fdb8d28fb1c429af72"
eodhd_api_key: str = "68bafd7d44a7f0.25202650"

# OANDA Trading
oanda_api_key: str = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"
oanda_url: str = "https://api-fxtrade.oanda.com/v3"

# Discord Webhook
discord_webhook_url = "https://discord.com/api/webhooks/1419645732218081290/xamfJQdl5kay1wo6w6gxQRrW77d1jpSzKBstQ16Qvb4t5ncGJ3nIHMmm3MQPNT_E-Bt2"
```

### Running the Bot
```bash
# Chạy bot với cấu hình mặc định
python trading_bot.py

# Hoặc chạy với custom parameters
python -c "
import asyncio
from trading_bot import TradingBot

async def run():
    bot = TradingBot(initial_capital=10000.0)
    await bot.start_trading(cycle_interval=300)  # 5 phút cycle_interval

asyncio.run(run())
"
```

## 📈 Supported Symbols

Bot hỗ trợ giao dịch các symbol sau:
- **Forex**: EURUSD
- **Commodities**: XAUUSD (Gold)
- **Crypto**: BTCUSD
- **Indices**: NASDAQ100

## 🛡️ Risk Management Features

### Position Sizing
- Dynamic sizing based on confidence, volatility, and risk score
- Kelly Criterion và Fixed Risk methods
- Maximum 2% risk per trade (configurable)

### Stop Loss Management
- ATR-based stops với régime adjustment
- Support/Resistance level consideration
- Smart trailing stops triggered by Master Agent

### Portfolio Protection
- Maximum total exposure limit (configurable)
- Correlation checking to prevent over-concentration
- Emergency shutdown for excessive drawdown

## 📊 Performance Monitoring

Bot tự động theo dõi và báo cáo:
- Portfolio value và P&L
- Win rate và Sharpe ratio
- Maximum drawdown
- Model performance metrics
- API health status

## 🎮 Discord Commands

Bot gửi thông báo Discord cho:
- `🚨 Trading Signals`: Tín hiệu mua/bán với confidence và reasoning
- `📈 Position Opens`: Thông tin vị thế mới mở
- `📉 Position Closes`: Kết quả đóng vị thế với P&L
- `🎯 Stop Loss Updates`: Cập nhật trailing stop
- `⚠️ System Alerts`: Cảnh báo hệ thống và lỗi
- `📊 Performance Reports`: Báo cáo hiệu suất định kỳ

## 🔧 Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```bash
   # Bot tự động quản lý rate limiting với retry logic
   # Nếu vẫn gặp vấn đề, kiểm tra API keys và quotas
   ```

2. **Model Training Errors**
   ```bash
   # Bot sẽ tự động fallback về rule-based signals
   # Kiểm tra logs để xem chi tiết lỗi mô hình
   ```

3. **Discord Notifications**
   ```bash
   # Test Discord integration
   python -c "
   import asyncio
   from discord_notifier import discord_notifier
   
   async def test():
       async with discord_notifier:
           await discord_notifier.send_test_message()
   
   asyncio.run(test())
   "
   ```

## 📱 Support

Chọn hỗ trợ và gửi thông tin:
- Docker logs (nếu sử dụng Docker)
- Screenshot Discord notifications
- Performance charts
- Config settings (không bao gồm keys)

## ⚠️ Disclaimer

Bot này là for educational purposes. Trading forex/crypto có rủi ro cao và có thể dẫn đến thua lỗ đáng kể. Luôn:
- Paper trade trước khi deploy với capital thật
- Start với amount nhỏ để test
- Monitor performance thường xuyên
- Understand risks involved
- Comply with local regulations

## 🔄 Updates

Bot được thiết kế để continuously improve:
- Auto-retraining based on performance
- Online學習 cho rapid adaptation
- Feature engineering improvements
- New ML models integration

Hệ thống này sẽ continuously evolve và adapt với market conditions!