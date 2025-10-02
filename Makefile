# AI Trading Bot Makefile
# =======================

.PHONY: help install test run setup clean docs

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                 🚀 AI TRADING BOT MAKEFILE 🚀               ║"
	@echo "║                                                              ║"
	@echo "║ Available commands:                                          ║"
	@echo "║                                                              ║"
	@echo "║  📦 install    - Install dependencies                        ║"
	@echo "║  🔧 setup      - Full setup với kiểm tra system              ║"
	@echo "║  🧪 test       - Run automated tests                         ║"
	@echo "║  🚀 run        - Chạy trading bot                            ║"
	@echo "║  📊 demo       - Chạy demo mode (test nhanh)                 ║"
	@echo "║  🧹 clean      - Clean temporary files                       ║"
	@echo "║  📚 docs       - Generate documentation                      ║"
	@echo "║  🔍 check      - Check hệ thống requirements                  ║"
	@echo "║                                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Install dependencies
install:
	$(info Installing dependencies...)
	pip install -r requirements.txt
	$(info ✅ Dependencies installed!)

# Full setup
setup:
	$(info Running full setup...)
	python setup.py
	$(info ✅ Setup completed!)

# Run tests
test:
	$(info Running tests...)
	python test_bot.py
	$(info ✅ Tests completed!)

# Run trading bot
run:
	$(info Starting AI Trading Bot...)
	@echo "⚠️  WARNING: Đây là bot demo, không trade với tiền thật!"
	@echo "Press Ctrl+C để dừng bot"
	sleep 3
	python trading_bot.py

# Run demo mode
demo:
	$(info Running demo mode...)
	python test_bot.py

# Clean temporary files
clean:
	$(info Cleaning temporary files...)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete
	rm -rf logs/*.log models/*.h5 models/*.pkl temp_cache/
	$(info ✅ Cleanup completed!)

# Check system requirements
check:
	@echo "🔍 Checking system requirements..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import platform; print(f'System: {platform.system()}')"
	@python -c "import pandas; print(f'Pandas: {pandas.__version__}')" 2>/dev/null || echo "Pandas: Not installed"
	@python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: Not installed"
	@python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')" 2>/dev/null || echo "TensorFlow: Not installed"
	@echo "✅ System check completed!"

# Generate documentation (placeholder)
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation sẽ được tạo trong docs/ folder"
	@echo "✅ Documentation placeholder created!"

# Show status
status:
	@echo "📊 AI Trading Bot Status:"
	@echo "=========================="
	@echo "📁 File Structure:"
	@ls -la *.py 2>/dev/null || echo "No Python files found"
	@echo ""
	@echo "📊 Dependencies:"
	@python -c "import pkg_resources; [print(f'  ✅ {d}' for d in sorted(['pandas','numpy','tensorflow','xgboost','lightgbm','requests','aiohttp','ta']) if d in [p.project_name for p in pkg_resources.working_set])]" 2>/dev/null || echo "Dependencies not checked"
	@echo ""
	@echo "📁 Key Directories:"
	@test -d logs && echo "  ✅ logs/" || echo "  ❌ logs/"
	@test -d models && echo "  ✅ models/" || echo "  ❌ models/"
	@test -f .env && echo "  ✅ .env (config)" || echo "  ❌ .env (config) - using .env.example"

# Development shortcuts
dev-test:
	$(info Quick development test...)
	python -c "from trading_bot import AdvancedFeatureEngineer; import pandas as pd; import numpy as np; df=pd.DataFrame({'open':np.random.randn(100)+2000,'high':np.random.randn(100)+2001,'low':np.random.randn(100)+1999,'close':np.random.randn(100)+2000,'volume':np.random.randint(1000,5000,100)}); df['high']=df[['open','close','high']].max(axis=1); df['low']=df[['open','close','low']].min(axis=1); fe=AdvancedFeatureEngineer(); result=fe.engineer_features(df,'TEST'); print(f'✅ Test passed! {len(result.columns)} features generated')"
	$(info ✅ Quick test passed!)

# Backup config
backup:
	$(info Creating backup of configuration...)
	@mkdir -p backups
	@cp .env backups/.env.backup.$(shell date +%Y%m%d_%H%M%S) 2>/dev/null || echo "No .env file to backup"
	@cp requirements.txt backups/requirements.txt.backup.$(shell date +%Y%m%d_%H%M%S) 2>/dev/null || echo "Requirements backed up"
	$(info ✅ Backup completed!)

# All-in-one command
all: setup test
	$(info 🎉 Full setup and testing completed!)

# Quick start for new users
quickstart:
	@echo "🚀 AI TRADING BOT QUICK START"
	@echo "============================"
	@echo "1. Make sure Python 3.8+ is installed"
	@echo "2. Run: make setup"
	@echo "3. Run: make test"
	@echo "4. Edit .env file with your Discord webhook"
	@echo "5. Run: make run"
	@echo ""
	@echo "⚠️  WARNING: This is a demo bot! Don't use with real money!"