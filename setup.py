#!/usr/bin/env python3
"""
Setup script cho AI Trading Bot
==============================

Script tự động setup environment và install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """In banner đẹp"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║               🚀 AI TRADING BOT SETUP v1.0 🚀               ║
    ║                                                              ║
    ║  Thiết lập tự động cho AI Trading Bot                        ║
    ║  ✅ Cài đặt dependencies                                     ║
    ║  ✅ Tạo thư mục cần thiết                                    ║
    ║  ✅ Cấu hình environment                                     ║
    ║  ✅ Kiểm tra hệ thống                                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("🐍 Kiểm tra phiên bản Python...")
    
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} không được hỗ trợ!")
        print("💡 Yêu cầu Python 3.8+ trở lên")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_system():
    """Kiểm tra hệ thống"""
    print(f"💻 Hệ thống: {platform.system()} {platform.release()}")
    print(f"🏗️ Kiến trúc: {platform.machine()}")
    print(f"🐍 Python path: {sys.executable}")

def install_dependencies():
    """Cài đặt dependencies từ requirements.txt"""
    print("\n📦 Cài đặt dependencies...")
    
    try:
        # Upgrade pip first
        print("🔄 Upgrading pip...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Install from requirements.txt
        print("📚 Installing packages fromrequirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt: {e}")
        print(f"💡 Output: {e.stdout}")
        print(f"💥 Error: {e.stderr}")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    print("\n📁 Tạo thư mục...")
    
    directories = [
        "logs",
        "models", 
        "data",
        "cache",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Tạo thư mục {directory}/")

def create_env_file():
    """Tạo file .env mẫu"""
    print("\n🔧 Tạo file .env mẫu...")
    
    env_content = """# AI Trading Bot Configuration
# Copy this file và chỉnh sửa các giá trị sau

# API Keys (đã có sẵn trong code)
ALPHA_VANTAGE_API_KEY=FK3YQ1IKSC4E1AL5
FINNHUB_API_KEY=d1b3ichr01qjhvtsbj8g
MARKETAUX_API_KEY=CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql
NEWSAPI_API_KEY=abd8f43b808f42fdb8d28fb1c429af72
EODHD_API_KEY=68bafd7d44a7f0.25202650
OANDA_API_KEY=814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d

# Discord Webhook (thay bằng webhook URL của bạn)
DISCORD_WEBHOOK=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE

# Gemini API Key cho sentiment analysis
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# Trading Parameters
PORTFOLIO_VALUE=10000.0
MAX_POSITIONS=4
MAX_DRAWDOWN=0.05

# Models
MODEL_RETRAIN_FREQUENCY=24
CONCEPT_DRIFT_THRESHOLD=0.1
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("✅ File .env.example đã được tạo")
    print("💡 Sao chép thành .env và chỉnh sửa các giá trị")

def test_imports():
    """Test import các thư viện chính"""
    print("\n🧪 Testing imports...")
    
    critical_libraries = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("tensorflow", "Deep learning"),
        ("xgboost", "Gradient boosting"),
        ("lightgbm", "Gradient boosting"),
        ("optuna", "Hyperparameter optimization"),
        ("requests", "HTTP requests"),
        ("aiohttp", "Async HTTP"),
        ("ta", "Technical analysis")
    ]
    
    failed_imports = []
    
    for lib_name, description in critical_libraries:
        try:
            __import__(lib_name)
            print(f"✅ {lib_name} - {description}")
        except ImportError as e:
            print(f"❌ {lib_name} - {description} (FAILED)")
            failed_imports.append(lib_name)
    
    if failed_imports:
        print(f"\n⚠️ {len(failed_imports)} libraries failed to import:")
        for lib in failed_imports:
            print(f"   - {lib}")
        return False
    else:
        print("\n✅ Tất cả libraries đã sẵn sàng!")
        return True

def run_quick_test():
    """Chạy test nhanh"""
    print("\n🚀 Chạy quick test...")
    
    try:
        # Import trading bot components
        import sys
        sys.path.append('.')
        
        from trading_bot import AdvancedFeatureEngineer, log_manager
        import pandas as pd
        import numpy as np
        
        print("✅ Core components imported successfully")
        
        # Test basic functionality
        engineer = AdvancedFeatureEngineer()
        print("✅ Feature engineer initialized")
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 2000,
            'high': np.random.randn(100).cumsum() + 2001,
            'low': np.random.randn(100).cumsum() + 1999,
            'close': np.random.randn(100).cumsum() + 2000,
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        # Ensure OHLC constraints
        test_data['high'] = test_data[['open', 'close', 'high']].max(axis=1)
        test_data['low'] = test_data[['open', 'close', 'low']].min(axis=1)
        
        # Engineer features
        featured = engineer.engineer_features(test_data, 'TEST')
        
        if not featured.empty:
            print(f"✅ Feature engineering test passed ({len(featured.columns)} features)")
        else:
            print("❌ Feature engineering test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    print("🎯 Bắt đầu quá trình setup...")
    
    steps = [
        ("Python Version Check", check_python_version),
        ("System Check", check_system),
        ("Install Dependencies", install_dependencies),
        ("Create Directories", create_directories),
        ("Create Env File", create_env_file),
        ("Test critical Imports", test_imports),
        ("Quick Functionality Test", run_quick_test)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"🔧 {step_name}")
        print(f"{'='*60}")
        
        try:
            result = step_func()
            if result is False:
                failed_steps.append(step_name)
                print(f"❌ {step_name} FAILED")
        except Exception as e:
            failed_steps.append(step_name)
            print(f"💥 {step_name} ERROR: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 SETUP SUMMARY")
    print(f"{'='*60}")
    
    success_count = len(steps) - len(failed_steps)
    total_count = len(steps)
    
    if failed_steps:
        print(f"❌ {len(failed_steps)} steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
    
    print(f"\n✅ {success_count}/{total_count} steps completed ({success_count/total_count*100:.1f}%)")
    
    if len(failed_steps) == 0:
        print("""
🎉 SETUP HOÀN THÀNH!
===================
✅ Bot đã sẵn sàng để chạy!

📋 NEXT STEPS:
1. Chỉnh sửa file .env.example thành .env
2. Cập nhật Discord webhook URL
3. Chạy test: python test_bot.py
4. Chạy bot: python trading_bot.py

⚠️ REMEMBER: Đây là bot demo, không trade với tiền thật!
        """)
    else:
        print("""
⚠️ SETUP KHÔNG HOÀN THÀNH
=========================
Một số bước đã thất bại. Kiểm tra:

🔍 Debug:
- Kiểm tra Python version (cần 3.8+)
- Verify internet connection
- Check pip permissions
- Review error messages above

💡 Solutions:
- Upgrade Python
- Restart terminal/IDE
- Run with admin privileges
- Check firewall settings
        """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Critical setup error: {e}")
        sys.exit(1)