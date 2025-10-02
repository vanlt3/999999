#!/usr/bin/env python3
"""
Simple launcher script for the Advanced Trading Bot
Easy-to-use entry point with command line options
"""

import argparse
import asyncio
import sys
from datetime import datetime
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot import TradingBot

def get_argument_texts():
    """Translator helper for argument texts"""
    return {
        'desc': '🚀 Khởi động Advanced Trading Bot với các tùy chọn tùy chỉnh',
        'capital': 'Số vốn ban đầu cho trading (mặc định: 10000 USD)',
        'interval': 'Thời gian giữa các lần giao dịch trong giây (mặc định: 300)',
        'symbols': 'Danh sách symbols để trade (phân cách bằng dấu phẩy)',
        'mode': 'Chế độ chạy: demo (paper trading) hoặc live',
        'test': 'Chạy test mode - chỉ kiểm tra các components mà không trade',
        'verbose': 'Bật verbose logging để debug',
    }

def main():
    """Main entry point with CLI options"""
    
    texts = get_argument_texts()
    
    parser = argparse.ArgumentParser(
        description=texts['desc'],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Ví dụ sử dụng:

  # Chạy với cấu hình mặc định
  python run_bot.py

  # Chạy với vốn 5000 USD và cycle interval 60 giây
  python run_bot.py --capital 5000 --interval 60

  # Trade chỉ EURUSD và BTCUSD
  python run_bot.py --symbols EURUSD,BTCUSD

  # Test mode để kiểm tra components
  python run_bot.py --test

  # Live mode với verbose logging
  python run_bot.py --mode live --verbose

⚠️  Cảnh báo: Luôn test với paper trading trước khi sử dụng với tiền thật!
        """
    )
    
    parser.add_argument(
        '--capital', 
        type=float, 
        default=10000.0,
        help=texts['capital']
    )
    
    parser.add_argument(
        '--interval', 
        type=int, 
        default=300,
        help=texts['interval']
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default="EURUSD,XAUUSD,BTCUSD,NASDAQ100",
        help=texts['symbols']
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'live'],
        default='demo',
        help=texts['mode']
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help=texts['test']
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help=texts['verbose']
    )
    
    args = parser.parse_args()
    
    # Setup logging level if verbose
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    print("🤖 Advanced Trading Bot")
    print("=" * 50)
    print(f"📅 Ngày khởi động: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Vốn ban đầu: ${args.capital:,.2f}")
    print(f"⏱️  Cycle Interval: {args.interval} giây")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"🎮 Chế độ: {'DEMO' if args.mode == 'demo' else 'LIVE'}")
    print("=" * 50)
    
    if args.mode == 'live':
        print("⚠️  CẢNH BÁO: Đang chạy ở chế độ LIVE với tiền thật!")
        print("⚠️  Đảm bảo đã test kỹ với paper trading!")
        print("⚠️  Bạn có chắc muốn tiếp tục? (Ctrl+C để hủy)")
        
        try:
            input("Nhấn Enter để tiếp tục...")
        except KeyboardInterrupt:
            print("\n❌ Đã hủy khởi động bot")
            return 1
    
    async def run_bot():
        """Actual bot runner"""
        try:
            # Initialize and configure bot
            bot = TradingBot(initial_capital=args.capital)
            
            # Override symbols if provided
            if args.symbols != "EURUSD,XAUUSD,BTCUSD,NASDAQ100":
                bot.symbols = symbols
            
            # Test mode - just initialize components
            if args.test:
                print("\n🧪 Chạy test mode...")
                success = await bot.initialize()
                if success:
                    print("✅ Tất cả components đã sẵn sàng!")
                    
                    # Show bot status
                    status = bot.get_bot_status()
                    print(f"\n📊 Bot Status:")
                    print(f"   Models Ready: {status['models_ready']}")
                    print(f"   Portfolio Value: ${status['portfolio_value']:,.2f}")
                    print(f"   Open Positions: {status['open_positions']}")
                    print(f"   Symbols Count: {status['symbols_count']}")
                    
                    return 0
                else:
                    print("❌ Bot initialization failed!")
                    return 1
            
            # Run in demo or live mode
            mode_text = "paper trading" if args.mode == 'demo' else "live trading"
            print(f"\n🚀 Khởi động bot với {mode_text}...")
            
            await blog.start_trading(cycle_interval=args.interval)
            
        except KeyboardInterrupt:
            print("\n⏹️  Bot được dừng bởi người dùng")
            return 0
        except Exception as e:
            print(f"\n❌ Bot bị lỗi: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Run the bot
    try:
        exit_code = asyncio.run(run_bot())
        return exit_code
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
        return 0

if __name__ == "__main__":
    exit(main())