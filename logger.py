"""
Advanced Logging System with Custom Formatting and Emoji Indicators
Visual and comprehensive logging for all bot activities
"""

import logging
import logging.handlers
import datetime
import json
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum
from config import LogLevel

class CustomFormatter(logging.Formatter):
    """Custom formatter with emoji indicators and colors"""
    
    # Color codes for different platforms
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green  
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Purple
        'TRADE': '\033[96m',     # Cyan
        'DATA': '\033[97m',      # White
        'ML': '\033[35m',        # Magenta
        'SYSTEM': '\033[33m',    # Orange
    }
    RESET_COLOR = '\033[0m'
    
    def __init__(self):
        super().__init__()
    
    def format(self, record):
        # Get emoji for log level
        emoji = self.get_emoji_for_level(record.levelname)
        
        # Get color for level
        color = self.COLORS.get(record.levelname, '')
        
        # Format timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Format record
        formatted_msg = f"{color}{emoji} {timestamp} | {record.levelname:8} | {record.name:20} | {record.getMessage()}{self.RESET_COLOR}"
        
        return formatted_msg
    
    def get_emoji_for_level(self, level_name: str) -> str:
        """Get emoji for log level"""
        emoji_map = {
            'DEBUG': LogLevel.DEBUG.value,
            'INFO': LogLevel.INFO.value,
            'WARNING': LogLevel.WARNING.value,
            'ERROR': LogLevel.ERROR.value,
            'CRITICAL': LogLevel.ERROR.value,
            'TRADE': LogLevel.TRADE.value,
            'DATA': LogLevel.DATA.value,
            'ML': LogLevel.ML.value,
            'SYSTEM': LogLevel.SYSTEM.value,
        }
        return emoji_map.get(level_name, "📝")

class LogManager:
    """Advanced log manager with filtering and analysis capabilities"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Store logs in memory for analysis
        self.in_memory_logs = []
        self.lock = threading.Lock()
        
        # Setup handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup all log handlers"""
        # Main log file
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading_bot.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log", 
            maxBytes=5*1024*1024,   # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        
        # Trade log file
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trades.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=10
        )
        
        # ML log file
        ml_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "ml_training.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())
        
        # JSON handler for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "structured.json",
            maxBytes=10*1024*1024,
            backupCount=3
        )
        
        # Custom JSON formatter
        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.handlers = {
            "console": console_handler,
            "main": main_handler,
            "error": error_handler,
            "trade": trade_handler,
            "ml": ml_handler,
            "json": json_handler
        }
    
    def setup_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """Setup logger with all handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add handlers
        logger.addHandler(self.handlers["console"])
        logger.addHandler(self.handlers["main"])
        
        # Add category-specific handlers
        if "trade" in name.lower() or "signal" in name.lower():
            logger.addHandler(self.handlers["trade"])
        
        if "ml" in name.lower() or "model" in name.lower():
            logger.addHandler(self.handlers["ml"])
        
        # All logs go to structured JSON
        logger.addHandler(self.handlers["json"])
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
            
        return json.dumps(log_entry)

class ModuleLogger:
    """Specialized logger factory for different modules"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.loggers = {}
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """Get or create logger for module"""
        if module_name not in self.loggers:
            self.loggers[module_name] = self.log_manager.setup_logger(module_name)
        return self.loggers[module_name]

class TradeLogger:
    """Specialized logger for trading activities"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_signal(self, symbol: str, signal: str, confidence: float, reason: str):
        """Log trading signal"""
        self.logger.log(
            logging.INFO,
            f"Signal Generated | {symbol} | {signal} | Confidence: {confidence:.2f}",
            extra={
                "category": "signal",
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "reason": reason
            }
        )
    
    def log_position_open(self, symbol: str, side: str, size: float, price: float, sl: float, tp: float):
        """Log position opening"""
        self.logger.log(
            logging.INFO,
            f"Position Opened | {symbol} | {side} | Size: {size} | Price: {price} | SL: {sl} | TP: {tp}",
            extra={
                "category": "position_open",
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "sl": sl,
                "tp": tp
            }
        )
    
    def log_position_close(self, symbol: str, side: str, size: float, entry_price: float, exit_price: float, pnl: float):
        """Log position closing"""
        self.logger.log(
            logging.INFO,
            f"Position Closed | {symbol} | {side} | Size: {size} | Entry: {entry_price} | Exit: {exit_price} | PnL: {pnl:.2f}",
            extra={
                "category": "position_close",
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl
            }
        )
    
    def log_sl_update(self, symbol: str, old_sl: float, new_sl: float, reason: str):
        """Log stop loss update"""
        self.logger.log(
            logging.INFO,
            f"Stop Loss Updated | {symbol} | Old: {old_sl} | New: {new_sl} | Reason: {reason}",
            extra={
                "category": "sl_update",
                "symbol": symbol,
                "old_sl": old_sl,
                "new_sl": new_sl,
                "reason": reason
            }
        )

# Global log manager instance
log_manager = LogManager()
module_logger = ModuleLogger(log_manager)

# Pre-configured loggers for common modules
bot_logger = module_logger.get_logger("TradingBot")
api_logger = module_logger.get_logger("APIManager")
data_logger = module_logger.get_logger("DataManager")
ml_logger = module_logger.get_logger("MLPredictor")
rl_logger = module_logger.get_logger("RLAgent")
risk_logger = module_logger.get_logger("RiskManager")
portfolio_logger = module_logger.get_logger("PortfolioManager")

# Specialized trade logger
trade_logger = TradeLogger(module_logger.get_logger("TradeLogger"))

def get_logger(name: str) -> logging.Logger:
    """Get logger for any module"""
    return module_logger.get_logger(name)