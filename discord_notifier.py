"""
Advanced Discord Notification System with Embedded Messages
Professional Discord integration with rich formatting, charts, and real-time updates
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from logger import get_logger
from config import DISCORD_WEBHOOK_URL

@dataclass 
class DiscordMessage:
    """Discord message structure"""
    title: str
    description: str
    color: int = 0x00ff00  # Green
    fields: List[Dict[str, Any]] = None
    footer: str = None
    timestamp: str = None
    thumbnail_url: str = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class DiscordNotifier:
    """Advanced Discord notification system"""
    
    def __init__(self, webhook_url: str = DISCORD_WEBHOOK_URL):
        self.webhook_url = webhook_url
        self.logger = get_logger("DiscordNotifier")
        self.session = None
        
        # Message types with custom formatting
        self.message_colors = {
            'trade_signal': 0xff6b35,      # Orange
            'position_open': 0x00ff00,     # Green
            'position_close': 0xff0000,    # Red
            'stop_loss_update': 0xffa500,  # Orange
            'system_alert': 0xff00ff,      # Magenta
            'performance_report': 0x00bfff, # Deep Sky Blue
            'error': 0x8b0000,             # Dark Red
            'info': 0x4169e1,              # Royal Blue
            'warning': 0xffd700             # Gold
        }
        
        # Emoji indicators
        self.emojis = {
            'buy': '📈',
            'sell': '📉',
            'hold': '⏸️',
            'close': '🔴',
            'profit': '💰',
            'loss': '❌',
            'warning': '⚠️',
            'success': '✅',
            'error': '🚨',
            'info': 'ℹ️',
            'bot': '🤖',
            'chart': '📊',
            'portfolio': '💼',
            'signal': '🎯',
            'risk': '⚠️',
            'news': '📰',
            'tech': '⚙️'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: DiscordMessage, content: str = None) -> bool:
        """Send Discord message"""
        try:
            embed = {
                "title": message.title,
                "description": message.description,
                "color": message.color,
                "fields": message.fields,
                "footer": {"text": message.footer} if message.footer else None,
                "timestamp": message.timestamp,
                "thumbnail": {"url": message.thumbnail_url} if message.thumbnail_url else None
            }
            
            payload = {
                "embeds": [embed],
                "username": "Advanced Trading Bot",
                "avatar_url": "https://cdn.discordapp.com/icons/123456789/avataurlink.png"  # Custom avatar
            }
            
            if content:
                payload["content"] = content
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:
                    self logger.debug("✅ Discord message sent successfully")
                    return True
                else:
                    self.logger.error(f"❌ Discord message failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Discord notification error: {e}")
            return False
    
    # Specific notification methods
    
    async def notify_signal(self, signal_data: Dict[str, Any]):
        """Notify about new trading signal"""
        
        symbol = signal_data['symbol']
        signal_type = signal_data['signal_type']
        confidence = signal_data['confidence']
        reason = signal_data['reason']
        risk_score = signal_data.get('risk_score', 0)
        
        emoji = self.emojis.get(signal_type.lower(), self.emojis['signal'])
        title = f"{emoji} Trading Signal: {symbol}"
        
        # Color based on signal type
        if signal_type == "BUY":
            color = 0x00ff00  # Green
        elif signal_type == "SELL":
            color = 0xff0000  # Red
        else:
            color = 0xffa500  # Orange
        
        # Confidence bar
        confidence_bar = self._create_progress_bar(confidence, 10, "█", "░")
        
        fields = [
            {
                "name": "🎯 Signal",
                "value": signal_type,
                "inline": True
            },
            {
                "name": "📊 Confidence",
                "value": f"{confidence:.1%} {confidence_bar}",
                "inline": True
            },
            {
                "name": "⚠️ Risk Score",
                "value": f"{risk_score:.2f}/1.0",
                "inline": True
            },
            {
                "name": "💡 Analysis",
                "value": reason[:1000],  # Discord limit
                "inline": False
            }
        ]
        
        # Add technical indicators if available
        if 'technical_data' in signal_data:
            tech_data = signal_data['technical_data']
            tech_summary = self._format_technical_summary(tech_data)
            fields.append({
                "name": "📈 Technical Analysis",
                "value": tech_summary,
                "inline": False
            })
        
        message = DiscordMessage(
            title=title,
            description=f"New {signal_type} signal generated for **{symbol}**",
            color=color,
            fields=fields,
            footer=f"Advanced Trading Bot • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await self.send_message(message)
    
    async def notify_position_open(self, position_data: Dict[str, Any]):
        """Notify about position opening"""
        
        symbol = position_data['symbol']
        side = position_data['side']
        size = position_data['size']
        price = position_data['price']
        sl = position_data['stop_loss']
        tp = position_data['take_profit']
        reason = position_data['reason']
        
        emoji = self.emojis['buy'] if side == "BUY" else self.emojis['sell']
        color = 0x00ff00 if side == "BUY" else 0xff0000
        
        # Calculate risk/reward ratio
        risk = abs(price - sl)
        reward = abs(tp - price)
        risk_reward = reward / risk if risk > 0 else 0
        
        fields = [
            {
                "name": "📍 Position",
                "value": f"{side} {size:.2f} units",
                "inline": True
            },
            {
                "name": "💰 Entry Price",
                "value": f"${price:.4f}",
                "inline": True
            },
            {
                "name": "🎯 R/R Ratio",
                "value": f"1:{risk_reward:.1f}",
                "inline": True
            },
            {
                "name": "🛡️ Stop Loss",
                "value": f"${sl:.4f}",
                "inline": True
            },
            {
                "name": "🎪 Take Profit",
                "value": f"${tp:.4f}",
                "inline": True
            },
            {
                "name": "💼 Risk Amount",
                "value": f"${risk * size:.2f}",
                "inline": True
            },
            {
                "name": "💭 Reason",
                "value": reason[:512],
                "inline": False
            }
        ]
        
        message = DiscordMessage(
            title=f"{emoji} Position Opened: {symbol}",
            description=f"Successfully opened **{symbol}** position",
            color=color,
            fields=fields,
            footer="Position opened"
        )
        
        await self.send_message(message)
    
    async def notify_position_close(self, position_data: Dict[str, Any]):
        """Notify about position closing"""
        
        symbol = position_data['symbol']
        side = position_data['side']
        entry_price = position_data['entry_price']
        exit_price = position_data['exit_price']
        pnl = position_data['pnl']
        reason = position_data['reason']
        duration = position_data.get('duration', 'N/A')
        
        is_profit = pnl > 0
        emoji = self.emojis['profit'] if is_profit else self.emojis['loss']
        color = 0x00ff00 if is_profit else 0xff0000
        
        # Calculate percentage return
        if side == "BUY":
            percentage = ((exit_price - entry_price) / entry_price) * 100
        else:
            percentage = ((entry_price - exit_price) / entry_price) * 100
        
        fields = [
            {
                "name": "📍 Position",
                "value": f"{side} {symbol}",
                "inline": True
            },
            {
                "name": "💰 P&L",
                "value": f"${pnl:.2f}",
                "inline": True
            },
            {
                "name": "📊 Return",
                "value": f"{percentage:+.2f}%",
                "inline": True
            },
            {
                "name": "🏁 Entry Price",
                "value": f"${entry_price:.4f}",
                "inline": True
            },
            {
                "name": "🚪 Exit Price",
                "value": f"${exit_price:.4f}",
                "inline": True
            },
            {
                "name": "⏱️ Duration",
                "value": duration,
                "inline": True
            },
            {
                "name": "💭 Reason",
                "value": reason[:512],
                "inline": False
            }
        ]
        
        # Add performance streak if consecutive results
        if 'streak' in position_data:
            streak_data = position_data['streak']
            streak_text = f"{streak_data['type']} streak: {streak_data['count']}"
            fields.append({
                "name": "🔥 Streak",
                "value": streak_text,
                "inline": True
            })
        
        message = DiscordMessage(
            title=f"{emoji} Position Closed: {symbol}",
            description=f"**{symbol}** position closed with {'profit' if is_profit else 'loss'} of ${pnl:.2f}",
            color=color,
            fields=fields,
            footer="Position closed"
        )
        
        await self.send_message(message)
    
    async def notify_stop_loss_update(self, update_data: Dict[str, Any]):
        """Notify about stop loss update"""
        
        symbol = update_data['symbol']
        old_sl = update_data['old_stop_loss']
        new_sl = update_data['new_stop_loss']
        current_price = update_data['current_price']
        pnl = update_data.get('pnl', 0)
        reason = update_data['reason']
        
        fields = [
            {
                "name": "📍 Symbol",
                "value": symbol,
                "inline": True
            },
            {
                "name": "💰 Current Price",
                "value": f"${current_price:.4f}",
                "inline": True
            },
            {
                "name": "📊 Unrealized PnL",
                "value": f"${pnl:.2f}",
                "inline": True
            },
            {
                "name": "🛡️ Old Stop Loss",
                "value": f"${old_sl:.4f}",
                "inline": True
            },
            {
                "name": "🆕 New Stop Loss",
                "value": f"${new_sl:.4f}",
                "inline": True
            },
            {
                "name": "📈 Improvement",
                "value": f"{((old_sl - new_sl) / old_sl * 100):+.1f}%" if old_sl != new_sl else "No change",
                "inline": True
            },
            {
                "name": "💭 Reason",
                "value": reason[:512],
                "inline": False
            }
        ]
        
        message = DiscordMessage(
            title=f"🎯 Stop Loss Updated: {symbol}",
            description=f"Stop loss improved for **{symbol}**",
            color=0xffa500,
            fields=fields,
            footer="Stop loss updated"
        )
        
        await self.send_message(message)
    
    async def notify_system_alert(self, alert_data: Dict[str, Any]):
        """Notify about system alerts"""
        
        alert_type = alert_data['alert_type']
        message_text = alert_data['message']
        severity = alert_data.get('severity', 'warning')
        
        emojis = {
            'error': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        colors = {
            'error': 0xff0000,
            'warning': 0xffa500,
            'info': 0x4169e1,
            'success': 0x00ff00
        }
        
        emoji = emojis.get(severity, emojis['info'])
        color = colors.get(severity, colors['info'])
        
        fields = [
            {
                "name": "🚨 Alert Type",
                "value": alert_type,
                "inline": True
            },
            {
                "name": "⚡ Severity",
                "value": severity.upper(),
                "inline": True
            }
        ]
        
        # Add additional data if available
        if 'additional_data' in alert_data:
            additional = alert_data['additional_data']
            for key, value in additional.items():
                fields.append({
                    "name": key.replace('_', ' ').title(),
                    "value": str(value),
                    "inline": True
                })
        
        fields.append({
            "name": "📝 Details",
            "value": message_text[:1000],
            "inline": False
        })
        
        message = DiscordMessage(
            title=f"{emoji} System Alert",
            description=f"{severity.title()} alert from Advanced Trading Bot",
            color=color,
            fields=fields,
            footer="System notification"
        )
        
        await self.send_message(message)
    
    async def notify_performance_report(self, portfolio_data: Dict[str, Any]):
        """Notify about portfolio performance"""
        
        total_value = portfolio_data['total_value']
        total_pnl = portfolio_data['total_pnl']
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        win_rate = portfolio_data.get('win_rate', 0)
        sharpe_ratio = portfolio_data.get('sharpe_ratio', 0)
        max_drawdown = portfolio_data['max_drawdown']
        position_count = portfolio_data['position_count']
        
        # Performance trend emoji
        if daily_pnl > 0:
            trend_emoji = self.emojis['profit']
        elif daily_pnl < 0:
            trend_emoji = self.emojis['loss']
        else:
            trend_emoji = self.emojis['hold']
        
        title = f"{trend_emoji} Portfolio Performance Report"
        
        fields = [
            {
                "name": "💼 Total Value",
                "value": f"${total_value:,.2f}",
                "inline": True
            },
            {
                "name": "📊 Total P&L",
                "value": f"${total_pnl:+,.2f}",
                "inline": True
            },
            {
                "name": "📈 Daily P&L",
                "value": f"${daily_pnl:+,.2f}",
                "inline": True
            },
            {
                "name": "🎯 Win Rate",
                "value": f"{win_rate:.1f}%",
                "inline": True
            },
            {
                "name": "📊 Sharpe Ratio",
                "value": f"{sharpe_ratio:.2f}",
                "inline": True
            },
            {
                "name": "📉 Max Drawdown",
                "value": f"{max_drawdown:.1f}%",
                "inline": True
            },
            {
                "name": "📋 Open Positions",
                "value": f"{position_count}",
                "inline": True
            }
        ]
        
        # Add position breakdown
        if 'positions' in portfolio_data:
            positions = portfolio_data['positions']
            position_summary = self._format_position_summary(positions)
            fields.append({
                "name": "💼 Position Summary",
                "value": position_summary,
                "inline": False
            })
        
        message = DiscordMessage(
            title=title,
            description=f"📊 Portfolio performance update",
            color=0x00bfff,
            fields=fields,
            footer="Performance report"
        )
        
        await self.send_message(message)
    
    async def notify_news_sentiment(self, news_data: Dict[str, Any]):
        """Notify about news sentiment analysis"""
        
        symbol = news_data['symbol']
        articles = news_data['articles']
        avg_sentiment = news_data['avg_sentiment']
        relevance_score = news_data['relevance_score']
        key_headlines = news_data.get('key_headlines', [])
        
        sentiment_emoji = "📈" if avg_sentiment > 0 else "📉" if avg_sentiment < 0 else "➡️"
        sentiment_color = 0x00ff00 if avg_sentiment > 0 else 0xff0000 if avg_sentiment < 0 else 0xffa500
        
        fields = [
            {
                "name": "📊 Articles Analyzed",
                "value": str(len(articles)),
                "inline": True
            },
            {
                "name": "🎭 Avg Sentiment",
                "value": f"{avg_sentiment:.2f}",
                "inline": True
            },
            {
                "name": "🎯 Relevance Score",
                "value": f"{relevance_score:.2f}",
                "inline": True
            }
        ]
        
        # Add key headlines
        if key_headlines:
            headlines_text = "\n".join([f"• {headline[:100]}..." for headline in key_headlines[:3]])
            fields.append({
                "name": "📰 Key Headlines",
                "value": headlines_text,
                "inline": False
            })
        
        message = DiscordMessage(
            title=f"{sentiment_emoji} News Sentiment: {symbol}",
            description=f"News analysis for **{symbol}** completed",
            color=sentiment_color,
            fields=fields,
            footer="News sentiment analysis"
        )
        
        await self.send_message(message)
    
    def _create_progress_bar(self, value: float, length: int = 10, fill: str = "█", empty: str = "░") -> str:
        """Create a visual progress bar"""
        filled_length = int(value * length)
        return fill * filled_length + empty * (length - filled_length)
    
    def _format_technical_summary(self, tech_data: Dict[str, Any]) -> str:
        """Format technical analysis summary"""
        summary_parts = []
        
        if 'RSI' in tech_data:
            rsi = tech_data['RSI']
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            summary_parts.append(f"RSI: {rsi:.1f} ({rsi_status})")
        
        if 'MACD' in tech_data:
            macd_signal = tech_data.get('MACD_signal', 'N/A')
            summary_parts.append(f"MACD: {macd_signal}")
        
        if 'ADX' in tech_data:
            adx = tech_data['ADX']
            trend_strength = "Strong" if adx > 25 else "Weak"
            summary_parts.append(f"ADX: {adx:.1f} ({trend_strength})")
        
        return " | ".join(summary_parts)[:200]
    
    def _format_position_summary(self, positions: Dict[str, Any]) -> str:
        """Format position summary"""
        if not positions:
            return "No open positions"
        
        summary_parts = []
        for symbol, pos_data in positions.items():
            side_emoji = "📈" if pos_data['side'] == "BUY" else "📉"
            pnl_emoji = "💰" if pos_data['pnl'] > 0 else "❌"
            summary_parts.append(f"{symbol}: {side_emoji}{pos_data['side']} {pnl_emoji}${pos_data['pnl']:.2f}")
        
        return "\n".join(summary_parts)[:500]
    
    async def send_test_message(self):
        """Send test message to verify Discord integration"""
        message = DiscordMessage(
            title="🤖 Discord Integration Test",
            description="Advanced Trading Bot Discord notifications are working correctly!",
            color=0x00ff00,
            fields=[
                {
                    "name": "✅ Status",
                    "value": "Operational",
                    "inline": True
                },
                {
                    "name": "⏰ Time",
                    "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "inline": True
                },
                {
                    "name": "🚀 Ready",
                    "value": "Bot is ready to send trading notifications",
                    "inline": False
                }
            ],
            footer="System test"
        )
        
        success = await self.send_message(message)
        if success:
            self.logger.info("✅ Discord integration test successful")
        else:
            self.logger.error("❌ Discord integration test failed")
        
        return success
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """Send daily performance summary"""
        
        date = datetime.now().strftime('%Y-%m-%d')
        total_trades = summary_data.get('total_trades', 0)
        winning_trades = summary_data.get('winning_trades', 0)
        total_pnl = summary_data.get('total_pnl', 0)
        best_trade = summary_data.get('best_trade', 0)
        worst_trade = summary_data.get('worst_trade', 0)
        most_profitable_symbol = summary_data.get('most_profitable_symbol', 'N/A')
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        fields = [
            {
                "name": "📅 Date",
                "value": date,
                "inline": True
            },
            {
                "name": "🎯 Total Trades",
                "value": str(total_trades),
                "inline": True
            },
            {
                "name": "🏆 Win Rate",
                "value": f"{win_rate:.1f}%",
                "inline": True
            },
            {
                "name": "💰 Total P&L",
                "value": f"${total_pnl:+,.2f}",
                "inline": True
            },
            {
                "name": "🥇 Best Trade",
                "value": f"${best_trade:.2f}",
                "inline": True
            },
            {
                "name": "🥉 Worst Trade",
                "value": f"${worst_trade:.2f}",
                "inline": True
            },
            {
                "name": "⭐ Top Performer",
                "value": most_profitable_symbol,
                "inline": True
            }
        ]
        
        message = DiscordMessage(
            title=f"📊 Daily Summary - {date}",
            description=f"Trading performance summary for {date}",
            color=0x00bfff,
            fields=fields,
            footer="Daily report"
        )
        
        await self.send_message(message)

# Global Discord notifier instance
discord_notifier = DiscordNotifier()