# -*- coding: utf-8 -*-
"""
Telegram Notification Module for Sniper Bot Pro.
Sends trading signals and alerts to a Telegram channel/chat.

Setup Instructions:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Get your chat ID (send a message to @userinfobot)
4. Add your credentials to config.py
"""

import urllib.request
import urllib.parse
import json
import logging
import threading

logger = logging.getLogger('SniperBot.Telegram')


class TelegramNotifier:
    """Sends notifications to Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Your Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if self.enabled:
            logger.info("✅ Telegram notifications enabled")
        else:
            logger.info("ℹ️ Telegram notifications disabled (no credentials)")
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """
        Send a message to Telegram (async, non-blocking).
        
        Args:
            message: The message text to send
            parse_mode: 'HTML' or 'Markdown'
        """
        if not self.enabled:
            return
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(
            target=self._send_message_sync,
            args=(message, parse_mode),
            daemon=True
        )
        thread.start()
    
    def _send_message_sync(self, message: str, parse_mode: str):
        """Internal sync method to send message."""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            encoded_data = urllib.parse.urlencode(data).encode('utf-8')
            req = urllib.request.Request(url, data=encoded_data, method='POST')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                if not result.get('ok'):
                    logger.warning(f"Telegram API error: {result}")
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_signal(self, symbol: str, side: str, price: float, tp: float, sl: float, score: int):
        """Send a trading signal notification."""
        emoji = "🟢" if side == "BUY" else "🔴"
        direction = "LONG" if side == "BUY" else "SHORT"
        
        message = f"""
{emoji} <b>NEW SIGNAL: {symbol}</b>

📊 Direction: <b>{direction}</b>
💰 Entry: <code>${price:,.2f}</code>
🎯 TP: <code>${tp:,.2f}</code>
🛑 SL: <code>${sl:,.2f}</code>
📈 Score: <b>{score}/7</b>

⏰ {self._get_time()}
"""
        self.send_message(message)
    
    def send_trade_result(self, symbol: str, side: str, pnl: float, reason: str):
        """Send trade result notification."""
        is_win = pnl > 0
        emoji = "✅" if is_win else "❌"
        result = "WIN" if is_win else "LOSS"
        pnl_sign = "+" if pnl > 0 else ""
        
        message = f"""
{emoji} <b>TRADE CLOSED: {symbol}</b>

📊 Direction: {side}
💵 PnL: <b>{pnl_sign}${pnl:.2f}</b>
📝 Reason: {reason}
🏆 Result: <b>{result}</b>

⏰ {self._get_time()}
"""
        self.send_message(message)
    
    def send_daily_limit_hit(self, daily_pnl: float):
        """Send daily limit warning."""
        message = f"""
🛑 <b>DAILY LOSS LIMIT REACHED</b>

💸 Daily PnL: <code>${daily_pnl:.2f}</code>
⚠️ Trading paused until tomorrow

Stay disciplined! 💪
"""
        self.send_message(message)
    
    def send_startup(self):
        """Send bot startup notification."""
        message = """
🚀 <b>SNIPER BOT PRO</b>

✅ Bot is now online and monitoring:
• BTC/USDT
• ETH/USDT

📊 Mode: Paper Trading
🔔 Alerts: Enabled
"""
        self.send_message(message)
    
    def _get_time(self):
        """Get current time string."""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


# Singleton instance - will be initialized from main app
_notifier = None

def get_notifier() -> TelegramNotifier:
    """Get the global notifier instance."""
    global _notifier
    return _notifier

def init_notifier(bot_token: str, chat_id: str, enabled: bool = True):
    """Initialize the global notifier."""
    global _notifier
    _notifier = TelegramNotifier(bot_token, chat_id, enabled)
    return _notifier
