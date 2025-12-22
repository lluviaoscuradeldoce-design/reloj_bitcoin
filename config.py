# -*- coding: utf-8 -*-
"""
Configuration file for Sniper Bot Pro.
Edit these values to customize bot behavior.
"""

# ============ TRADING SCHEDULE (PHASE 7) ============
SCHEDULE_CONFIG = {
    'enabled': True,                     # Restrict trading to specific hours
    'windows': [
        {'start': '13:00', 'end': '21:00'}, # NY Session (approx UTC)
        {'start': '07:00', 'end': '16:00'}, # London Session (approx UTC)
    ],
    'timezone': 'UTC'                    # Reference timezone
}

# ============ TRADING CONFIGURATION ============
TRADING_CONFIG = {
    'symbols': ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX'],
    'risk_per_trade': 0.5,               # 50% risk (AGGRESSIVE MODE)
    'max_leverage': 10,                  # Max 10x leverage cap
    'max_active_trades': 2,              # Limit concurrent exposure
    'daily_loss_limit': -50.0,           # Max daily loss in USD ($50)
    'signal_cooldown': 300,              # 5 minutes between signals
    'alert_cooldown': 300,               # 5 minutes between sound alerts
}

# ============ SIGNAL THRESHOLDS ============
SIGNAL_CONFIG = {
    'min_score_long': 5,                 # Stricter: Requires Trend Alignment
    'min_score_short': -5,               # Stricter: Requires Trend Alignment
    'atr_sl_multiplier': 2.0,            # ATR multiplier for Stop Loss (Optimized)
    'atr_tp_multiplier': 3.0,            # ATR multiplier for Take Profit (Optimized)
}

# ============ LIQUIDITY & MM STRATEGY (PHASE 10) ============
LIQUIDITY_CONFIG = {
    'vpin_threshold': 0.75,              # Flow Toxicity: >0.75 indicates MM withdrawal
    'liq_mass_threshold': 150000,        # Min USD volume to consider "Massive Liquidation"
    'liq_exhaustion_time': 30,           # Seconds to track liquidation clusters
    'abs_cvd_threshold': 0.8,            # 80% absorption ratio (Price vs CVD)
    'sweep_lookback': 20,                # Candles to look back for Liquidity Sweep
}

# ============ TRAILING STOP LEVELS ============
TRAILING_CONFIG = {
    'level_1_activation': 0.3,           # ATR profit to activate Break Even
    'level_2_activation': 1.0,           # ATR profit to lock 25%
    'level_2_lock_percent': 0.25,        # Percentage of profit to lock
    'level_3_activation': 1.5,           # ATR profit to start trailing
    'level_3_trail_atr': 0.5,            # ATR distance for trailing
}

# ============ INDICATOR SETTINGS ============
INDICATOR_CONFIG = {
    'rsi_period': 14,
    'stoch_rsi_period': 14,
    'macd_fast': 8,
    'macd_slow': 17,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std_dev': 2,
    'atr_period': 14,
    'ema_period': 200,
}

# ============ SCORING THRESHOLDS ============
SCORING_CONFIG = {
    'stoch_oversold': 25,                # StochRSI below = bullish
    'stoch_overbought': 75,              # StochRSI above = bearish
    'obi_bullish': 1.15,                 # OBI above = bullish
    'obi_bearish': 0.85,                 # OBI below = bearish
    'cvd_threshold': 500000,             # Significant CVD delta
}

# ============ UI CONFIGURATION ============
UI_CONFIG = {
    'bg_color': '#1a1a1a',
    'text_color': '#ffffff',
    'positive_color': '#00ff00',
    'negative_color': '#ff4444',
    'warning_color': '#ffff00',
    'disabled_color': '#ff00ff',
    'font_family': 'Segoe UI',
    'font_size_price': 120,
    'font_size_signal': 30,
    'font_size_stats': 20,
}

# ============ WEBSOCKET CONFIGURATION ============
WEBSOCKET_CONFIG = {
    'ping_interval': 60,
    'ping_timeout': 30,
    'reconnect_delay': 5,
}

# ============ TELEGRAM NOTIFICATIONS ============
# To enable Telegram alerts:
# 1. Create a bot with @BotFather and get the token
# 2. Get your chat ID from @userinfobot
# 3. Fill in the values below
TELEGRAM_CONFIG = {
    'enabled': False,                     # Set to True to enable
    'bot_token': '',                      # Your bot token from @BotFather
    'chat_id': '',                        # Your chat ID
    'notify_signals': True,               # Notify on new signals
    'notify_trades': True,                # Notify on trade close
    'notify_daily_limit': True,           # Notify when daily limit hit
}

# ============ ADDITIONAL TRADING PAIRS ============
# Add more pairs here - they will be monitored if added to TRADING_CONFIG['symbols']
AVAILABLE_PAIRS = {
    'BTC': {'name': 'Bitcoin', 'decimals': 2},
    'ETH': {'name': 'Ethereum', 'decimals': 2},
    'SOL': {'name': 'Solana', 'decimals': 2},
    'BNB': {'name': 'Binance Coin', 'decimals': 2},
    'XRP': {'name': 'Ripple', 'decimals': 4},
    'DOGE': {'name': 'Dogecoin', 'decimals': 5},
    'ADA': {'name': 'Cardano', 'decimals': 4},
    'AVAX': {'name': 'Avalanche', 'decimals': 2},
    'LINK': {'name': 'Chainlink', 'decimals': 2},
    'DOT': {'name': 'Polkadot', 'decimals': 2},
}
