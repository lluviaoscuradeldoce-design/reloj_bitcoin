import tkinter as tk
import urllib.request
import urllib.error
import json
import threading
import time
import winsound
import math
import os
import logging
import websocket # pip install websocket-client
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress annoying sklearn warnings about parallel execution
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SniperBot')

# --- IMPORT CONFIG AND NOTIFICATIONS ---
try:
    from config import (
        TELEGRAM_CONFIG, TRADING_CONFIG, WEBSOCKET_CONFIG, 
        LIQUIDITY_CONFIG, SCORING_CONFIG, TRAILING_CONFIG, SIGNAL_CONFIG,
        AVAILABLE_PAIRS, SCHEDULE_CONFIG
    )
    from notifications import init_notifier, get_notifier
    from sentiment_engine import sentiment_engine
    
    # Try importing ML Trainer
    try:
        from ml_trainer import MLTrainer
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        logger.warning("ML Libraries not found. Running in Technical Mode only.")
    
    # Initialize Telegram notifier
    telegram = init_notifier(
        bot_token=TELEGRAM_CONFIG.get('bot_token', ''),
        chat_id=TELEGRAM_CONFIG.get('chat_id', ''),
        enabled=TELEGRAM_CONFIG.get('enabled', False)
    )
except ImportError as e:
    logger.warning(f"Could not import config/notifications: {e}")
    telegram = None

# --- PAPER TRADING ENGINE WITH ADVANCED RISK MANAGEMENT ---
class PaperTrader:
    def __init__(self):
        self.balance = 100.0 
        self.initial_balance = 100.0
        self.positions = [] 
        self.history = [] 
        self.csv_file = "training_data.csv"
        
        # Daily Risk Management
        self.daily_pnl = 0.0
        self.daily_loss_limit = -20.0  # Max -$20 per day (20% of account)
        self.trading_enabled = True
        self.last_reset_date = time.strftime('%Y-%m-%d')
        
        self._init_csv()
    
    def _init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w") as f:
                f.write("symbol,side,rsi,stoch_rsi,macd_hist,atr,obi,cvd,vol_ratio,vpin,liq_vol,funding,oi,sentiment,score,outcome,pnl_percent\n")
    
    def check_daily_reset(self):
        """Reset daily PnL at midnight."""
        today = time.strftime('%Y-%m-%d')
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.trading_enabled = True
            self.last_reset_date = today
            return True
        return False
    
    def can_trade(self):
        """Check if trading is allowed (daily limit + max positions)."""
        self.check_daily_reset()
        
        # Check daily loss limit
        if not self.trading_enabled:
            return False
            
        # Check concurrent positions limit
        max_trades = TRADING_CONFIG.get('max_active_trades', 5)
        if len(self.positions) >= max_trades:
            return False
            
        return True

    def open_position(self, symbol, side, entry_price, sl, tp, features, atr):
        if not self.can_trade():
            return None
            
        risk_amt = self.balance * TRADING_CONFIG['risk_per_trade']  # Dynamic risk from config
        dist = abs(entry_price - sl)
        if dist == 0: return None
        
        size_qty = risk_amt / dist
        
        # Leverage Cap (e.g., max 10x)
        max_lev = TRADING_CONFIG.get('max_leverage', 10)
        max_notional = self.balance * max_lev
        notional = size_qty * entry_price
        
        if notional > max_notional:
            size_qty = max_notional / entry_price
            logger.info(f"⚡ LEVERAGE CAP: Position resized to {max_lev}x leverage.")
        
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry_price,
            'size': size_qty,
            'margin': risk_amt,        # Margin used
            'notional': notional,      # Total position size
            'sl': sl,
            'tp': tp,
            'original_sl': sl,  # Keep original for reference
            'atr': atr,
            'trailing_level': 0,  # 0=None, 1=BreakEven, 2=Lock25%, 3=Trail
            'trailing_active': False,
            'time': time.strftime('%H:%M:%S'),
            'pnl': 0.0,
            'features': features 
        }
        self.positions.append(trade)
        return trade

    def update(self, current_prices):
        closed = []
        for p in self.positions:
            sym = p['symbol']
            if sym not in current_prices: continue
            
            curr = current_prices[sym]
            atr = p['atr']
            entry = p['entry']
            
            # Calculate profit in ATR units
            if p['side'] == 'BUY':
                pnl = (curr - entry) * p['size']
                profit_atr = (curr - entry) / atr if atr > 0 else 0
            else:  # SELL
                pnl = (entry - curr) * p['size']
                profit_atr = (entry - curr) / atr if atr > 0 else 0
            
            # --- MULTI-LEVEL TRAILING STOP ---
            new_sl = p['sl']
            
            if profit_atr >= TRAILING_CONFIG['level_3_activation'] and p['trailing_level'] < 3:
                # LEVEL 3: Aggressive Trail
                p['trailing_level'] = 3
                p['trailing_active'] = True
                if p['side'] == 'BUY':
                    new_sl = curr - (TRAILING_CONFIG['level_3_trail_atr'] * atr)
                else:
                    new_sl = curr + (TRAILING_CONFIG['level_3_trail_atr'] * atr)
                    
            elif profit_atr >= TRAILING_CONFIG['level_2_activation'] and p['trailing_level'] < 2:
                # LEVEL 2: Lock profit
                p['trailing_level'] = 2
                p['trailing_active'] = True
                if p['side'] == 'BUY':
                    new_sl = entry + (TRAILING_CONFIG['level_2_lock_percent'] * atr)
                else:
                    new_sl = entry - (TRAILING_CONFIG['level_2_lock_percent'] * atr)
                    
            elif profit_atr >= TRAILING_CONFIG['level_1_activation'] and p['trailing_level'] < 1:
                # LEVEL 1: Break Even
                p['trailing_level'] = 1
                p['trailing_active'] = True
                new_sl = entry
            
            # Continue trailing if Level 3 is active
            if p['trailing_level'] == 3:
                if p['side'] == 'BUY':
                    trail_sl = curr - (TRAILING_CONFIG['level_3_trail_atr'] * atr)
                    if trail_sl > p['sl']:
                        new_sl = trail_sl
                else:
                    trail_sl = curr + (TRAILING_CONFIG['level_3_trail_atr'] * atr)
                    if trail_sl < p['sl']:
                        new_sl = trail_sl
            
            # Update SL (only if better)
            if p['side'] == 'BUY' and new_sl > p['sl']:
                p['sl'] = new_sl
            elif p['side'] == 'SELL' and new_sl < p['sl']:
                p['sl'] = new_sl
            
            # Check TP/SL hits
            if p['side'] == 'BUY':
                hit_tp = curr >= p['tp']
                hit_sl = curr <= p['sl']
            else:
                hit_tp = curr <= p['tp']
                hit_sl = curr >= p['sl']
            
            p['pnl'] = pnl
            
            if hit_tp or hit_sl:
                level_names = {0: "Initial", 1: "BE", 2: "Lock25", 3: "Trail"}
                reason = "TP Hit" if hit_tp else f"{level_names[p['trailing_level']]} SL"
                p['reason'] = reason
                p['close_price'] = curr
                self.balance += pnl
                self.daily_pnl += pnl
                
                # Check daily loss limit
                if self.daily_pnl <= self.daily_loss_limit:
                    self.trading_enabled = False
                
                self.history.append(p)
                closed.append(p)
                self.save_to_dataset(p, hit_tp or pnl > 0)
        
        for c in closed:
            self.positions.remove(c)
            
        return closed 

    def save_to_dataset(self, trade, is_win):
        f = trade['features']
        outcome = 1 if is_win else 0
        pnl_pct = (trade['pnl'] / self.balance) * 100 
        score = f.get('score', 0)
        
        # Extended Header with Alpha Features (Institutional Grade)
        row = f"{trade['symbol']},{trade['side']},{f['rsi']:.2f}," \
              f"{f['stoch_rsi']:.2f},{f['macd_hist']:.4f},{f['atr']:.2f}," \
              f"{f['obi']:.2f},{f['cvd']:.2f},{f['vol_ratio']:.2f}," \
              f"{f.get('vpin', 0):.2f},{f.get('liq_vol', 0):.2f}," \
              f"{f.get('funding', 0):.6f},{f.get('oi', 0):.0f}," \
              f"{f.get('sentiment', 0):.2f}," \
              f"{score},{outcome},{pnl_pct:.4f}\n"
              
        with open(self.csv_file, "a") as file:
            file.write(row)



class CryptoWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Sniper Bot Pro")
        self.running = True  # Initialize before starting any threads
        
        # Responsive UI - Get screen dimensions
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#1a1a1a')
        self.root.attributes('-topmost', True)
        
        # Store screen size for responsive elements
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.root.bind('<B1-Motion>', self.do_move)
        self.root.bind('<Button-1>', self.start_move)
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        self.root.bind('<m>', self.minimize_app) 
        
        self.log_file = "trading_log.txt"
        self.trader = PaperTrader()
        
        # Initialize data structures dynamically from config
        self.data = {}
        self.liq_map = {}
        self.last_signal_time = {}
        
        for symbol in TRADING_CONFIG['symbols']:
            self.data[symbol] = self.init_coin_state()
            self.liq_map[symbol] = []
            self.last_signal_time[symbol] = {'BUY': 0, 'SELL': 0}

        self.last_alert_time = 0
        self.alert_cooldown = 300 
        self.net_error_count = 0  
        self.net_healthy = True
        self.net_lock = threading.Lock() # Protect network counter mode:AGENT_MODE_EXECUTION        
        self.sentiment = sentiment_engine
        self.sentiment.start()
        
        # --- PHASE 11: AUTONOMOUS RELIABILITY ---
        self.last_heartbeat = time.time()
        self.watchdog_thread = threading.Thread(target=self.watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        
        self.ml_model = None
        if ML_AVAILABLE:
            self.ml_trainer = MLTrainer()
            if self.ml_trainer.load_model("trading_model.pkl"):
                self.ml_model = self.ml_trainer
                # Set single thread for inference to avoid joblib warnings in loops
                if hasattr(self.ml_model.model, 'n_jobs'):
                    self.ml_model.model.n_jobs = 1
                logger.info("🧠 ML Model Loaded Successfully (Inference Mode: Stable)")
            else:
                logger.warning("ML Model not found. Train it using ml_trainer.py")
        
        self.setup_ui()
        
        # Threads launched only after setup is ready
        self.analytics_thread = threading.Thread(target=self.analytics_loop, daemon=True)
        self.analytics_thread.start()
        self.ws_thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.ws_thread.start()

    def init_coin_state(self):
        return {
            'price': 0.0, 'mark': 0.0, 
            'rsi': 50, 'stoch_rsi': 50, 'vol_ratio': 1.0, 
            'funding': 0.0, 'oi': 0.0, 
            'trend_15m': 'FLAT', 'trend_1h': 'FLAT',
            'ema200_15m': 0.0, 'ema200_1h': 0.0,
            'obi': 1.0, 'atr': 0.0,
            'macd': {'hist': 0, 'line': 0, 'sig': 0},
            'bb': {'upper': 0, 'lower': 0, 'mid': 0},
            'cvd': 0.0,
            # --- PHASE 10: LIQUIDITY & MM ---
            'liq_vol_1m': 0.0,
            'vpin': 0.0,
            'buy_vol_bucket': 0.0,
            'sell_vol_bucket': 0.0,
            'last_sweep': 'NONE',
            'last_h_l_update': 0,
            'last_analytic_fetch': 0
        }

    def watchdog_loop(self):
        """Monitor threads and ensure system health."""
        while self.running:
            time.sleep(60)
            elapsed = time.time() - self.last_heartbeat
            if elapsed > 120:  # If analytics_loop stalled for > 2 mins
                logger.error(f"🚑 WATCHDOG: Analytics loop stalled (Last seen {elapsed:.0f}s ago). Attempting recovery...")
                # We can't easily kill threads, but we can restart important ones if needed
                # For now, we log it and try to force a heartbeat update to resume
                # Real production bots might even trigger a restart of the process
            
            # Hourly health log
            if int(time.time()) % 3600 < 60:
                logger.info("💓 HEARTBEAT: Bot is alive and monitoring 8 pairs.")

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, bg='#1a1a1a')
        self.main_container.pack(fill='both', expand=True)

        # --- RIGHT PANEL (LOG) --- (Packed FIRST to reserve space)
        log_width = max(550, int(self.screen_w * 0.30))  # 30% of screen, min 550px
        self.log_frame = tk.Frame(self.main_container, bg='#111111', width=log_width)
        self.log_frame.pack(side='right', fill='y')
        self.log_frame.pack_propagate(False) 
        
        tk.Label(self.log_frame, text="📋 TRANSACTIONS", font=('Segoe UI', 13, 'bold'), bg='#111111', fg='#888888').pack(pady=8)
        
        self.log_list = tk.Listbox(self.log_frame, bg='#111111', fg='#00ff00', font=('Consolas', 10), bd=0, highlightthickness=0)
        self.log_list.pack(fill='both', expand=True, padx=8, pady=5)
        
        # --- STATS PANEL (Below log) ---
        stats_panel = tk.Frame(self.log_frame, bg='#0d0d0d')
        stats_panel.pack(fill='x', padx=10, pady=10)
        
        tk.Label(stats_panel, text="📊 STATISTICS", font=('Segoe UI', 12, 'bold'), bg='#0d0d0d', fg='#666').pack(pady=5)
        
        # Stats Labels
        self.stat_winrate = tk.Label(stats_panel, text="Win Rate: 0%", font=('Consolas', 12), bg='#0d0d0d', fg='#aaa')
        self.stat_winrate.pack(anchor='w', padx=8, fill='x')
        
        self.stat_trades = tk.Label(stats_panel, text="Trades: 0 (0W / 0L)", font=('Consolas', 12), bg='#0d0d0d', fg='#aaa')
        self.stat_trades.pack(anchor='w', padx=8, fill='x')
        
        self.stat_pf = tk.Label(stats_panel, text="Profit Factor: 0.00", font=('Consolas', 12), bg='#0d0d0d', fg='#aaa')
        self.stat_pf.pack(anchor='w', padx=8, fill='x')
        
        self.stat_best = tk.Label(stats_panel, text="Best: $0.00", font=('Consolas', 12), bg='#0d0d0d', fg='#00ff00')
        self.stat_best.pack(anchor='w', padx=8, fill='x')
        
        self.stat_worst = tk.Label(stats_panel, text="Worst: $0.00", font=('Consolas', 12), bg='#0d0d0d', fg='#ff4444')
        self.stat_worst.pack(anchor='w', padx=8, pady=(0, 8), fill='x')

        # --- LEFT PANEL (MAIN CONTENT) ---
        self.frame = tk.Frame(self.main_container, bg='#1a1a1a', padx=30, pady=30)
        self.frame.pack(side='left', fill='both', expand=True)
        
        # Top control bar
        ctrl_frame = tk.Frame(self.frame, bg='#1a1a1a')
        ctrl_frame.pack(fill='x', pady=(0, 20))
        
        # Left side - Wallet
        bal_txt = f"Wallet: ${self.trader.balance:,.0f}"
        self.wallet_lbl = tk.Label(ctrl_frame, text=bal_txt, font=('Consolas', 22, 'bold'), bg='#1a1a1a', fg='#ffff00')
        self.wallet_lbl.pack(side='left')
        
        # Right side - Controls
        btn_frame = tk.Frame(ctrl_frame, bg='#1a1a1a')
        btn_frame.pack(side='right')
        
        self.min_btn = tk.Label(btn_frame, text="–", bg='#1a1a1a', fg='#666666', font=('Arial', 40, 'bold'), cursor="hand2")
        self.min_btn.pack(side='left', padx=15)
        self.min_btn.bind('<Button-1>', self.minimize_app)
        
        self.close_btn = tk.Label(btn_frame, text="×", bg='#1a1a1a', fg='#666666', font=('Arial', 40), cursor="hand2")
        self.close_btn.pack(side='left')
        self.close_btn.bind('<Button-1>', lambda e: self.root.destroy())

        # --- SCROLLABLE AREA SETUP ---
        # 1. Canvas and Scrollbar
        self.canvas = tk.Canvas(self.frame, bg='#1a1a1a', highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        
        # 2. Scrollable Frame (holds the coins)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#1a1a1a')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # 3. Create window in canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.screen_w * 0.7) # Approx width
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 4. Pack scroll elements
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # --- DYNAMIC COIN UI GENERATION ---
        for i, symbol in enumerate(TRADING_CONFIG['symbols']):
            self.create_coin_ui(symbol)
            # Add separator if not last
            if i < len(TRADING_CONFIG['symbols']) - 1:
                tk.Frame(self.scrollable_frame, height=2, bg='#333333').pack(fill='x', pady=20)

        self.status_label = tk.Label(self.scrollable_frame, text="System Active - 8 Pairs Optimized", font=('Segoe UI', 14), bg='#1a1a1a', fg='#444444')
        self.status_label.pack(anchor='w', pady=(30, 0))
        


    def minimize_app(self, event=None):
        self.root.iconify()

    def log_event(self, msg):
        ts = time.strftime('%H:%M:%S')
        full_msg = f"[{ts}] {msg}"
        self.log_list.insert(0, full_msg)
        
        color = "#cccccc"
        if "LONG" in msg: color = "#00ff00"
        elif "SHORT" in msg: color = "#ff4444"
        elif "WIN" in msg: color = "#00ffea"
        elif "LOSS" in msg: color = "#ff00ea"
            
        self.log_list.itemconfig(0, fg=color)
        if self.log_list.size() > 50: self.log_list.delete(50, tk.END)
        
        with open(self.log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d')} {full_msg}\n")

    def create_coin_ui(self, symbol):
        # Use scrollable_frame as parent instead of self.frame
        frame = tk.Frame(self.scrollable_frame, bg='#1a1a1a')
        frame.pack(fill='both', expand=True, pady=5)
        
        # Row 1: Price and Signal
        row1 = tk.Frame(frame, bg='#1a1a1a')
        row1.pack(fill='x', pady=5)
        
        # Price label - reduced font for better fit
        lbl_price = tk.Label(row1, text=f"{symbol}: $...", font=('Segoe UI', 80, 'bold'), bg='#1a1a1a', fg='white')
        lbl_price.pack(side='left')
        
        # Signal label
        lbl_sig = tk.Label(row1, text="", font=('Segoe UI', 28, 'bold'), bg='#1a1a1a', fg='gold')
        lbl_sig.pack(side='right', padx=20)
        
        # Row 2: Target info (TP/SL)
        row_tgt = tk.Frame(frame, bg='#1a1a1a')
        row_tgt.pack(fill='x')
        lbl_tgt = tk.Label(row_tgt, text="", font=('Segoe UI', 20, 'bold'), bg='#1a1a1a', fg='#aaaaaa')
        lbl_tgt.pack(side='left', padx=10)
        setattr(self, f"{symbol}_tgt", lbl_tgt)
        
        # Row 3: Stats (RSI, Stoch, CVD)
        row2 = tk.Frame(frame, bg='#1a1a1a')
        row2.pack(fill='x')
        lbl_stats = tk.Label(row2, text="Loading...", font=('Segoe UI', 16), bg='#1a1a1a', fg='#888888')
        lbl_stats.pack(side='left', padx=10)
        
        setattr(self, f"{symbol}_price", lbl_price)
        setattr(self, f"{symbol}_sig", lbl_sig)
        setattr(self, f"{symbol}_stats", lbl_stats)

    def start_websocket(self):
        """Start WebSocket with auto-reconnect."""
        # Dynamic stream generation
        streams_list = []
        for s in TRADING_CONFIG['symbols']:
            curr = s.lower()
            streams_list.append(f"{curr}usdt@markPrice@1s")
            streams_list.append(f"{curr}usdt@aggTrade")
        
        streams_list.append("!forceOrder@arr") # Global force order stream
        
        streams = "/".join(streams_list)
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                logger.info("Connecting to Binance WebSocket...")
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self.on_ws_message,
                    on_error=self.on_ws_error,
                    on_open=lambda w: logger.info("✅ WebSocket Connected"),
                    on_close=lambda w, c, m: logger.warning(f"WebSocket Closed: {c} - {m}")
                )
                ws.run_forever(
                    ping_interval=WEBSOCKET_CONFIG['ping_interval'], 
                    ping_timeout=WEBSOCKET_CONFIG['ping_timeout']
                )
            except Exception as e:
                logger.error(f"WebSocket Exception: {e}")
            
            if self.running:
                delay = min(WEBSOCKET_CONFIG.get('reconnect_delay', 5) * 2, 60) # Simple backoff
                logger.warning(f"WebSocket disconnected. Reconnecting in {delay}s...")
                time.sleep(delay)

    def on_ws_message(self, ws, message):
        try:
            msg = json.loads(message)
            stream = msg['stream']
            data = msg['data']

            if 'markPrice' in stream:
                symbol = data['s'].replace('USDT', '')
                if symbol in self.data:
                    self.data[symbol]['mark'] = float(data['p'])
                    self.data[symbol]['funding'] = float(data['r'])  # Capture Funding Rate
                    # Do NOT update price UI here to avoid flickering to Mark Price
            
            elif 'aggTrade' in stream:
                symbol = data['s'].replace('USDT', '')
                if symbol in self.data:
                    price = float(data['p'])
                    qty = float(data['q'])
                    is_maker = data['m']
                    vol = price * qty
                    delta = vol if not is_maker else -vol
                    self.data[symbol]['price'] = price # Official Ticker/Last Price sync
                    self.data[symbol]['cvd'] += delta 
                    self.root.after(0, self.update_price_ui, symbol) 
                    # VPIN Calculation (Flow Toxicity)
                    if not is_maker: self.data[symbol]['buy_vol_bucket'] += vol
                    else: self.data[symbol]['sell_vol_bucket'] += vol
                    
                    # Normalize VPIN every 1s (bucket reset happens in analytics_loop)
                    buy = self.data[symbol]['buy_vol_bucket']
                    sell = self.data[symbol]['sell_vol_bucket']
                    if buy + sell > 0:
                        self.data[symbol]['vpin'] = abs(buy - sell) / (buy + sell)

            elif 'forceOrder' in stream:
                order = data['o']
                symbol = order['s'].replace('USDT', '')
                if symbol in self.data:
                    side = order['S']
                    qty = float(order['q'])
                    price = float(order['p'])
                    vol = qty * price
                    
                    # Accumulate for Liquidity Strategy
                    self.data[symbol]['liq_vol_1m'] += vol
                    
                    # Visual Alerts for Massive Liquidations
                    if vol > LIQUIDITY_CONFIG['liq_mass_threshold']:
                        logger.warning(f"🔥 MASSIVE LIQUIDATION: {symbol} {side} ${vol/1000:.1f}k at {price}")
                        self.data[symbol]['last_liq'] = {'side': side, 'vol': vol, 'time': time.time()}
                        
                        # Add to Liq Map for UI
                        self.liq_map[symbol].append({'price': price, 'vol': vol, 'side': side, 'time': time.time()})
                        if len(self.liq_map[symbol]) > 10: self.liq_map[symbol].pop(0)

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
        except KeyError as e:
            logger.debug(f"Missing key in WS message: {e}")
        except Exception as e:
            logger.warning(f"WS message error: {e}")

    def on_ws_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def can_log_signal(self, symbol, sig_type):
        curr = time.time()
        last = self.last_signal_time[symbol][sig_type]
        if curr - last > 300: 
            self.last_signal_time[symbol][sig_type] = curr
            return True
        return False

    def fetch_analytics(self, symbol):
        headers = {'User-Agent': 'Mozilla/5.0'}
        d = self.data[symbol]
        
        try:
            # 1. 15m Klines (Main Indicators)
            req = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}USDT&interval=15m&limit=210", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                klines_15 = json.loads(r.read().decode())
                closes_15 = [float(k[4]) for k in klines_15]
                highs_15 = [float(k[2]) for k in klines_15]
                lows_15 = [float(k[3]) for k in klines_15]
                volumes_15 = [float(k[5]) for k in klines_15]
                
                d['rsi'] = self.calculate_rsi(closes_15)
                d['stoch_rsi'] = self.calculate_stoch_rsi(closes_15)
                d['ema200_15m'] = self.calculate_ema_value(closes_15, 200)
                d['trend_15m'] = 'BULL' if closes_15[-1] > d['ema200_15m'] else 'BEAR'
                d['macd'] = self.calculate_macd(closes_15)
                d['bb'] = self.calculate_bb(closes_15)
                d['atr'] = self.calculate_atr(highs_15, lows_15, closes_15)
                
                d['vol_ratio'] = volumes_15[-1] / (sum(volumes_15[-21:-1])/20) if len(volumes_15) > 20 else 1.0

            # 2. 1h Klines (Trend Confirmation)
            req_1h = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}USDT&interval=1h&limit=210", headers=headers)
            with urllib.request.urlopen(req_1h, timeout=5) as r:
                klines_1h = json.loads(r.read().decode())
                closes_1h = [float(k[4]) for k in klines_1h]
                d['ema200_1h'] = self.calculate_ema_value(closes_1h, 200)
                d['trend_1h'] = 'BULL' if closes_1h[-1] > d['ema200_1h'] else 'BEAR'

            # 3. Order Book Depth (OBI) - Crucial for Scalping
            req_depth = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}USDT&limit=20", headers=headers)
            with urllib.request.urlopen(req_depth, timeout=5) as r:
                depth = json.loads(r.read().decode())
                b = sum([float(x[1]) for x in depth['bids']])
                a = sum([float(x[1]) for x in depth['asks']])
                d['obi'] = b / a if a > 0 else 1.0

            # 4. Open Interest (Real-time Snapshot)
            try:
                req_oi = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}USDT", headers=headers)
                with urllib.request.urlopen(req_oi, timeout=5) as r:
                    oi_data = json.loads(r.read().decode())
                    d['oi'] = float(oi_data.get('openInterest', 0))
            except: d['oi'] = 0.0
            
            # Reset error count on successful fetch (Thread-safe)
            with self.net_lock:
                self.net_error_count = max(0, self.net_error_count - 1)
                if self.net_error_count == 0:
                    self.net_healthy = True 


        except urllib.error.URLError as e:
            with self.net_lock:
                self.net_error_count += 1
                self.net_healthy = False
                
                # If it's a DNS/Connection error, be extremely silent
                is_dns = "11001" in str(e.reason) or "timed out" in str(e.reason).lower()
                
                if is_dns and self.net_error_count == 1:
                    logger.error("📶 NETWORK DISCONNECTED: DNS/Timeout detected. Entering Silent Reconnection Mode...")
                elif not is_dns and self.net_error_count <= 3:
                    logger.warning(f"Network error fetching {symbol}: {e.reason}")
        except Exception as e:
            self.net_healthy = False
            logger.debug(f"Fetch error: {e}")

    def calculate_ema_series(self, values, period):
        if len(values) < period: return [values[-1]] * len(values)
        emas = []
        sma = sum(values[:period]) / period
        emas.extend([sma] * period)
        k = 2 / (period + 1)
        curr = sma
        for val in values[period:]:
            curr = (val * k) + (curr * (1 - k))
            emas.append(curr)
        return emas

    def calculate_ema_value(self, values, period):
        return self.calculate_ema_series(values, period)[-1]

    def calculate_rsi(self, closes, period=14):
        if len(closes) < period + 1: return 50.0
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_stoch_rsi(self, closes, period=14, smoothK=3, smoothD=3):
        rsis = []
        for i in range(20, 0, -1):
            sub = closes[:-i]
            if len(sub) > period: rsis.append(self._calc_rsi_val(sub, period))
        rsis.append(self._calc_rsi_val(closes, period))
        
        if len(rsis) < 14: return 50.0
        curr_rsi = rsis[-1]
        min_rsi = min(rsis[-14:])
        max_rsi = max(rsis[-14:])
        if max_rsi - min_rsi == 0: return 50.0
        stoch = (curr_rsi - min_rsi) / (max_rsi - min_rsi) * 100
        return stoch 

    def _calc_rsi_val(self, closes, period=14):
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, highs, lows, closes, period=14):
        if len(closes) < period: return 100.0
        tr_list = []
        for i in range(1, len(closes)):
            h = highs[i]
            l = lows[i]
            cp = closes[i-1]
            tr = max(h-l, abs(h-cp), abs(l-cp))
            tr_list.append(tr)
        return sum(tr_list[-period:]) / period

    def calculate_macd(self, closes, fast=8, slow=17, sig=9):
        stats_fast = self.calculate_ema_series(closes, fast)
        stats_slow = self.calculate_ema_series(closes, slow)
        macd_line = [f - s for f, s in zip(stats_fast, stats_slow)]
        if len(macd_line) > len(closes): macd_line = macd_line[-len(closes):]
        sig_line = self.calculate_ema_series(macd_line, sig)
        return {'line': macd_line[-1], 'sig': sig_line[-1], 'hist': macd_line[-1] - sig_line[-1]}

    def calculate_bb(self, closes, period=20, std_dev=2):
        if len(closes) < period: return {'upper':0, 'lower':0, 'mid':0}
        sma = sum(closes[-period:]) / period
        variance = sum([((x - sma) ** 2) for x in closes[-period:]]) / period
        sd = math.sqrt(variance)
        return {'mid': sma, 'upper': sma + (sd*std_dev), 'lower': sma - (sd*std_dev)}

    def calculate_signal_score(self, d):
        # PHASE 7: Schedule Filtering
        if not self.is_within_trading_window():
            return 0
            
        """
        Multi-indicator scoring system.
        Returns score from -7 (strong SHORT) to +7 (strong LONG).
        Integrates: Trend, StochRSI, MACD, Bollinger Bands, OBI, CVD, Volume
        """
        score = 0
        breakdown = []  # For debugging
        
        # --- PHASE 8: AI HYBRID SENSORS ---
        
        # 1. Global Sentiment (RSS News)
        try:
            sent_score, _ = self.sentiment.get_sentiment()
            if sent_score > 0.4:
                score += 1
                breakdown.append("News(Bull)+1")
            elif sent_score < -0.4:
                score -= 1
                breakdown.append("News(Bear)-1")
        except: pass

        # 2. Machine Learning Validation
        if self.ml_model:
            try:
                trend_val = 1 if d['trend_15m'] == 'BULL' else 0
                bb_pos = 1
                if d['bb']['lower'] > 0:
                    if d['mark'] < d['bb']['lower']: bb_pos = 0
                    elif d['mark'] > d['bb']['upper']: bb_pos = 2
                    
                features = {
                    'rsi': d['rsi'], 'stoch_rsi': d['stoch_rsi'],
                    'macd_hist': d['macd']['hist'], 'atr': d['atr'],
                    'bb_position': bb_pos, 'trend': trend_val,
                    'vol_ratio': d.get('vol_ratio', 1.0)
                }
                
                pred = self.ml_model.predict(features)
                if pred['confidence'] > 60:
                    pts = 2 if pred['confidence'] > 80 else 1
                    if pred['direction'] == 'UP':
                        score += pts
                        breakdown.append(f"AI(Up)+{pts}")
                    else:
                        score -= pts
                        breakdown.append(f"AI(Down)-{pts}")
            except Exception: pass
        
        # --- TECHNICAL INDICATORS ---
        
        # 1. TREND ALIGNMENT (+/-2 points)
        # Both timeframes aligned = strong signal
        if d['trend_1h'] == d['trend_15m']:
            if d['trend_1h'] == 'BULL':
                score += 2
                breakdown.append("Trend+2")
            else:
                score -= 2
                breakdown.append("Trend-2")
        
        # 2. STOCHRSI EXTREMES (+/-1 point)
        if d['stoch_rsi'] < SCORING_CONFIG['stoch_oversold']:
            score += 1
            breakdown.append("Stoch+1")
        elif d['stoch_rsi'] > SCORING_CONFIG['stoch_overbought']:
            score -= 1
            breakdown.append("Stoch-1")
        
        # 3. MACD HISTOGRAM (+/-1 point)
        macd_hist = d['macd']['hist']
        if macd_hist > 0:
            score += 1
            breakdown.append("MACD+1")
        elif macd_hist < 0:
            score -= 1
            breakdown.append("MACD-1")
        
        # 4. BOLLINGER BAND POSITION (+/-1 point)
        price = d['mark']
        if d['bb']['lower'] > 0:  
            if price < d['bb']['lower']:
                score += 1
                breakdown.append("BB+1")
            elif price > d['bb']['upper']:
                score -= 1
                breakdown.append("BB-1")
        
        # 5. ORDER BOOK IMBALANCE (+/-1 point)
        if d['obi'] > SCORING_CONFIG['obi_bullish']:
            score += 1
            breakdown.append("OBI+1")
        elif d['obi'] < SCORING_CONFIG['obi_bearish']:
            score -= 1
            breakdown.append("OBI-1")
        
        # 6. CVD MOMENTUM (+/-1 point)
        cvd = d['cvd']
        if cvd > SCORING_CONFIG['cvd_threshold']:
            score += 1
            breakdown.append("CVD+1")
        elif cvd < -SCORING_CONFIG['cvd_threshold']:
            score -= 1
            breakdown.append("CVD-1")
        
        # --- PHASE 10: INSTITUTIONAL LIQUIDITY (MM) ---

        # 1. Liquidation Exhaustion (Reversal Trap)
        liq_1m = d.get('liq_vol_1m', 0)
        if liq_1m > LIQUIDITY_CONFIG['liq_mass_threshold']:
            last_l = d.get('last_liq', {})
            if last_l.get('side') == 'SELL': 
                 score -= 2 
                 breakdown.append("LiqShort-2")
            elif last_l.get('side') == 'BUY': 
                 score += 2 
                 breakdown.append("LiqLong+2")

        # 2. VPIN / Flow Toxicity
        if d.get('vpin', 0) > LIQUIDITY_CONFIG['vpin_threshold']:
            if score > 0: score -= 1; breakdown.append("Toxic-1")
            elif score < 0: score += 1; breakdown.append("Toxic+1")

        # 3. Institutional Absorption (CVD Divergence)
        if d['trend_15m'] == 'BEAR' and d['cvd'] > 200000:
            score += 1
            breakdown.append("Absorb+1")
        elif d['trend_15m'] == 'BULL' and d['cvd'] < -200000:
            score -= 1
            breakdown.append("Absorb-1")

        # Store breakdown for display
        d['signal_breakdown'] = breakdown
        
        return score  # Range: -7 to +7

    def analytics_loop(self):
        last_fetch = 0
        while self.running:
            try:
                curr = time.time()
                if curr - last_fetch > 5:
                    # Parallel fetching using ThreadPoolExecutor for 8 symbols
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        executor.map(self.fetch_analytics, TRADING_CONFIG['symbols'])
                    
                    # MM & LIQUIDITY LOGIC: Apply decay and resets
                    for sym in TRADING_CONFIG['symbols']:
                        if sym in self.data:
                             d = self.data[sym]
                             # 1. CVD Decay (Existing)
                             d['cvd'] *= 0.95
                             
                             # 2. Liquidation Decay (30s window approx)
                             # Since we run every 5s, 0.82 decay ~= 30s half-life
                             d['liq_vol_1m'] *= 0.8 
                             
                             # 3. VPIN Bucket Reset (Keep 20% to smooth transitions)
                             d['buy_vol_bucket'] *= 0.2
                             d['sell_vol_bucket'] *= 0.2

                             # 4. Background Scoring (Move from UI thread to here)
                             score = self.calculate_signal_score(d)
                             sent_val, _ = self.sentiment.get_sentiment()
                             
                             d['latest_features'] = {
                                'rsi': d['rsi'], 'stoch_rsi': d['stoch_rsi'],
                                'macd_hist': d['macd']['hist'], 'atr': d['atr'],
                                'obi': d['obi'], 'cvd': d['cvd'],
                                'vol_ratio': d['vol_ratio'], 'vpin': d.get('vpin', 0),
                                'liq_vol': d.get('liq_vol_1m', 0), 'funding': d.get('funding', 0),
                                'oi': d.get('oi', 0), 'sentiment': sent_val,
                                'score': score  
                             }
                             d['latest_score'] = score

                    # 5. DATA CLEANING & MEMORY MGMT (Every 5 mins approx)
                    if int(curr) % 300 < 10:
                        # Trim liq_map to keep only last 1 hour of levels
                        for s in self.liq_map:
                            self.liq_map[s] = [l for l in self.liq_map[s] if curr - l['time'] < 3600]
                    
                    self.last_heartbeat = curr
                    self.root.after(0, self.update_stats_ui)
                    last_fetch = curr
                
                # Prepare prices dict
                prices = {}
                for sym in TRADING_CONFIG['symbols']:
                    if sym in self.data:
                        prices[sym] = self.data[sym]['mark']
                
                closed_trades = self.trader.update(prices)
                if closed_trades:
                    for t in closed_trades:
                        pnl_txt = f"{'+' if t['pnl'] > 0 else ''}${t['pnl']:.2f}"
                        msg = "WIN" if t['pnl'] > 0 else "LOSS"
                        self.root.after(0, self.log_event, f"{t['symbol']} {msg} {pnl_txt} ({t['reason']})")
                        
                        # Telegram notification
                        if telegram:
                            telegram.send_trade_result(t['symbol'], t['side'], t['pnl'], t['reason'])
                            # Check if daily limit was hit
                            if not self.trader.trading_enabled:
                                telegram.send_daily_limit_hit(self.trader.daily_pnl)
                    
                    self.root.after(0, self.update_wallet_ui)
            except Exception as e:
                logger.error(f"Critical Error in Analytics Loop: {e}")
                
            time.sleep(1)

    def update_wallet_ui(self):
        bal = self.trader.balance
        open_pnl = 0
        
        # Get all current prices
        prices = {}
        for sym in TRADING_CONFIG['symbols']:
            if sym in self.data:
                prices[sym] = self.data[sym]['mark']
                
        for p in self.trader.positions:
            curr = prices.get(p['symbol'], p['entry'])
            if p['side'] == 'BUY': open_pnl += (curr - p['entry']) * p['size']
            else: open_pnl += (p['entry'] - curr) * p['size']
            
        total_eq = bal + open_pnl
        daily = self.trader.daily_pnl
        
        # Color based on daily PnL
        if not self.trader.trading_enabled:
            col = "#ff00ff"  # Magenta = trading disabled
            wallet_txt = f"🛑 LIMIT HIT | Equity: ${total_eq:,.2f}"
        elif daily >= 0:
            col = "#00ff00"
            wallet_txt = f"Equity: ${total_eq:,.2f} (Day: +${daily:.1f})"
        else:
            col = "#ffff00" if daily > -250 else "#ff4444"
            wallet_txt = f"Equity: ${total_eq:,.2f} (Day: -${abs(daily):.1f})"
        
        self.wallet_lbl.configure(text=wallet_txt, fg=col)
        
        # Also update stats panel
        self.update_stats_panel()

    def update_stats_panel(self):
        """Update the statistics panel with trading performance metrics."""
        history = self.trader.history
        
        if not history:
            return
        
        # Calculate metrics
        wins = [t for t in history if t['pnl'] > 0]
        losses = [t for t in history if t['pnl'] <= 0]
        
        total = len(history)
        win_count = len(wins)
        loss_count = len(losses)
        
        # Win Rate
        winrate = (win_count / total * 100) if total > 0 else 0
        wr_color = "#00ff00" if winrate >= 50 else "#ff4444"
        self.stat_winrate.configure(text=f"Win Rate: {winrate:.1f}%", fg=wr_color)
        
        # Trades count
        self.stat_trades.configure(text=f"Trades: {total} ({win_count}W / {loss_count}L)")
        
        # Profit Factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        pf_color = "#00ff00" if pf >= 1.5 else "#ffff00" if pf >= 1.0 else "#ff4444"
        self.stat_pf.configure(text=f"Profit Factor: {pf:.2f}", fg=pf_color)
        
        # Best trade
        best = max(history, key=lambda t: t['pnl'])
        self.stat_best.configure(text=f"Best: +${best['pnl']:.2f}")
        
        # Worst trade
        worst = min(history, key=lambda t: t['pnl'])
        self.stat_worst.configure(text=f"Worst: ${worst['pnl']:.2f}")

    def update_price_ui(self, symbol):
        lbl = getattr(self, f"{symbol}_price")
        price = self.data[symbol]['price']
        dec = AVAILABLE_PAIRS.get(symbol, {}).get('decimals', 2)
        lbl.configure(text=f"{symbol}: ${price:,.{dec}f}")

    def update_stats_ui(self):
        for sym in TRADING_CONFIG['symbols']:
            d = self.data[sym]
            
            lbl_stats = getattr(self, f"{sym}_stats")
            cvd = d['cvd'] / 1000000 
            vpin = d.get('vpin', 0)
            
            # Color VPIN (Toxic flow warning)
            vpin_col = ""
            if vpin > LIQUIDITY_CONFIG['vpin_threshold']: vpin_col = "🔥"
            
            # Liq Map Check (Enhanced)
            price = d['mark']
            liq_txt = ""
            for l in self.liq_map[sym]:
                if abs(price - l['price']) / price < 0.002:
                     lvl_type = "RESIST" if price < l['price'] else "SUP"
                     liq_txt = f" | ⚠️ LIQ {lvl_type}"
            
            lbl_stats.configure(text=f"RSI:{d['rsi']:.1f} | CVD:{cvd:.1f}M | VPIN:{vpin:.2f}{vpin_col}{liq_txt}")
            
            trend_1h = d['trend_1h']
            trend_15m = d['trend_15m']
            stoch = d['stoch_rsi']
            obi = d['obi']
            atr = d['atr']
            
            lbl_sig = getattr(self, f"{sym}_sig")
            lbl_tgt = getattr(self, f"{sym}_tgt")
            
            sig_txt = "SCANNING..."
            sig_col = "#333333"
            tgt_txt = ""

            has_pos = any(p['symbol'] == sym for p in self.trader.positions)

            if not has_pos and atr > 0:
                # Use pre-calculated score and features from background thread
                score = d.get('latest_score', 0)
                features = d.get('latest_features', {})
                
                # If background scoring hasn't run yet, skip
                if not features: continue
                
                # Display current score
                score_color = "#00ff00" if score > 0 else "#ff4444" if score < 0 else "#888888"
                breakdown_txt = " ".join(d.get('signal_breakdown', []))
                
                # LONG SIGNAL: Score threshold from config
                if score >= SIGNAL_CONFIG['min_score_long']:
                    sig_txt = f"🔵 LONG [{score}/7]"
                    sig_col = "#00ff00"
                    
                    sl = price - (SIGNAL_CONFIG['atr_sl_multiplier'] * atr)
                    tp = price + (SIGNAL_CONFIG['atr_tp_multiplier'] * atr)
                    tgt_txt = f"TP: ${tp:,.2f} | SL: ${sl:,.2f}"
                    
                    self.trigger_alert()
                    if self.can_log_signal(sym, 'BUY'):
                        self.log_event(f"{sym} [LONG] Score:{score} {breakdown_txt}")
                        self.trader.open_position(sym, 'BUY', price, sl, tp, features, atr)
                        self.update_wallet_ui()
                        # Telegram notification
                        if telegram:
                            telegram.send_signal(sym, 'BUY', price, tp, sl, score)

                # SHORT SIGNAL: Score threshold from config
                elif score <= SIGNAL_CONFIG['min_score_short']:
                    sig_txt = f"🔴 SHORT [{score}/7]"
                    sig_col = "#ff4444"
                    
                    sl = price + (SIGNAL_CONFIG['atr_sl_multiplier'] * atr)
                    tp = price - (SIGNAL_CONFIG['atr_tp_multiplier'] * atr)
                    tgt_txt = f"TP: ${tp:,.2f} | SL: ${sl:,.2f}"
                    
                    self.trigger_alert()
                    if self.can_log_signal(sym, 'SELL'):
                        self.log_event(f"{sym} [SHORT] Score:{score} {breakdown_txt}")
                        self.trader.open_position(sym, 'SELL', price, sl, tp, features, atr)
                        self.update_wallet_ui()
                        # Telegram notification
                        if telegram:
                            telegram.send_signal(sym, 'SELL', price, tp, sl, score)
                
                # NEUTRAL: Show current score while scanning
                else:
                    if not self.is_within_trading_window():
                        sig_txt = "OFF-SESSION"
                        sig_col = "#666666"
                    else:
                        sig_txt = f"SCAN [{score:+d}]"
                        sig_col = score_color
                    
                    if abs(score) >= 2:  # Show breakdown if close to signal
                        tgt_txt = breakdown_txt
            
            if sig_txt == "SCANNING..." and has_pos:
                pos = next(p for p in self.trader.positions if p['symbol'] == sym)
                pnl = pos['pnl']
                col = "#00ff00" if pnl > 0 else "#ff4444"
                
                # Show Trailing Level with emoji
                level = pos.get('trailing_level', 0)
                level_icons = {0: "", 1: "⚡BE", 2: "🔒25%", 3: "🎯TRAIL"}
                level_txt = level_icons.get(level, "")
                
                sig_txt = f"ACTIVE {level_txt}" if level > 0 else "ACTIVE"
                sig_col = col
                # Show Notional Size and Margin used
                tgt_txt = f"PnL: ${pnl:.2f} | Size: ${pos['notional']:,.0f} | Margin: ${pos['margin']:.1f}"
                lbl_tgt.configure(text=tgt_txt, fg=col)

            elif sig_txt == "SCANNING...":
                lbl_tgt.configure(text="")

            lbl_sig.configure(text=sig_txt, fg=sig_col)
            
        # Refresh wallet with real-time unrealized PnL
        self.update_wallet_ui()
            
        # Update System Status Heartbeat + AI SENTIMENT
        now_ts = time.strftime('%H:%M:%S')
        sent_score, _ = self.sentiment.get_sentiment()
        sent_txt = "BULLISH 🚀" if sent_score > 0.3 else "BEARISH 📉" if sent_score < -0.3 else "NEUTRAL ⚖️"
        
        if not self.net_healthy:
            status_txt = f"📶 RECONNECTING... | {now_ts} | Network issues detected"
            status_col = "#ffcc00" # Warning yellow
        else:
            status_txt = f"⚡ ACTIVE | {now_ts} | Sentiment: {sent_txt} ({sent_score:+.2f})"
            status_col = "#00ff00"
            
        self.status_label.configure(text=status_txt, fg=status_col)

    def trigger_alert(self):
        t = time.time()
        if t - self.last_alert_time > self.alert_cooldown:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            self.last_alert_time = t

    def is_within_trading_window(self):
        """Checks if current time is within allowed trading sessions."""
        if not SCHEDULE_CONFIG.get('enabled', False):
            return True
            
        now = datetime.now(timezone.utc).time() if SCHEDULE_CONFIG.get('timezone') == 'UTC' else datetime.now().time()
        
        for window in SCHEDULE_CONFIG.get('windows', []):
            try:
                start_str = window.get('start', '00:00')
                end_str = window.get('end', '00:00')
                start = datetime.strptime(start_str, "%H:%M").time()
                end = datetime.strptime(end_str, "%H:%M").time()
                
                # Handle overnight windows (e.g., 22:00 to 04:00)
                if start <= end:
                    if start <= now <= end: return True
                else:
                    if now >= start or now <= end: return True
            except Exception as e:
                logger.error(f"Error parsing trading window {window}: {e}")
                
        return False

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + (event.x - self.x)
        y = self.root.winfo_y() + (event.y - self.y)
        self.root.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("🚀 SNIPER BOT PRO - Starting...")
    
    # AGGRESSIVE RISK WARNING
    risk_pct = TRADING_CONFIG.get('risk_per_trade', 0.01) * 100
    if risk_pct >= 20:
        logger.warning(f"⚠️ AGGRESSIVE RISK: {risk_pct}% per trade!")
        logger.warning("   Max simultaneous trades: " + str(TRADING_CONFIG.get('max_active_trades', 5)))
        logger.warning("   Proceed with caution.")
        
    logger.info("="*50)
    
    # Send Telegram startup notification
    if telegram:
        telegram.send_startup()
    
    root = tk.Tk()
    app = CryptoWidget(root)
    logger.info("Application initialized. Running mainloop...")
    root.mainloop()
    logger.info("Application closed.")

