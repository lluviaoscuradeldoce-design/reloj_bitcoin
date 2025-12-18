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
    from config import TELEGRAM_CONFIG, TRADING_CONFIG, WEBSOCKET_CONFIG
    from notifications import init_notifier, get_notifier
    
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
                f.write("symbol,side,rsi,stoch_rsi,macd_hist,atr,obi,cvd,trend_1h,vol_ratio,score,outcome,pnl_percent\n")
    
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
        """Check if trading is allowed (daily limit not hit)."""
        self.check_daily_reset()
        return self.trading_enabled

    def open_position(self, symbol, side, entry_price, sl, tp, features, atr):
        if not self.can_trade():
            return None
            
        risk_amt = self.balance * TRADING_CONFIG['risk_per_trade']  # Dynamic risk from config
        dist = abs(entry_price - sl)
        if dist == 0: return None
        
        size_qty = risk_amt / dist
        
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry_price,
            'size': size_qty,
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
            
            if profit_atr >= 1.5 and p['trailing_level'] < 3:
                # LEVEL 3: Aggressive Trail (0.5 ATR)
                p['trailing_level'] = 3
                p['trailing_active'] = True
                if p['side'] == 'BUY':
                    new_sl = curr - (0.5 * atr)
                else:
                    new_sl = curr + (0.5 * atr)
                    
            elif profit_atr >= 1.0 and p['trailing_level'] < 2:
                # LEVEL 2: Lock 25% profit
                p['trailing_level'] = 2
                p['trailing_active'] = True
                if p['side'] == 'BUY':
                    new_sl = entry + (0.25 * atr)
                else:
                    new_sl = entry - (0.25 * atr)
                    
            elif profit_atr >= 0.3 and p['trailing_level'] < 1:
                # LEVEL 1: Break Even
                p['trailing_level'] = 1
                p['trailing_active'] = True
                new_sl = entry
            
            # Continue trailing if Level 3 is active
            if p['trailing_level'] == 3:
                if p['side'] == 'BUY':
                    trail_sl = curr - (0.5 * atr)
                    if trail_sl > p['sl']:
                        new_sl = trail_sl
                else:
                    trail_sl = curr + (0.5 * atr)
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
        
        row = f"{trade['symbol']},{trade['side']},{f['rsi']:.2f}," \
              f"{f['stoch_rsi']:.2f},{f['macd_hist']:.4f},{f['atr']:.2f}," \
              f"{f['obi']:.2f},{f['cvd']:.2f},{f['trend_1h']},{f['vol_ratio']:.2f}," \
              f"{score},{outcome},{pnl_pct:.4f}\n"
              
        with open(self.csv_file, "a") as file:
            file.write(row)



class CryptoWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Sniper Bot Pro")
        
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
        
        self.setup_ui()
        
        self.running = True
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
            'cvd': 0.0
        }

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
        if self.log_list.size() > 100: self.log_list.delete(100, tk.END)
        
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
                logger.warning("WebSocket disconnected. Reconnecting in 5s...")
                time.sleep(5)

    def on_ws_message(self, ws, message):
        try:
            msg = json.loads(message)
            stream = msg['stream']
            data = msg['data']

            if 'markPrice' in stream:
                symbol = data['s'].replace('USDT', '')
                if symbol in self.data:
                    self.data[symbol]['mark'] = float(data['p'])
                    self.root.after(0, self.update_price_ui, symbol)
            
            elif 'aggTrade' in stream:
                symbol = data['s'].replace('USDT', '')
                if symbol in self.data:
                    price = float(data['p'])
                    qty = float(data['q'])
                    is_maker = data['m']
                    vol = price * qty
                    delta = vol if not is_maker else -vol
                    self.data[symbol]['cvd'] += delta 

            elif 'forceOrder' in stream:
                order = data['o']
                symbol = order['s'].replace('USDT', '')
                if symbol in self.data:
                    side = order['S']
                    qty = float(order['q'])
                    price = float(order['p'])
                    vol = qty * price
                    if vol > 100000: # Increase threshold for Map
                        self.data[symbol]['last_liq'] = {'side': side, 'vol': vol, 'time': time.time()}
                        # Add to Liq Map
                        self.liq_map[symbol].append({'price': price, 'vol': vol, 'side': side, 'time': time.time()})
                        # Keep only last 5
                        if len(self.liq_map[symbol]) > 5: self.liq_map[symbol].pop(0)

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
            with urllib.request.urlopen(req, timeout=2) as r:
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
            with urllib.request.urlopen(req_1h, timeout=2) as r:
                klines_1h = json.loads(r.read().decode())
                closes_1h = [float(k[4]) for k in klines_1h]
                d['ema200_1h'] = self.calculate_ema_value(closes_1h, 200)
                d['trend_1h'] = 'BULL' if closes_1h[-1] > d['ema200_1h'] else 'BEAR'

            # 3. Order Book Depth (OBI) - Crucial for Scalping
            req_depth = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}USDT&limit=20", headers=headers)
            with urllib.request.urlopen(req_depth, timeout=2) as r:
                depth = json.loads(r.read().decode())
                b = sum([float(x[1]) for x in depth['bids']])
                a = sum([float(x[1]) for x in depth['asks']])
                d['obi'] = b / a if a > 0 else 1.0

        except urllib.error.URLError as e:
            logger.warning(f"Network error fetching {symbol}: {e.reason}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error fetching analytics for {symbol}: {e}")

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
        """
        Multi-indicator scoring system.
        Returns score from -7 (strong SHORT) to +7 (strong LONG).
        Integrates: Trend, StochRSI, MACD, Bollinger Bands, OBI, CVD, Volume
        """
        score = 0
        breakdown = []  # For debugging
        
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
        # Oversold = bullish, Overbought = bearish
        if d['stoch_rsi'] < 25:
            score += 1
            breakdown.append("Stoch+1")
        elif d['stoch_rsi'] > 75:
            score -= 1
            breakdown.append("Stoch-1")
        
        # 3. MACD HISTOGRAM (+/-1 point)
        # Positive histogram = bullish momentum
        macd_hist = d['macd']['hist']
        if macd_hist > 0:
            score += 1
            breakdown.append("MACD+1")
        elif macd_hist < 0:
            score -= 1
            breakdown.append("MACD-1")
        
        # 4. BOLLINGER BAND POSITION (+/-1 point)
        # Below lower band = oversold (bullish), above upper = overbought (bearish)
        price = d['mark']
        if d['bb']['lower'] > 0:  # Ensure BB is calculated
            if price < d['bb']['lower']:
                score += 1
                breakdown.append("BB+1")
            elif price > d['bb']['upper']:
                score -= 1
                breakdown.append("BB-1")
        
        # 5. ORDER BOOK IMBALANCE (+/-1 point)
        # More bids than asks = bullish
        if d['obi'] > 1.15:
            score += 1
            breakdown.append("OBI+1")
        elif d['obi'] < 0.85:
            score -= 1
            breakdown.append("OBI-1")
        
        # 6. CVD MOMENTUM (+/-1 point)
        # Positive CVD (net buying) = bullish
        cvd = d['cvd']
        if cvd > 500000:  # Significant positive delta
            score += 1
            breakdown.append("CVD+1")
        elif cvd < -500000:  # Significant negative delta
            score -= 1
            breakdown.append("CVD-1")
        
        # Store breakdown for display
        d['signal_breakdown'] = breakdown
        
        return score  # Range: -7 to +7

    def analytics_loop(self):
        last_fetch = 0
        while self.running:
            try:
                curr = time.time()
                if curr - last_fetch > 5:
                    for sym in TRADING_CONFIG['symbols']:
                        self.fetch_analytics(sym)
                        # CVD Decay: Prevent infinite accumulation bias
                        if sym in self.data:
                             self.data[sym]['cvd'] *= 0.95
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
            wallet_txt = f"🛑 LIMIT HIT | ${total_eq:,.2f}"
        elif daily >= 0:
            col = "#00ff00"
            wallet_txt = f"Wallet: ${total_eq:,.2f} (Day: +${daily:.0f})"
        else:
            col = "#ffff00" if daily > -250 else "#ff4444"
            wallet_txt = f"Wallet: ${total_eq:,.2f} (Day: -${abs(daily):.0f})"
        
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
        price = self.data[symbol]['mark']
        lbl.configure(text=f"{symbol}: ${price:,.2f}")

    def update_stats_ui(self):
        for sym in TRADING_CONFIG['symbols']:
            d = self.data[sym]
            
            lbl_stats = getattr(self, f"{sym}_stats")
            cvd = d['cvd'] / 1000000 
            cvd_txt = f"{cvd:.1f}M"
            
            # Liq Map Check (NEW)
            price = d['mark']
            liq_txt = ""
            for l in self.liq_map[sym]:
                # Check within 0.2%
                if abs(price - l['price']) / price < 0.002:
                     lvl_type = "RESIST" if price < l['price'] else "SUP"
                     liq_txt = f" | ⚠️ LIQ {lvl_type}"
            
            lbl_stats.configure(text=f"RSI:{d['rsi']:.1f} | Stoch:{d['stoch_rsi']:.1f} | CVD: {cvd_txt}{liq_txt}")
            
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
                # Calculate multi-indicator score
                score = self.calculate_signal_score(d)
                
                features = {
                    'rsi': d['rsi'],
                    'stoch_rsi': d['stoch_rsi'],
                    'macd_hist': d['macd']['hist'],
                    'atr': d['atr'],
                    'obi': d['obi'],
                    'cvd': d['cvd'],
                    'trend_1h': 1 if trend_1h == 'BULL' else -1,
                    'vol_ratio': d['vol_ratio'],
                    'score': score  # NEW: Store score for analysis
                }
                
                # Display current score
                score_color = "#00ff00" if score > 0 else "#ff4444" if score < 0 else "#888888"
                breakdown_txt = " ".join(d.get('signal_breakdown', []))
                
                # LONG SIGNAL: Score >= 4 (need at least 4 out of 7 bullish signals)
                if score >= 4:
                    sig_txt = f"🔵 LONG [{score}/7]"
                    sig_col = "#00ff00"
                    
                    sl = price - (1.5 * atr)
                    tp = price + (2.5 * atr)
                    tgt_txt = f"TP: ${tp:,.2f} | SL: ${sl:,.2f}"
                    
                    self.trigger_alert()
                    if self.can_log_signal(sym, 'BUY'):
                        self.log_event(f"{sym} [LONG] Score:{score} {breakdown_txt}")
                        self.trader.open_position(sym, 'BUY', price, sl, tp, features, atr)
                        self.update_wallet_ui()
                        # Telegram notification
                        if telegram:
                            telegram.send_signal(sym, 'BUY', price, tp, sl, score)

                # SHORT SIGNAL: Score <= -4 (need at least 4 out of 7 bearish signals)
                elif score <= -4:
                    sig_txt = f"🔴 SHORT [{score}/7]"
                    sig_col = "#ff4444"
                    
                    sl = price + (1.5 * atr)
                    tp = price - (2.5 * atr)
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
                tgt_txt = f"PnL: ${pnl:.2f} | SL: ${pos['sl']:,.0f}"
                lbl_tgt.configure(text=tgt_txt, fg=col)

            elif sig_txt == "SCANNING...":
                lbl_tgt.configure(text="")

            lbl_sig.configure(text=sig_txt, fg=sig_col)
            
        # Update System Status Heartbeat
        now_ts = time.strftime('%H:%M:%S')
        self.status_label.configure(text=f"⚡ ACTIVE | Last Scan: {now_ts} | AI Score Mode", fg='#00ff00')

    def trigger_alert(self):
        t = time.time()
        if t - self.last_alert_time > self.alert_cooldown:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            self.last_alert_time = t

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
    logger.info("="*50)
    
    # Send Telegram startup notification
    if telegram:
        telegram.send_startup()
    
    root = tk.Tk()
    app = CryptoWidget(root)
    logger.info("Application initialized. Running mainloop...")
    root.mainloop()
    logger.info("Application closed.")

