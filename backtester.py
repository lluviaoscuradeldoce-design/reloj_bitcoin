# -*- coding: utf-8 -*-
"""
Backtesting Engine for Sniper Bot Pro.
Simulates trading strategy using historical data.

Usage:
    python backtester.py --days 30 --symbol BTC
    python backtester.py --optimize
"""

import time
import math
import collections
from datetime import datetime
import statistics

# Import DataCollector to reuse data fetching logic
try:
    from data_collector import DataCollector
except ImportError:
    print("❌ Error: data_collector.py not found.")
    exit(1)

class Backtester:
    def __init__(self, initial_balance=10000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = []
        self.history = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.trades_log = []
        
        # Strategy Parameters (Default)
        self.params = {
            'min_score_long': 4,
            'min_score_short': -4,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.5,
            'risk_per_trade': 0.01,
            'max_daily_loss': -500.0
        }

    def reset(self):
        self.balance = self.initial_balance
        self.positions = []
        self.history = []
        self.daily_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.trades_log = []

    def calculate_indicators(self, klines):
        """Calculate all indicators needed for strategy."""
        collector = DataCollector()
        
        # Parse klines
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        taker_buy_vol = [float(k[9]) for k in klines] # Index 9 is taker buy base asset volume
        
        # 1. EMA 200 (15m)
        ema200 = collector.calculate_ema(closes, 200)
        
        # 2. EMA 200 (1H) - Approximate by using 15m * 4 period?
        # Better: calculate EMA 800 on 15m (approx 200 * 4)
        ema200_1h_approx = collector.calculate_ema(closes, 800)
        
        # 3. RSI & Stoch RSI
        stoch_rsis = []
        # Calculate full series of Stoch RSI is tricky with just simple helper, need rolling
        # Let's do a loop for full series calculation
        for i in range(len(closes)):
            if i < 20: 
                stoch_rsis.append(50)
                continue
            slice_data = closes[:i+1]
            stoch_rsis.append(collector.calculate_stoch_rsi(slice_data))
            
        # 4. MACD
        # Need series of MACD hist
        macds = []
        fast_ema = collector.calculate_ema(closes, 8)
        slow_ema = collector.calculate_ema(closes, 17)
        for i in range(len(closes)):
            macd_line = fast_ema[i] - slow_ema[i]
            macds.append(macd_line)
        signal_ema = collector.calculate_ema(macds, 9)
        macd_hists = [macds[i] - signal_ema[i] for i in range(len(closes))]
        
        # 5. Bollinger Bands
        # Need series of BB
        bbs = []
        for i in range(len(closes)):
             if i < 20:
                 bbs.append({'upper':0, 'lower':0})
                 continue
             slice_data = closes[:i+1]
             bbs.append(collector.calculate_bb(slice_data, 20))

        # 6. ATR
        atrs = []
        for i in range(len(closes)):
            if i < 14:
                atrs.append(0)
                continue
            # Helper computes simple ATR of last N. Need rolling series.
            # Efficient way:
            h, l, c_prev = highs[i], lows[i], closes[i-1]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            if i == 14:
                atrs.append(tr) # Approximate init
            else:
                prev_atr = atrs[-1]
                # Rolling ATR formula: (PrevATR * (n-1) + TR) / n
                atrs.append((prev_atr * 13 + tr) / 14)

        # 7. CVD Approximation (Net Buy Volume)
        # Using Taker Buy Volume - (Total - Taker Buy) = Net Buy
        cvds = []
        cum_cvd = 0
        for i in range(len(closes)):
            net_vol = taker_buy_vol[i] - (volumes[i] - taker_buy_vol[i])
            cum_cvd += net_vol
            cvds.append(cum_cvd)

        return {
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'timestamps': [k[0] for k in klines],
            'ema200': ema200,
            'ema200_1h': ema200_1h_approx,
            'stoch_rsi': stoch_rsis,
            'macd_hist': macd_hists,
            'bb': bbs,
            'atr': atrs,
            'cvd': cvds
        }

    def calculate_score(self, i, data):
        """Calculate strategy score for index i."""
        score = 0
        
        # 1. TREND ALIGNMENT (1H + 15M)
        trend_15m = 'BULL' if data['closes'][i] > data['ema200'][i] else 'BEAR'
        trend_1h = 'BULL' if data['closes'][i] > data['ema200_1h'][i] else 'BEAR'
        
        if trend_15m == 'BULL' and trend_1h == 'BULL':
            score += 2
        elif trend_15m == 'BEAR' and trend_1h == 'BEAR':
            score -= 2
            
        # 2. STOCH RSI
        stoch = data['stoch_rsi'][i]
        if stoch < 25: score += 1
        elif stoch > 75: score -= 1
        
        # 3. MACD
        hist = data['macd_hist'][i]
        hist_prev = data['macd_hist'][i-1] if i > 0 else 0
        if hist > 0 and hist > hist_prev: score += 1
        elif hist < 0 and hist < hist_prev: score -= 1
        
        # 4. BOLLINGER BANDS
        bb = data['bb'][i]
        price = data['closes'][i]
        if bb['lower'] > 0: # Check if valid
            if price <= bb['lower']: score += 1
            elif price >= bb['upper']: score -= 1
            
        # 5. OBI (Order Book Imbalance) - Skipped in backtest (no historical depth)
        
        # 6. CVD Momentum
        # Compare current CVD to MA of CVD? Or just direction?
        # Crypto Widget uses absolute threshold 500k.
        # We will use simple delta of CVD over last 15 mins
        if i > 5:
            cvd_delta_5 = data['cvd'][i] - data['cvd'][i-5]
            if cvd_delta_5 > 1000: score += 1 # Arbitrary threshold for backtest
            elif cvd_delta_5 < -1000: score -= 1
            
        return score

    def run(self, symbol='BTC', days=30, data=None):
        if data is None:
            print(f"🔄 Fetching {days} days of data for {symbol}...")
            collector = DataCollector()
            klines = collector.fetch_klines(symbol, interval='15m', limit=int(days*24*4)) # Approx candles
            
            if not klines:
                print("❌ No data found.")
                return None

            print(f"📊 Processing {len(klines)} candles...")
            data = self.calculate_indicators(klines)
        
        # print("🚀 Running simulation...") # Silence this for optimization
        self.reset()
        
        # Iterate through candles
        for i in range(200, len(data['closes'])):
            # Check Daily Limit Reset (simple approx: every 96 candles = 1 day)
            if i % 96 == 0:
                self.daily_pnl = 0.0
            
            current_price = data['closes'][i]
            timestamp = datetime.fromtimestamp(data['timestamps'][i]/1000)
            atr = data['atr'][i]
            
            # --- MANAGE OPEN POSITIONS ---
            active_positions = []
            for p in self.positions:
                # Check PnL
                curr = current_price
                entry = p['entry']
                sl = p['sl']
                tp = p['tp']
                
                # Check SL hit
                if (p['side'] == 'BUY' and data['lows'][i] <= sl) or \
                   (p['side'] == 'SELL' and data['highs'][i] >= sl):
                    # SL Hit
                    exit_price = sl # Assume filled at SL
                    pnl = (exit_price - entry) * p['size'] if p['side'] == 'BUY' else (entry - exit_price) * p['size']
                    self.close_trade(p, pnl, exit_price, "Stop Loss", timestamp)
                    continue
                
                # Check TP hit
                if (p['side'] == 'BUY' and data['highs'][i] >= tp) or \
                   (p['side'] == 'SELL' and data['lows'][i] <= tp):
                    # TP Hit
                    exit_price = tp # Assume filled at TP
                    pnl = (exit_price - entry) * p['size'] if p['side'] == 'BUY' else (entry - exit_price) * p['size']
                    self.close_trade(p, pnl, exit_price, "Take Profit", timestamp)
                    continue
                
                # --- TRAILING STOP LOGIC ---
                # Calculate current profit in ATR
                if p['side'] == 'BUY':
                    profit_atr = (curr - entry) / atr if atr > 0 else 0
                else:
                    profit_atr = (entry - curr) / atr if atr > 0 else 0
                
                new_sl = p['sl']
                changed = False
                
                # Level 3: Trail
                if profit_atr >= 1.5 and p['trailing_level'] < 3:
                     p['trailing_level'] = 3
                     new_sl = curr - (0.5 * atr) if p['side'] == 'BUY' else curr + (0.5 * atr)
                     changed = True
                # Level 2: Lock 25%
                elif profit_atr >= 1.0 and p['trailing_level'] < 2:
                     p['trailing_level'] = 2
                     new_sl = entry + (0.25 * atr) if p['side'] == 'BUY' else entry - (0.25 * atr)
                     changed = True
                # Level 1: BE
                elif profit_atr >= 0.3 and p['trailing_level'] < 1:
                     p['trailing_level'] = 1
                     new_sl = entry
                     changed = True
                
                # Update SL if trailing active and price moves favorably
                if p['trailing_level'] == 3:
                    if p['side'] == 'BUY':
                        trail = curr - (0.5 * atr)
                        if trail > p['sl']: new_sl = trail; changed = True
                    else:
                        trail = curr + (0.5 * atr)
                        if trail < p['sl']: new_sl = trail; changed = True
                
                if changed:
                     p['sl'] = new_sl
                
                active_positions.append(p)
            
            self.positions = active_positions
            
            # --- CHECK FOR NEW SIGNALS ---
            # Don't open if daily limit is hit
            if self.daily_pnl <= self.params['max_daily_loss']:
                continue
            
            # Don't open if already in position
            if self.positions:
                continue

            score = self.calculate_score(i, data)
            
            if score >= self.params['min_score_long']:
                self.open_position('BUY', current_price, atr, timestamp, score)
            elif score <= self.params['min_score_short']:
                self.open_position('SELL', current_price, atr, timestamp, score)

        return self.print_results()

    def open_position(self, side, price, atr, timestamp, score):
        if atr == 0: return
        
        sl_dist = atr * self.params['atr_sl_mult']
        tp_dist = atr * self.params['atr_tp_mult']
        
        if side == 'BUY':
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            
        dist = abs(price - sl)
        risk_amt = self.balance * self.params['risk_per_trade']
        size = risk_amt / dist
        
        self.positions.append({
            'side': side,
            'entry': price,
            'sl': sl,
            'tp': tp,
            'size': size,
            'atr': atr,
            'trailing_level': 0,
            'open_time': timestamp,
            'score': score
        })

    def close_trade(self, position, pnl, exit_price, reason, timestamp):
        self.balance += pnl
        self.daily_pnl += pnl
        self.history.append({
            'side': position['side'],
            'pnl': pnl,
            'percent': (pnl / self.initial_balance) * 100,
            'open_time': position['open_time'],
            'close_time': timestamp,
            'reason': reason
        })
        
        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def print_results(self):
        wins = [t for t in self.history if t['pnl'] > 0]
        losses = [t for t in self.history if t['pnl'] <= 0]
        
        total_pnl_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # print("\n" + "="*50)
        # print("📊 BACKTEST RESULTS")
        # print("="*50)
        
        # print(f"Final Balance: ${self.balance:,.2f} ({total_pnl_pct:+.2f}%)")
        # print(f"Total Trades: {len(self.history)}")
        
        wr = 0
        pf = 0
        
        if self.history:
            wr = len(wins) / len(self.history) * 100
            # print(f"Win Rate: {wr:.2f}%")
            
            gross_win = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in losses))
            pf = gross_win / gross_loss if gross_loss > 0 else 999
            # print(f"Profit Factor: {pf:.2f}")
            # print(f"Max Drawdown: {self.max_drawdown:.2f}%")
        # else:
            # print("No trades executed.")
            
        # print("="*50)
        
        return {
            'balance': self.balance,
            'pnl_pct': total_pnl_pct,
            'trades': len(self.history),
            'win_rate': wr,
            'profit_factor': pf,
            'drawdown': self.max_drawdown
        }

if __name__ == "__main__":
    import sys
    bt = Backtester()
    symbol = 'BTC'
    days = 7
    
    if '--days' in sys.argv:
        days = int(sys.argv[sys.argv.index('--days')+1])
    
    bt.run(symbol=symbol, days=days)
