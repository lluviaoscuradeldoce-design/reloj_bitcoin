# -*- coding: utf-8 -*-
"""
Data Collector for ML Training.
Collects historical data from Binance to train ML models.

This script downloads historical klines data and calculates indicators
to create a training dataset for machine learning models.

Usage:
    python data_collector.py                    # Collect 30 days of data
    python data_collector.py --days 90          # Collect 90 days
    python data_collector.py --symbol SOL       # Collect for specific symbol
"""

import urllib.request
import json
import time
import csv
import math
import os
from datetime import datetime, timedelta


class DataCollector:
    """Collects and processes historical market data for ML training."""
    
    def __init__(self, symbols: list = None):
        self.symbols = symbols or ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX']
        self.base_url = "https://fapi.binance.com/fapi/v1"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
    def fetch_klines(self, symbol: str, interval: str = '15m', limit: int = 1500, start_time: int = None) -> list:
        """Fetch historical klines from Binance."""
        url = f"{self.base_url}/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
        if start_time:
            url += f"&startTime={start_time}"
            
        req = urllib.request.Request(url, headers=self.headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"   ⚠️ Error fetching {symbol}: {e}")
            return []
    
    def calculate_rsi(self, closes: list, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_ema(self, values: list, period: int) -> list:
        """Calculate EMA series."""
        if len(values) < period:
            return [values[-1]] * len(values)
        emas = []
        sma = sum(values[:period]) / period
        emas.extend([sma] * period)
        k = 2 / (period + 1)
        curr = sma
        for val in values[period:]:
            curr = (val * k) + (curr * (1 - k))
            emas.append(curr)
        return emas
    
    def calculate_macd(self, closes: list) -> dict:
        """Calculate MACD."""
        fast = self.calculate_ema(closes, 8)
        slow = self.calculate_ema(closes, 17)
        macd_line = [f - s for f, s in zip(fast, slow)]
        sig_line = self.calculate_ema(macd_line, 9)
        return {
            'line': macd_line[-1],
            'signal': sig_line[-1],
            'hist': macd_line[-1] - sig_line[-1]
        }
    
    def calculate_atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float:
        """Calculate ATR."""
        if len(closes) < period:
            return 0
        tr_list = []
        for i in range(1, len(closes)):
            h, l, cp = highs[i], lows[i], closes[i-1]
            tr = max(h - l, abs(h - cp), abs(l - cp))
            tr_list.append(tr)
        return sum(tr_list[-period:]) / period
    
    def calculate_bb(self, closes: list, period: int = 20) -> dict:
        """Calculate Bollinger Bands."""
        if len(closes) < period:
            return {'upper': 0, 'lower': 0, 'mid': 0}
        sma = sum(closes[-period:]) / period
        variance = sum((x - sma) ** 2 for x in closes[-period:]) / period
        sd = math.sqrt(variance)
        return {'mid': sma, 'upper': sma + (sd * 2), 'lower': sma - (sd * 2)}
    
    def calculate_stoch_rsi(self, closes: list) -> float:
        """Calculate Stochastic RSI."""
        rsis = []
        for i in range(20, 0, -1):
            sub = closes[:-i] if i > 0 else closes
            if len(sub) > 14:
                rsis.append(self.calculate_rsi(sub))
        rsis.append(self.calculate_rsi(closes))
        
        if len(rsis) < 14:
            return 50.0
        curr_rsi = rsis[-1]
        min_rsi = min(rsis[-14:])
        max_rsi = max(rsis[-14:])
        if max_rsi - min_rsi == 0:
            return 50.0
        return (curr_rsi - min_rsi) / (max_rsi - min_rsi) * 100
    
    def process_klines(self, klines: list, symbol: str) -> list:
        """Process klines and calculate all indicators."""
        if len(klines) < 210:
            print(f"Not enough data for {symbol}")
            return []
        
        records = []
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        # Process each candle (starting from index 200 for EMA200)
        for i in range(200, len(klines)):
            close_slice = closes[:i+1]
            high_slice = highs[:i+1]
            low_slice = lows[:i+1]
            vol_slice = volumes[:i+1]
            
            # Calculate indicators
            rsi = self.calculate_rsi(close_slice)
            stoch_rsi = self.calculate_stoch_rsi(close_slice)
            macd = self.calculate_macd(close_slice)
            atr = self.calculate_atr(high_slice, low_slice, close_slice)
            bb = self.calculate_bb(close_slice)
            ema200 = self.calculate_ema(close_slice, 200)[-1]
            
            # Trend
            trend = 1 if close_slice[-1] > ema200 else -1
            
            # Volume ratio
            vol_ratio = vol_slice[-1] / (sum(vol_slice[-21:-1]) / 20) if len(vol_slice) > 20 else 1.0
            
            # Price position relative to BB
            price = close_slice[-1]
            bb_position = 0
            if bb['lower'] > 0:
                if price < bb['lower']:
                    bb_position = 1  # Oversold
                elif price > bb['upper']:
                    bb_position = -1  # Overbought
            
            # Future outcome (for labeling) - did price go up or down in next 5 candles?
            future_idx = min(i + 5, len(closes) - 1)
            future_price = closes[future_idx]
            future_return = (future_price - price) / price * 100
            
            # Label: 1 if price went up more than 0.1%, else 0
            label = 1 if future_return > 0.1 else 0
            
            record = {
                'symbol': symbol,
                'side': 'BUY' if label == 1 else 'SELL',
                'rsi': round(rsi, 2),
                'stoch_rsi': round(stoch_rsi, 2),
                'macd_hist': round(macd['hist'], 6),
                'atr': round(atr, 2),
                'obi': 1.0,           # Placeholder
                'cvd': 0.0,           # Placeholder
                'vol_ratio': round(vol_ratio, 2),
                'vpin': 0.0,          # Placeholder
                'liq_vol': 0.0,       # Placeholder
                'funding': 0.0,       # Placeholder
                'oi': 0,              # Placeholder
                'sentiment': 0.0,      # Placeholder
                'score': 0,           # Placeholder for historical
                'outcome': label,     # outcome for analyzer
                'pnl_percent': future_return
            }
            records.append(record)
        
        return records
    
    def collect_data(self, days: int = 365) -> list:
        """Collect data for all symbols using time loops for deep history."""
        all_records = []
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
        
        for symbol in self.symbols:
            print(f"📊 Collecting deep history for {symbol} ({days} days)...")
            symbol_klines = []
            current_start = start_ms
            
            while current_start < now_ms:
                batch = self.fetch_klines(symbol, interval='15m', limit=1500, start_time=current_start)
                if not batch: break
                
                symbol_klines.extend(batch)
                
                # Check if we have reached now (buffer of 15m)
                last_ts = batch[-1][0]
                if last_ts >= now_ms - (15 * 60 * 1000):
                    break
                    
                if last_ts <= current_start: break # Prevent infinite loops
                
                current_start = last_ts + 1
                
                print(f"   📥 Batched {len(symbol_klines)} candles...", end='\r')
                time.sleep(0.5) # Protection for deep history
                
            if symbol_klines:
                # Deduplicate and sort
                symbol_klines = sorted({tuple(k[0:6]): k for k in symbol_klines}.values(), key=lambda x: x[0])
                records = self.process_klines(symbol_klines, symbol)
                all_records.extend(records)
                print(f"\n   ✅ Processed {len(records)} training points for {symbol}")
            
        return all_records
    
    def save_to_csv(self, records: list, filename: str = "ml_training_data.csv"):
        """Save records to CSV file."""
        if not records:
            print("No records to save")
            return
        
        fieldnames = records[0].keys()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"✅ Saved {len(records)} records to {filename}")
    
    def run(self, days: int = 30):
        """Run the data collection process."""
        print("="*50)
        print("🚀 ML Data Collector - Starting...")
        print("="*50)
        
        records = self.collect_data(days)
        self.save_to_csv(records)
        
        # Print summary
        if records:
            labels = [r['outcome'] for r in records]
            positive = sum(labels)
            print(f"\n📊 Dataset Summary:")
            print(f"   Total samples: {len(records)}")
            print(f"   Positive labels (price up): {positive} ({positive/len(records)*100:.1f}%)")
            print(f"   Negative labels (price down): {len(records)-positive} ({(len(records)-positive)/len(records)*100:.1f}%)")


def main():
    import sys
    
    days = 365
    symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX']
    
    # Parse command line args
    if '--days' in sys.argv:
        idx = sys.argv.index('--days')
        days = int(sys.argv[idx + 1])
    
    if '--symbol' in sys.argv:
        idx = sys.argv.index('--symbol')
        symbols = [sys.argv[idx + 1].upper()]
    
    collector = DataCollector(symbols=symbols)
    collector.run(days=days)


if __name__ == "__main__":
    main()
