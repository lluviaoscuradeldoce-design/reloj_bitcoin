# -*- coding: utf-8 -*-
"""
Hyperparameter Optimizer for Sniper Bot Pro.
Finds the best trading parameters using Grid Search.

Usage:
    python optimizer.py --days 14 --symbol BTC
"""

import itertools
import time
from backtester import Backtester
from data_collector import DataCollector

class Optimizer:
    def __init__(self):
        self.param_grid = {
            'atr_sl_mult': [1.0, 1.5, 2.0],
            'atr_tp_mult': [1.5, 2.0, 2.5, 3.0],
            'min_score_long': [3, 4],
            # 'min_score_short': [-4, -5] # Linked to long score
        }
        
    def run(self, symbol='BTC', days=14):
        print(f"🔄 Fetching data for optimization ({days} days)...")
        bt = Backtester()
        
        # Prepare data once
        collector = DataCollector()
        klines = collector.fetch_klines(symbol, interval='15m', limit=int(days*24*4))
        if not klines:
            print("❌ No data found.")
            return
            
        print("📊 Calculating indicators...")
        data = bt.calculate_indicators(klines)
        
        # Generate combinations
        keys = self.param_grid.keys()
        combinations = list(itertools.product(*self.param_grid.values()))
        
        print(f"🚀 Starting Grid Search on {len(combinations)} combinations...")
        print(f"{'SL':<5} {'TP':<5} {'Score':<6} | {'Trades':<6} {'Win%':<6} {'PF':<6} {'PnL%':<6}")
        print("-" * 55)
        
        best_result = None
        best_pnl = -9999
        
        for vals in combinations:
            params = dict(zip(keys, vals))
            params['min_score_short'] = -params['min_score_long']
            
            # Update backtester params
            bt.params.update(params)
            
            # Run backtest
            try:
                res = bt.run(data=data)
            except Exception as e:
                print(f"Error running backtest with params {params}: {e}")
                continue
            
            if res is None:
                print("⚠️ Backtest returned None")
                continue

            # Print row if there were trades
            if res['trades'] > 0:
                print(f"{params['atr_sl_mult']:<5.1f} {params['atr_tp_mult']:<5.1f} {params['min_score_long']:<6} | "
                      f"{res['trades']:<6} {res['win_rate']:<6.1f} {res['profit_factor']:<6.2f} {res['pnl_pct']:<+6.2f}")
                
                # Track best
                if res['pnl_pct'] > best_pnl:
                    best_pnl = res['pnl_pct']
                    best_result = (params, res)
        
        print("-" * 55)
        
        if best_result:
            params, res = best_result
            print("\n🏆 BEST PARAMETERS FOUND:")
            print(f"   SL Multiplier: {params['atr_sl_mult']}")
            print(f"   TP Multiplier: {params['atr_tp_mult']}")
            print(f"   Min Score: ±{params['min_score_long']}")
            print(f"\n   Stats: {res['pnl_pct']:.2f}% PnL, {res['win_rate']:.1f}% Win Rate, {res['profit_factor']:.2f} PF")
            print(f"\n💡 Update config.py with these values for better performance.")
        else:
             print("❌ No profitable combination found.")

if __name__ == "__main__":
    import sys
    
    days = 14
    if '--days' in sys.argv:
        days = int(sys.argv[sys.argv.index('--days')+1])

    opt = Optimizer()
    opt.run(days=days)
