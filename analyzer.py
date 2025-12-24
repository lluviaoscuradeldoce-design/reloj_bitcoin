# -*- coding: utf-8 -*-
"""
Trading Performance Analyzer for Sniper Bot Pro.
Analyzes trading_data.csv and generates performance reports.

Usage:
    python analyzer.py              # Full analysis
    python analyzer.py --report     # Generate HTML report
    python analyzer.py --optimize   # Suggest parameter optimizations
"""

import os
import csv
import json
from datetime import datetime
from collections import defaultdict


class TradingAnalyzer:
    """Analyzes trading performance from CSV data."""
    
    def __init__(self, csv_file: str = "training_data.csv"):
        self.csv_file = csv_file
        self.trades = []
        self.load_data()
    
    def load_data(self):
        """Load trades from CSV file."""
        if not os.path.exists(self.csv_file):
            print(f"⚠️ File not found: {self.csv_file}")
            return
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            self.trades = list(reader)
        
        # Convert numeric fields
        valid_trades = []
        for t in self.trades:
            try:
                t['rsi'] = float(t.get('rsi', 50) or 50)
                t['stoch_rsi'] = float(t.get('stoch_rsi', 50) or 50)
                t['macd_hist'] = float(t.get('macd_hist', 0) or 0)
                t['atr'] = float(t.get('atr', 0) or 0)
                t['obi'] = float(t.get('obi', 1) or 1)
                t['cvd'] = float(t.get('cvd', 0) or 0)
                t['vol_ratio'] = float(t.get('vol_ratio', 1) or 1)
                t['score'] = int(t.get('score', 0) or 0)
                t['outcome'] = int(t.get('outcome', 0) or 0)
                t['pnl_percent'] = float(t.get('pnl_percent', 0) or 0)
                valid_trades.append(t)
            except (ValueError, TypeError):
                continue
        self.trades = valid_trades
        
        print(f"✅ Loaded {len(self.trades)} trades from {self.csv_file}")
    
    def get_basic_stats(self) -> dict:
        """Calculate basic trading statistics."""
        if not self.trades:
            return {"error": "No trades found"}
        
        total = len(self.trades)
        wins = [t for t in self.trades if t['outcome'] == 1]
        losses = [t for t in self.trades if t['outcome'] == 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        
        # Win Rate
        win_rate = (win_count / total * 100) if total > 0 else 0
        
        # Average PnL
        avg_win = sum(t['pnl_percent'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_percent'] for t in losses) / len(losses) if losses else 0
        
        # Profit Factor
        gross_profit = sum(t['pnl_percent'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_percent'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Total PnL
        total_pnl = sum(t['pnl_percent'] for t in self.trades)
        
        # Best/Worst
        best_trade = max(self.trades, key=lambda t: t['pnl_percent']) if self.trades else None
        worst_trade = min(self.trades, key=lambda t: t['pnl_percent']) if self.trades else None
        
        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calc_consecutive()
        
        return {
            'total_trades': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win_pct': round(avg_win, 4),
            'avg_loss_pct': round(avg_loss, 4),
            'total_pnl_pct': round(total_pnl, 4),
            'best_trade_pct': round(best_trade['pnl_percent'], 4) if best_trade else 0,
            'worst_trade_pct': round(worst_trade['pnl_percent'], 4) if worst_trade else 0,
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses,
        }
    
    def _calc_consecutive(self) -> tuple:
        """Calculate max consecutive wins and losses."""
        max_wins = max_losses = 0
        current_wins = current_losses = 0
        
        for t in self.trades:
            if t['outcome'] == 1:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def analyze_by_symbol(self) -> dict:
        """Analyze performance by trading symbol."""
        by_symbol = defaultdict(list)
        
        for t in self.trades:
            by_symbol[t.get('symbol', 'UNKNOWN')].append(t)
        
        results = {}
        for symbol, trades in by_symbol.items():
            wins = [t for t in trades if t['outcome'] == 1]
            total_pnl = sum(t['pnl_percent'] for t in trades)
            results[symbol] = {
                'trades': len(trades),
                'win_rate': round(len(wins) / len(trades) * 100, 2) if trades else 0,
                'total_pnl_pct': round(total_pnl, 4)
            }
        
        return results
    
    def analyze_by_side(self) -> dict:
        """Analyze performance by trade side (LONG/SHORT)."""
        by_side = defaultdict(list)
        
        for t in self.trades:
            by_side[t.get('side', 'UNKNOWN')].append(t)
        
        results = {}
        for side, trades in by_side.items():
            wins = [t for t in trades if t['outcome'] == 1]
            total_pnl = sum(t['pnl_percent'] for t in trades)
            results[side] = {
                'trades': len(trades),
                'win_rate': round(len(wins) / len(trades) * 100, 2) if trades else 0,
                'total_pnl_pct': round(total_pnl, 4)
            }
        
        return results
    
    def analyze_by_score(self) -> dict:
        """Analyze win rate by signal score."""
        by_score = defaultdict(list)
        
        for t in self.trades:
            by_score[t.get('score', 0)].append(t)
        
        results = {}
        for score, trades in sorted(by_score.items()):
            wins = [t for t in trades if t['outcome'] == 1]
            results[score] = {
                'trades': len(trades),
                'win_rate': round(len(wins) / len(trades) * 100, 2) if trades else 0,
            }
        
        return results
    
    def get_optimization_suggestions(self) -> list:
        """Suggest parameter optimizations based on data."""
        suggestions = []
        stats = self.get_basic_stats()
        by_score = self.analyze_by_score()
        by_side = self.analyze_by_side()
        
        # Check overall win rate
        if stats.get('win_rate', 0) < 50:
            suggestions.append("⚠️ Win Rate < 50%. Consider increasing score threshold from 4 to 5.")
        
        # Check which scores perform best
        best_score = None
        best_wr = 0
        for score, data in by_score.items():
            if data['trades'] >= 3 and data['win_rate'] > best_wr:
                best_wr = data['win_rate']
                best_score = score
        
        if best_score is not None and abs(best_score) > 4:
            suggestions.append(f"💡 Score {best_score} has {best_wr}% win rate. Consider using threshold {abs(best_score)}.")
        
        # Check LONG vs SHORT performance
        long_wr = by_side.get('BUY', {}).get('win_rate', 50)
        short_wr = by_side.get('SELL', {}).get('win_rate', 50)
        
        if long_wr > short_wr + 15:
            suggestions.append(f"📈 LONGs ({long_wr}%) outperform SHORTs ({short_wr}%). Consider LONG-only mode.")
        elif short_wr > long_wr + 15:
            suggestions.append(f"📉 SHORTs ({short_wr}%) outperform LONGs ({long_wr}%). Consider SHORT-only mode.")
        
        # Check profit factor
        if stats.get('profit_factor', 0) < 1.0:
            suggestions.append("🚨 Profit Factor < 1.0. Strategy is losing money. Review indicators.")
        elif stats.get('profit_factor', 0) < 1.2:
            suggestions.append("⚠️ Profit Factor < 1.2. Marginal profitability. Consider tighter entries.")
        
        if not suggestions:
            suggestions.append("✅ No major issues detected. Continue monitoring.")
        
        return suggestions
    
    def print_report(self):
        """Print a formatted performance report."""
        print("\n" + "="*60)
        print("📊 SNIPER BOT PRO - PERFORMANCE REPORT")
        print("="*60)
        
        stats = self.get_basic_stats()
        
        if 'error' in stats:
            print(f"\n{stats['error']}")
            return
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"   Win Rate: {stats['win_rate']}%")
        print(f"   Profit Factor: {stats['profit_factor']}")
        print(f"   Total PnL: {stats['total_pnl_pct']}%")
        print(f"   Best Trade: +{stats['best_trade_pct']}%")
        print(f"   Worst Trade: {stats['worst_trade_pct']}%")
        print(f"   Max Consecutive Wins: {stats['max_consecutive_wins']}")
        print(f"   Max Consecutive Losses: {stats['max_consecutive_losses']}")
        
        # By Symbol
        print(f"\n📊 BY SYMBOL:")
        for symbol, data in self.analyze_by_symbol().items():
            print(f"   {symbol}: {data['trades']} trades, {data['win_rate']}% WR, {data['total_pnl_pct']}% PnL")
        
        # By Side
        print(f"\n📊 BY SIDE:")
        for side, data in self.analyze_by_side().items():
            direction = "LONG" if side == "BUY" else "SHORT"
            print(f"   {direction}: {data['trades']} trades, {data['win_rate']}% WR, {data['total_pnl_pct']}% PnL")
        
        # By Score
        print(f"\n📊 BY SCORE:")
        for score, data in self.analyze_by_score().items():
            bar = "█" * int(data['win_rate'] / 10)
            print(f"   Score {score:+d}: {data['trades']:3d} trades, {data['win_rate']:5.1f}% WR {bar}")
        
        # Optimization Suggestions
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in self.get_optimization_suggestions():
            print(f"   {suggestion}")
        
        print("\n" + "="*60)
    
    def export_json(self, filename: str = "analysis_report.json"):
        """Export analysis to JSON file."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'basic_stats': self.get_basic_stats(),
            'by_symbol': self.analyze_by_symbol(),
            'by_side': self.analyze_by_side(),
            'by_score': self.analyze_by_score(),
            'suggestions': self.get_optimization_suggestions()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report exported to {filename}")


def main():
    import sys
    
    analyzer = TradingAnalyzer()
    
    if '--report' in sys.argv:
        analyzer.export_json()
    elif '--optimize' in sys.argv:
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for s in analyzer.get_optimization_suggestions():
            print(f"   {s}")
    else:
        analyzer.print_report()


if __name__ == "__main__":
    main()
