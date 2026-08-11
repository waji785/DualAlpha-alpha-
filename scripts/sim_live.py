#!/usr/bin/env python
# scripts/sim_live.py
"""盘中模拟：盘前信号 + 价格区间 + 量确认 + T+1 + 交易费率"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import time, json
from datetime import datetime as dt
import pandas as pd
from config.settings import *
from utils.logger import setup_logger

logger = setup_logger(__name__)

SIGNALS_FILE = os.path.join(OUTPUT_DIR, "today_signals.csv")
POSITION_FILE = os.path.join(OUTPUT_DIR, "positions.json")
TRADE_LOG = os.path.join(OUTPUT_DIR, "trade_log.csv")
POLL_INTERVAL = 60  # 秒
VOL_SPIKE = 1.5
COMMISSION = 0.0003    # 佣金 万3
STAMP_TAX = 0.001      # 印花税（仅卖出）
MAX_TOTAL_POS = 0.8    # 总仓位上限


class SimTrader:
    def __init__(self):
        self.positions = {}       # {code: {shares, cost, entry_dt, today_bought}}
        self.cash = 1_000_000.0
        self.today_bought = set() # T+1
        self.initial_capital = 1_000_000.0
        self.signals = {}
        self.prev_vol = {}
        self.tick_count = 0
        self.fetch_ok = 0
        self.fetch_fail = 0

    def load_signals(self):
        if not os.path.exists(SIGNALS_FILE):
            logger.error(f"缺少 {SIGNALS_FILE}")
            return False
        df = pd.read_csv(SIGNALS_FILE)
        for _, r in df.iterrows():
            code = str(r['code']).zfill(6)
            self.signals[code] = {
                'signal': r['signal'], 'up_prob': float(r.get('up_prob', 0)),
                'close': float(r.get('close', 0)),
                'entry_low': float(r.get('entry_low', 0)),
                'entry_high': float(r.get('entry_high', 0)),
                'target_price': float(r.get('target_price', 0)),
            }
        buys = sum(1 for s in self.signals.values() if s['signal'] == '买入')
        logger.info(f"盘前信号: {len(self.signals)}只, 买入{buys}只")
        self.regime_mult = float(df['regime_mult'].iloc[0]) if 'regime_mult' in df.columns else 1.0
        if self.regime_mult < 1.0:
            logger.warning(f"⚠ 仓位系数={self.regime_mult}")
        self.load_positions()
        return True

    def load_positions(self):
        if os.path.exists(POSITION_FILE):
            with open(POSITION_FILE) as f:
                d = json.load(f)
            self.positions = d.get('positions', {})
            self.cash = d.get('cash', 1_000_000.0)
            self.today_bought = set(d.get('today_bought', []))

    def save_positions(self):
        with open(POSITION_FILE, 'w') as f:
            json.dump({
                'positions': self.positions, 'cash': self.cash,
                'today_bought': list(self.today_bought),
            }, f, indent=2)

def fetch_quotes(self):
    codes = set(self.signals.keys()) | set(self.positions.keys())
    if not codes: return {}
    # sina 实时行情（主力）
    try:
        import requests, re
        sina = ','.join(f"{'sh' if c.startswith('6') else 'sz'}{c}" for c in codes)
        r = requests.get(f"http://hq.sinajs.cn/list={sina}",
                         headers={'Referer':'http://finance.sina.com.cn'}, timeout=5)
        r.encoding='gbk'; quotes={}
        for m in re.finditer(r'hq_str_(\w+)="(.+?)"', r.text):
            code=m.group(1)[2:]; p=m.group(2).split(',')
            if len(p)>=32 and float(p[3])>0:
                quotes[code]={'price':float(p[3]),'volume':float(p[8]),
                    'high':float(p[4]),'low':float(p[5]),
                    'pct':(float(p[3])/float(p[2])-1)*100}
        if quotes: self.fetch_ok+=1; return quotes
    except: pass
    # akshare 备选（5分钟一次防封）
    if self.tick_count%60==0:
        try:
            import akshare as ak; df=ak.stock_zh_a_spot_em(); quotes={}
            for _,r in df.iterrows():
                c=str(r['代码']).zfill(6)
                if c in codes and c not in quotes:
                    quotes[c]={'price':float(r['最新价']),'volume':float(r['成交量']),
                        'high':float(r['最高']),'low':float(r['最低']),'pct':float(r['涨跌幅'])}
            self.fetch_ok+=1; return quotes
        except: self.fetch_fail+=1
    self.fetch_ok+=1; return {}

    def _trade_cost(self, amount, is_sell=False):
        """实际到账金额 = 成交额 - 佣金 - 印花税"""
        c = amount * COMMISSION
        if is_sell:
            c += amount * STAMP_TAX
        return amount - c

    def _total_invested(self):
        return sum(p['shares'] * self.signals.get(c, {}).get('close', p['cost'])
                   for c, p in self.positions.items())

    def on_tick(self, quotes):
        for code, q in quotes.items():
            if code not in self.signals:
                continue
            sig = self.signals[code]
            price = q['price']

            # 涨跌停跳过
            if abs(q['pct']) >= 9.5:
                continue

            # ---- T+1: 今日买入的不可卖出 ----
            if code in self.positions and code not in self.today_bought:
                pos = self.positions[code]
                profit = (price - pos['cost']) / pos['cost']
                if profit >= TAKE_PROFIT:
                    self._sell(code, price, pos['shares'], '止盈', profit,
                               f"盈{profit*100:.1f}%≥{TAKE_PROFIT*100:.0f}%"); continue
                if profit <= STOP_LOSS:
                    self._sell(code, price, pos['shares'], '止损', profit,
                               f"亏{profit*100:.1f}%≤{STOP_LOSS*100:.0f}%"); continue
                if price >= sig['target_price'] and profit > 0.03:
                    self._sell(code, price, pos['shares'], '达目标价', profit,
                               f"现价{price:.2f}≥目标{sig['target_price']:.2f}"); continue

            # ---- 买入 ----
            if sig['signal'] != '买入' or code in self.positions:
                continue
            if price < sig['entry_low'] or price > sig['entry_high']:
                continue

            prev_v = self.prev_vol.get(code, 0)
            if prev_v > 0 and q['volume'] < prev_v * VOL_SPIKE:
                self.prev_vol[code] = q['volume']
                continue

            target_pos = min((sig['up_prob'] - 0.45) * 1.0, MAX_POSITION) * self.regime_mult
            if price < sig['close'] * 0.97:
                target_pos *= 0.5

            # 总仓位上限
            already = self._total_invested() / self.initial_capital
            if already >= MAX_TOTAL_POS:
                continue
            target_pos = min(target_pos, MAX_TOTAL_POS - already)

            gross = self.cash * target_pos
            cost = gross * (1 + COMMISSION)
            shares = int(gross / price / 100) * 100
            if shares >= 100 and cost <= self.cash:
                detail = f"up={sig['up_prob']:.3f} 价∈[{sig['entry_low']:.2f},{sig['entry_high']:.2f}]"
                self._buy(code, price, shares, detail)
            self.prev_vol[code] = q['volume']

    def _buy(self, code, price, shares, detail=""):
        gross = shares * price
        cost = gross * (1 + COMMISSION)
        self.cash -= cost
        self.positions[code] = {'shares': shares, 'cost': price,
                                'entry_dt': dt.now().strftime("%H:%M")}
        self.today_bought.add(code)
        self._log(code, '买入', price, shares, 0, detail)
        logger.info(f"🟢 买入 {code} {shares}股 @{price:.2f} 费¥{gross*COMMISSION:.0f} | 现金¥{self.cash:,.0f}")
        logger.info(f"   依据: {detail}")

    def _sell(self, code, price, shares, reason, profit, detail=""):
        pos = self.positions.get(code)
        if not pos: return
        s = min(shares, pos['shares'])
        gross = s * price
        net = self._trade_cost(gross, is_sell=True)
        self.cash += net
        pos['shares'] -= s
        if pos['shares'] == 0:
            del self.positions[code]
        self._log(code, reason, price, s, profit, detail)
        logger.info(f"{'✅' if profit>0 else '❌'} {reason} {code} {s}股 @{price:.2f} "
                    f"费¥{gross-net:.0f} {profit*100:+.1f}%")
        logger.info(f"   依据: {detail}")

    def _log(self, code, action, price, shares, profit, detail=""):
        pd.DataFrame([{
            'time': dt.now().strftime("%H:%M:%S"), 'code': code,
            'action': action, 'price': price, 'shares': shares,
            'profit': round(profit, 4), 'cash': round(self.cash, 2),
            'detail': detail,
        }]).to_csv(TRADE_LOG, mode='a',
                   header=not os.path.exists(TRADE_LOG),
                   index=False, encoding='utf-8-sig')

    def print_status(self, quotes):
        if not self.positions: return
        total = self.cash
        for c, p in self.positions.items():
            total += p['shares'] * quotes.get(c, {}).get('price', p['cost'])
        logger.info(f"💰 总¥{total:,.0f} | 盈亏¥{total-self.initial_capital:+,.0f} | {len(self.positions)}只")

    def is_trading(self):
        n = dt.now()
        t = n.strftime("%H:%M")
        if n.weekday() >= 5: return False
        return ("09:25" <= t <= "11:30") or ("13:00" <= t <= "14:57")

    def run(self):
        if not self.load_signals(): return
        logger.info("模拟盘启动")
        while True:
            if not self.is_trading():
                if self.tick_count == 0:
                    logger.info(f"⏳ 非交易时间 ({dt.now().strftime('%H:%M')}), 等待中...")
                    self.tick_count = 1
                time.sleep(POLL_INTERVAL); continue
            q = self.fetch_quotes()
            self.tick_count += 1
            if q:
                self.on_tick(q)
            # 每次获取数据都打心跳
                self.print_status(q) if self.positions else None
                nq = len(q)
                logger.info(f"💓 {dt.now().strftime('%H:%M')} 行情{nq}只 "
                           f"持仓{len(self.positions)} OK={self.fetch_ok} 失败={self.fetch_fail}")
            self.save_positions()
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    SimTrader().run()
