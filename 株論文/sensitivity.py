#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings('ignore')

US = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XLC','XLRE']
JP = ['1617.T','1618.T','1619.T','1620.T','1621.T','1622.T','1623.T',
      '1624.T','1625.T','1626.T','1627.T','1628.T','1629.T','1630.T',
      '1631.T','1632.T','1633.T']
BENCH = '^N225'
NU, NJ = len(US), len(JP); N = NU + NJ
CYC_US = np.array([1,1,1,0,0,-1,-1,-1,-1,0,1], dtype=float)
CYC_JP = np.zeros(NJ)
for _i, _t in enumerate(JP):
    if _t in {'1618.T','1625.T','1629.T','1631.T'}: CYC_JP[_i] = 1
    if _t in {'1617.T','1621.T','1627.T','1630.T'}: CYC_JP[_i] = -1
L, K, LAM, AVG = 60, 3, 0.9, 5
TC = 0.0015
START = '2013-12-01'
TEST  = '2015-01-01'

def make_V0():
    cyc = np.r_[CYC_US, CYC_JP]
    v1 = np.ones(N) / N**0.5
    v2 = np.r_[np.ones(NU)/NU**0.5, -np.ones(NJ)/NJ**0.5]
    v2 -= (v2@v1)*v1; v2 /= np.linalg.norm(v2)
    v3 = cyc.copy(); v3 -= (v3@v1)*v1 + (v3@v2)*v2
    nm = np.linalg.norm(v3)
    if nm > 1e-10: v3 /= nm
    return np.c_[v1, v2, v3]

V0 = make_V0()

def pca_sub(win, us_t):
    mu = np.nanmean(win, 0); sd = np.nanstd(win, 0, ddof=1).clip(1e-10)
    z = np.nan_to_num((win-mu)/sd); C = z.T@z/max(len(z)-1, 1)
    D0 = np.diag(np.maximum(np.diag(V0.T@C@V0), 0))
    Cr = (1-LAM)*C + LAM*(V0@D0@V0.T)
    ev, ec = np.linalg.eigh(Cr)
    VK = ec[:, np.argsort(ev)[::-1][:K]]
    VU, VJ = VK[:NU], VK[NU:]
    z_us = np.nan_to_num((us_t - mu[:NU]) / sd[:NU])
    return VJ @ (VU.T @ z_us)

print("データ取得中...")
tickers = US + JP + [BENCH]
raw = yf.download(tickers, start=START, end='2025-12-31',
                  auto_adjust=True, progress=False)['Close'].ffill()
ret = raw.pct_change()
us_r = ret[US].values; jp_r = ret[JP].values; bn_r = ret[BENCH].values
dates = ret.index; T = len(dates)
ANN = 252

def run(alpha, freq):
    sig_buf = []; w = np.zeros(NJ); rows = []; cnt = 0
    for i in range(L, T-1):
        d = dates[i+1]
        if d < pd.Timestamp(TEST): continue
        win = np.c_[us_r[i-L:i], jp_r[i-L:i]]
        if np.isnan(win).mean() < 0.4:
            try: sig = pca_sub(win, us_r[i])
            except: sig = np.zeros(NJ)
        else:
            sig = np.zeros(NJ)
        sig_buf.append(sig)
        if len(sig_buf) > AVG: sig_buf.pop(0)
        cost = 0.0; cnt += 1
        if cnt % freq == 0:
            av = np.mean(sig_buf, 0)
            eq = np.ones(NJ) / NJ
            sp = np.maximum(av, 0)
            sp = sp / sp.sum() if sp.sum() > 1e-10 else eq.copy()
            new_w = (1-alpha)*eq + alpha*sp
            to = np.abs(new_w - w).sum() / 2
            cost = to * TC * 2; w = new_w
        jp_t = np.nan_to_num(jp_r[i+1])
        bn_v = float(bn_r[i+1]) if not np.isnan(bn_r[i+1]) else 0.0
        rows.append({'r': float(w@jp_t) - cost, 'b': bn_v})
    df_ = pd.DataFrame(rows)
    r_ = df_['r'].values
    ar = r_.mean()*ANN; risk = r_.std(ddof=1)*ANN**0.5
    rr = ar/risk if risk > 0 else 0
    cum = pd.Series(1+r_).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    tot = cum.iloc[-1] - 1
    return ar, risk, rr, mdd, tot

# 日経225 ベースライン
bn = bn_r[bn_r.shape[0] - len([i for i in range(L, T-1) if dates[i+1] >= pd.Timestamp(TEST)]):]
# 簡単に再計算
rows_b = []
for i in range(L, T-1):
    d = dates[i+1]
    if d < pd.Timestamp(TEST): continue
    rows_b.append(float(bn_r[i+1]) if not np.isnan(bn_r[i+1]) else 0.0)
r_b = np.array(rows_b)
ar_b = r_b.mean()*ANN; risk_b = r_b.std(ddof=1)*ANN**0.5; rr_b = ar_b/risk_b
cum_b = pd.Series(1+r_b).cumprod(); mdd_b = ((cum_b-cum_b.cummax())/cum_b.cummax()).min()
tot_b = cum_b.iloc[-1]-1
print(f"日経225: 年率{ar_b*100:.1f}%, R/R={rr_b:.3f}, 最大DD={mdd_b*100:.1f}%, 累積{tot_b*100:.0f}%\n")

# alpha 感応度（週次）
print("=" * 58)
print("alpha 感応度 (週次リバランス, freq=5)")
print("-" * 58)
print(f"{'alpha':<8} {'年率':>7} {'リスク':>7} {'R/R':>6} {'最大DD':>8} {'累積':>7}")
print("-" * 58)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    ar, risk, rr, mdd, tot = run(alpha, 5)
    mark = " ★" if rr > rr_b else ""
    print(f"a={alpha:<5} {ar*100:>6.1f}%  {risk*100:>6.1f}%  {rr:>6.3f}  {mdd*100:>7.1f}%  {tot*100:>6.0f}%{mark}")

# freq 感応度（alpha=0.3）
print()
print("=" * 58)
print("リバランス頻度感応度 (alpha=0.3)")
print("-" * 58)
print(f"{'頻度':<12} {'年率':>7} {'リスク':>7} {'R/R':>6} {'最大DD':>8} {'累積':>7}")
print("-" * 58)
for freq, label in [(1,'日次'), (5,'週次'), (10,'隔週'), (21,'月次'), (63,'四半期')]:
    ar, risk, rr, mdd, tot = run(0.3, freq)
    mark = " ★" if rr > rr_b else ""
    print(f"{label:<12} {ar*100:>6.1f}%  {risk*100:>6.1f}%  {rr:>6.3f}  {mdd*100:>7.1f}%  {tot*100:>6.0f}%{mark}")

print()
print("[ 参考 ] 論文オリジナル PCA SUB (コスト除外・ロングショート)")
print("  年率23.79%  リスク10.70%  R/R 2.22  最大DD -9.58%")
