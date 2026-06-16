#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版v2: 日経225先物ヘッジ付きロングオンリー戦略
─────────────────────────────────────────────────────
アイデア: 個人投資家でもショートなしでニュートラルを実現

  ①  シグナル上位セクターETFをロング
  ②  日経225ミニ先物 (相当額) をショート  ← ベータヘッジ
  ③  これで市場ベータ≈0、純粋なセクター選択シグナルを捕捉

先物コスト: 日経225ミニ先物は1枚=日経×100円, 証拠金約25万円
           個人投資家も楽天・SBI・松井証券で取引可能

数式（改良版v2）:
  ロング: w_long_j = max(0, s_j) / Σ_k max(0,s_k)  [M1と同一]
  ショート: w_short = -β * 1  ← 日経先物 β=1 で LS再現

  ポートフォリオリターン = Σ w_long_j * r_jp_j + (-β) * r_bench
  これは LS リターンを βを調整して近似

v3: インデックスタイル版
  市場リスクを完全に除かず、インデックスを70%保有してリスクを分散
  w = 0.7 * eq_weight + 0.3 * signal_weight

実行: python 株論文/backtest_v2_hedged.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── 設定（v1と同一） ─────────────────────────────
US = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XLC','XLRE']
JP = ['1617.T','1618.T','1619.T','1620.T','1621.T','1622.T','1623.T',
      '1624.T','1625.T','1626.T','1627.T','1628.T','1629.T','1630.T',
      '1631.T','1632.T','1633.T']
JP_NAME = ['食品','エネルギー','建設','素材化学','医薬品','自動車',
           '鉄鋼','機械','電機精密','情報通信','電力ガス','運輸',
           '商社','小売','銀行','金融','不動産']
BENCH = '^N225'   # 日経225（先物ヘッジ対象）
TOPIX = '1306.T'  # TOPIX ETF (インデックスタイル版用)

NU, NJ = len(US), len(JP)
N = NU + NJ

CYC_US = np.array([1,1,1,0,0,-1,-1,-1,-1,0,1], dtype=float)
CYC_JP = np.zeros(NJ)
for _i, _t in enumerate(JP):
    if _t in {'1618.T','1625.T','1629.T','1631.T'}: CYC_JP[_i] =  1
    if _t in {'1617.T','1621.T','1627.T','1630.T'}: CYC_JP[_i] = -1

L    = 60
K    = 3
LAM  = 0.9
AVG  = 5
FREQ = 5
TOP  = 4
TC   = 0.0015   # 片道0.15%

START = '2013-12-01'
TEST  = '2015-01-01'

def make_V0():
    cyc = np.r_[CYC_US, CYC_JP]
    v1 = np.ones(N) / N**0.5
    v2 = np.r_[np.ones(NU)/NU**0.5, -np.ones(NJ)/NJ**0.5]
    v2 -= (v2@v1)*v1; v2 /= np.linalg.norm(v2)
    v3 = cyc.copy()
    v3 -= (v3@v1)*v1 + (v3@v2)*v2
    nm = np.linalg.norm(v3)
    if nm > 1e-10: v3 /= nm
    return np.c_[v1, v2, v3]

V0 = make_V0()

def pca_sub_signal(ret_win, us_today):
    mu = np.nanmean(ret_win, axis=0)
    sd = np.nanstd(ret_win, axis=0, ddof=1).clip(1e-10)
    z  = np.nan_to_num((ret_win - mu) / sd)
    C  = z.T @ z / max(len(z)-1, 1)
    D0 = np.diag(np.maximum(np.diag(V0.T @ C @ V0), 0))
    Cr = (1-LAM)*C + LAM*(V0 @ D0 @ V0.T)
    evals, evecs = np.linalg.eigh(Cr)
    VK = evecs[:, np.argsort(evals)[::-1][:K]]
    VU, VJ = VK[:NU], VK[NU:]
    z_us = np.nan_to_num((us_today - mu[:NU]) / sd[:NU])
    return VJ @ (VU.T @ z_us)

def long_only_alloc(sig, top=TOP):
    w = np.zeros(NJ)
    rank = np.argsort(sig)[::-1]
    pos = [r for r in rank[:top] if sig[r] > 0]
    if pos:
        s = sig[pos]; w[pos] = s / s.sum()
    return w

def index_tilt_alloc(sig, alpha=0.3):
    """
    インデックスタイル: (1-α)*等ウェイト + α*シグナルウェイト
    α=0.3 → 30%だけシグナルで傾ける
    """
    eq = np.ones(NJ) / NJ
    sig_pos = np.maximum(sig, 0)
    if sig_pos.sum() > 1e-10:
        sw = sig_pos / sig_pos.sum()
    else:
        sw = eq
    return (1 - alpha) * eq + alpha * sw

def download():
    tickers = US + JP + [BENCH, TOPIX]
    print("データ取得中...")
    raw = yf.download(tickers, start=START, end='2025-12-31',
                      auto_adjust=True, progress=False)['Close']
    raw = raw.ffill()
    ret = raw.pct_change()
    print(f"取得完了: {ret.index[0].date()} ～ {ret.index[-1].date()}")
    return ret

def run_strategy(ret, mode='long_only'):
    """
    mode:
      'long_only'  - v1: ロングのみ（先のバックテスト）
      'hedged'     - v2: ロング + 日経先物ショート（ベータ中立）
      'index_tilt' - v3: インデックスタイル（30%傾斜）
    """
    us_r  = ret[US].values
    jp_r  = ret[JP].values
    bn_r  = ret[BENCH].values
    dates = ret.index
    T     = len(dates)

    sig_buf = []
    w       = np.zeros(NJ) if mode != 'long_only' else np.zeros(NJ)
    rows    = []
    cnt     = 0

    for i in range(L, T - 1):
        d = dates[i + 1]
        if d < pd.Timestamp(TEST): continue

        win = np.c_[us_r[i-L:i], jp_r[i-L:i]]
        if np.isnan(win).mean() < 0.4:
            try: sig = pca_sub_signal(win, us_r[i])
            except: sig = np.zeros(NJ)
        else:
            sig = np.zeros(NJ)

        sig_buf.append(sig)
        if len(sig_buf) > AVG: sig_buf.pop(0)

        cost = 0.0
        cnt += 1
        if cnt % FREQ == 0:
            avg_sig = np.mean(sig_buf, axis=0)
            if mode == 'long_only':
                new_w = long_only_alloc(avg_sig)
            elif mode == 'hedged':
                new_w = long_only_alloc(avg_sig)   # ロングは同じ
            elif mode == 'index_tilt':
                new_w = index_tilt_alloc(avg_sig, alpha=0.3)

            to   = np.abs(new_w - w).sum() / 2
            cost = to * TC * 2
            w    = new_w

        jp_tmrw = np.nan_to_num(jp_r[i + 1])
        bn_tmrw = float(bn_r[i + 1]) if not np.isnan(bn_r[i + 1]) else 0.0

        strat_ret = float(w @ jp_tmrw) - cost

        if mode == 'hedged':
            # 日経先物ショート: -1倍の日経リターンを加算
            # コスト: 日経ミニ先物のロールコスト≈年0.5% ≈ 日次0.002%
            hedge_cost = 0.00002
            strat_ret += (-1.0) * bn_tmrw - hedge_cost

        rows.append({'date': d, 'ret': strat_ret, 'bench': bn_tmrw})

    return pd.DataFrame(rows).set_index('date')

def stats(r):
    ANN  = 252
    r    = np.asarray(r)
    ar   = r.mean() * ANN
    risk = r.std(ddof=1) * np.sqrt(ANN)
    rr   = ar / risk if risk > 1e-10 else 0.0
    cum  = pd.Series(1 + r).cumprod()
    mdd  = ((cum - cum.cummax()) / cum.cummax()).min()
    tot  = cum.iloc[-1] - 1
    # 市場ベータ計算（後で追加する）
    return dict(ar=ar, risk=risk, rr=rr, mdd=mdd, tot=tot)

def beta(r_strat, r_bench):
    """市場ベータ（ストラテジーの市場感応度）"""
    r1 = np.asarray(r_strat)
    r2 = np.asarray(r_bench)
    cov = np.cov(r1, r2)
    return cov[0,1] / cov[1,1] if cov[1,1] > 0 else 0.0

def report_all(dfs):
    ANN = 252
    labels = {
        'long_only'  : 'v1: ロングのみ (今回)',
        'hedged'     : 'v2: ロング+日経先物ヘッジ',
        'index_tilt' : 'v3: インデックスタイル(α=30%)',
    }
    print("\n" + "="*70)
    print(f"  評価期間: {TEST} ～ 2025-12-30")
    print("="*70)
    print(f"{'戦略':<32} {'年率':>7} {'リスク':>7} {'R/R':>6} {'最大DD':>8} {'β':>5}")
    print("-"*70)

    # 日経225
    bn = dfs['long_only']['bench'].fillna(0)
    s  = stats(bn)
    print(f"{'日経225（ベンチマーク）':<32} {s['ar']*100:>6.1f}% {s['risk']*100:>6.1f}% {s['rr']:>6.3f} {s['mdd']*100:>7.1f}%  1.00")

    for mode, df_ in dfs.items():
        r_ = df_['ret'].fillna(0)
        b_ = df_['bench'].fillna(0)
        s_ = stats(r_)
        β  = beta(r_, b_)
        print(f"{labels[mode]:<32} {s_['ar']*100:>6.1f}% {s_['risk']*100:>6.1f}% {s_['rr']:>6.3f} {s_['mdd']*100:>7.1f}%  {β:>4.2f}")

    print("="*70)
    print("※ v2の先物ヘッジコスト: 年約1.0%（ロールコスト0.5% + スプレッド）")
    print("  証拠金: 日経225ミニ1枚≈25万円（ポジション100万円規模で1枚程度）")

    # 年別比較（v1 vs v2 vs v3）
    print("\n年別リターン:")
    print(f"{'年':<5} {'v1ロング':>9} {'v2ヘッジ':>9} {'v3タイル':>9} {'日経225':>9}")
    print("-"*46)

    for mode in ['long_only','hedged','index_tilt']:
        dfs[mode].index = pd.to_datetime(dfs[mode].index)

    ann = {}
    for mode in ['long_only','hedged','index_tilt']:
        ann[mode] = dfs[mode].resample('YE').apply(lambda x: (1+x).prod()-1)

    years = ann['long_only'].index
    for yr in years:
        v1 = float(ann['long_only'].loc[yr, 'ret'])
        v2 = float(ann['hedged'].loc[yr, 'ret'])
        v3 = float(ann['index_tilt'].loc[yr, 'ret'])
        bn = float(ann['long_only'].loc[yr, 'bench'])
        print(f"{yr.year:<5} {v1*100:>+8.1f}%  {v2*100:>+8.1f}%  {v3*100:>+8.1f}%  {bn*100:>+8.1f}%")


def print_implementation_guide():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v2（先物ヘッジ版）の個人投資家向け実装手順
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【必要口座】
  - 国内株式: SBI証券/楽天証券（NEXT FUNDS TOPIX-17 ETF売買）
  - 先物口座: 同証券会社の先物取引口座（日経225ミニ）

【毎週月曜の作業フロー】
  1. 金曜夜: 米国11セクターETFの終値取得 → シグナル計算（自動化済）
  2. 月曜朝: 上位4セクターETFを等分購入（始値）
  3. 月曜朝: 日経225ミニ先物を購入金額相当ショート
  4. 金曜夕: 翌週まで保持（先物の移行コスト週1回）

【コスト概算（100万円規模）】
  - ETF取引: 往復0.3% × 週次 = 年15.6%（高い！→ 月次なら3.1%）
  - 先物ヘッジ: 年0.5-1.0%（ロールコスト+スプレッド）
  合計: 年4-17%のコスト（月次なら収益が残る可能性大）

【現実的な推奨: 月次リバランス版を試す】
  - 取引コスト: 年約3-4%
  - もし年10-15%のグロスアルファがあれば: 純利益6-12%
""")


if __name__ == '__main__':
    print("=" * 60)
    print("  改良版v2: 3戦略比較バックテスト")
    print("  v1:ロングのみ / v2:先物ヘッジ / v3:インデックスタイル")
    print("=" * 60)

    ret = download()

    dfs = {}
    for mode in ['long_only', 'hedged', 'index_tilt']:
        print(f"  {mode} 実行中...")
        dfs[mode] = run_strategy(ret, mode=mode)

    report_all(dfs)
    print_implementation_guide()
