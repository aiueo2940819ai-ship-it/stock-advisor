#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
個人投資家向け改良版バックテスト: 日米業種リードラグ戦略
元論文: 部分空間正則化付きPCAを用いた日米業種リードラグ投資

4改良:
  [M1] ロングオンリー     w_j = max(0,s_j)/sum(max(0,s))
  [M2] 週次リバランス     5営業日ごとのみ売買
  [M3] 多日シグナル平均   s_bar = mean(s_{t-A+1},...,s_t), A=5
  [M4] 取引コスト明示     片道0.15% を往復差し引き

実行: python 株論文/backtest_improved.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── 銘柄設定 ───────────────────────────────────────
US = ['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XLC','XLRE']
JP = ['1617.T','1618.T','1619.T','1620.T','1621.T','1622.T','1623.T',
      '1624.T','1625.T','1626.T','1627.T','1628.T','1629.T','1630.T',
      '1631.T','1632.T','1633.T']
JP_NAME = ['食品','エネルギー','建設','素材化学','医薬品','自動車',
           '鉄鋼','機械','電機精密','情報通信','電力ガス','運輸',
           '商社','小売','銀行','金融','不動産']
BENCH = '^N225'

NU, NJ = len(US), len(JP)   # 11, 17
N = NU + NJ                  # 28

# シクリカル(+1)/ニュートラル(0)/ディフェンシブ(-1) — 論文 p.79 と同一
CYC_US = np.array([1,1,1,0,0,-1,-1,-1,-1,0,1], dtype=float)
# 順: XLB XLE XLF XLI XLK XLP XLU XLV XLY XLC XLRE
CYC_JP = np.zeros(NJ)
for _i, _t in enumerate(JP):
    if _t in {'1618.T','1625.T','1629.T','1631.T'}: CYC_JP[_i] =  1
    if _t in {'1617.T','1621.T','1627.T','1630.T'}: CYC_JP[_i] = -1

# ─── ハイパーパラメータ ─────────────────────────────
L    = 60      # 推定ウィンドウ（論文と同一）
K    = 3       # 主成分数（論文と同一）
LAM  = 0.9    # 正則化係数（論文と同一）
AVG  = 5      # [M3] シグナル平均日数
FREQ = 5      # [M2] リバランス間隔（週次）
TOP  = 4      # [M1] ロング上位セクター数
TC   = 0.0015 # [M4] 片道取引コスト 0.15%

START = '2013-12-01'   # ウィンドウ初期化用（テスト前L日確保）
TEST  = '2015-01-01'   # バックテスト評価開始

# ─── 事前部分空間 V0 (N×3) ─────────────────────────
def make_V0():
    """論文 Sec.3.1 の v1(グローバル), v2(国スプレッド), v3(シクリカル)"""
    cyc = np.r_[CYC_US, CYC_JP]

    v1 = np.ones(N) / N**0.5

    v2 = np.r_[np.ones(NU)/NU**0.5, -np.ones(NJ)/NJ**0.5]
    v2 -= (v2 @ v1) * v1
    v2 /= np.linalg.norm(v2)

    v3 = cyc.copy()
    v3 -= (v3 @ v1)*v1 + (v3 @ v2)*v2
    nm = np.linalg.norm(v3)
    if nm > 1e-10: v3 /= nm

    return np.c_[v1, v2, v3]   # (N, 3)

V0 = make_V0()

# ─── PCA-SUB シグナル（論文 Sec.3.2-3.3 と同一）──────
def pca_sub_signal(ret_win, us_today):
    """
    ret_win  : (L, N)  過去L日 全銘柄リターン（NaN含む可）
    us_today : (NU,)   当日 米国終値-to-終値リターン
    returns  : (NJ,)   日本翌日シグナル
    """
    # 標準化（論文 eq.8-9）
    mu = np.nanmean(ret_win, axis=0)
    sd = np.nanstd(ret_win, axis=0, ddof=1).clip(1e-10)
    z  = np.nan_to_num((ret_win - mu) / sd)

    # サンプル相関行列 Ct（論文 eq.12）
    C  = z.T @ z / max(len(z) - 1, 1)

    # 事前行列 C0（論文 eq.10-11）
    D0 = np.diag(np.maximum(np.diag(V0.T @ C @ V0), 0))
    C0 = V0 @ D0 @ V0.T

    # 正則化相関行列（論文 eq.13）
    Cr = (1 - LAM) * C + LAM * C0

    # 固有値分解（論文 eq.14-15）
    evals, evecs = np.linalg.eigh(Cr)
    VK = evecs[:, np.argsort(evals)[::-1][:K]]   # (N, K) 上位K固有ベクトル

    # 米国・日本ブロック分割（論文 eq.16）
    VU, VJ = VK[:NU], VK[NU:]

    # ファクタースコア → 日本シグナル（論文 eq.18-19）
    z_us = np.nan_to_num((us_today - mu[:NU]) / sd[:NU])
    return VJ @ (VU.T @ z_us)   # (NJ,)


# ─── [M1] ロングオンリー配分 ─────────────────────────
def long_only_alloc(sig):
    """シグナル上位 TOP セクターにシグナル強度比例で配分（ロングのみ）"""
    w    = np.zeros(NJ)
    rank = np.argsort(sig)[::-1]
    pos  = [r for r in rank[:TOP] if sig[r] > 0]
    if pos:
        s = sig[pos]
        w[pos] = s / s.sum()
    return w


# ─── データ取得 ────────────────────────────────────
def download():
    tickers = US + JP + [BENCH]
    print("データ取得中（yfinance）...")
    raw = yf.download(tickers, start=START, end='2025-12-31',
                      auto_adjust=True, progress=False)['Close']
    raw = raw.ffill()
    ret = raw.pct_change()
    print(f"取得完了: {ret.index[0].date()} ～ {ret.index[-1].date()}, {len(ret)}営業日")
    return ret


# ─── バックテスト ──────────────────────────────────
def backtest(ret):
    us_r  = ret[US].values     # (T, NU)
    jp_r  = ret[JP].values     # (T, NJ)
    bn_r  = ret[BENCH].values  # (T,)
    dates = ret.index
    T     = len(dates)

    sig_buf = []          # 直近 AVG 日のシグナル [M3]
    w       = np.zeros(NJ)
    rows    = []
    cnt     = 0           # テスト開始後カウンタ（リバランス判定）

    for i in range(L, T - 1):
        d = dates[i + 1]   # 翌日 = 実際の取引日（信号は当日米国終値で確定）
        if d < pd.Timestamp(TEST):
            continue

        win = np.c_[us_r[i-L:i], jp_r[i-L:i]]   # (L, N)
        nan_ok = np.isnan(win).mean() < 0.4

        if nan_ok:
            try:
                sig = pca_sub_signal(win, us_r[i])
            except Exception:
                sig = np.zeros(NJ)
        else:
            sig = np.zeros(NJ)

        # [M3] シグナルバッファ
        sig_buf.append(sig)
        if len(sig_buf) > AVG:
            sig_buf.pop(0)

        # [M2] 週次リバランス
        cost = 0.0
        cnt += 1
        if cnt % FREQ == 0:
            avg_sig = np.mean(sig_buf, axis=0)
            new_w   = long_only_alloc(avg_sig)   # [M1]
            to      = np.abs(new_w - w).sum() / 2
            cost    = to * TC * 2                # [M4] 往復コスト
            w       = new_w

        jp_tmrw = np.nan_to_num(jp_r[i + 1])
        bn_v    = float(bn_r[i + 1]) if not np.isnan(bn_r[i + 1]) else 0.0

        rows.append({'date': d, 'ret': float(w @ jp_tmrw) - cost, 'bench': bn_v})

    return pd.DataFrame(rows).set_index('date')


# ─── パフォーマンス評価 ────────────────────────────
def stats(r):
    ANN  = 252
    r    = np.asarray(r)
    ar   = r.mean() * ANN
    risk = r.std(ddof=1) * np.sqrt(ANN)
    rr   = ar / risk if risk > 1e-10 else 0.0
    cum  = pd.Series(1 + r).cumprod()
    mdd  = ((cum - cum.cummax()) / cum.cummax()).min()
    tot  = cum.iloc[-1] - 1
    return dict(ar=ar, risk=risk, rr=rr, mdd=mdd, tot=tot)


def report(df):
    s = stats(df['ret'].fillna(0))
    b = stats(df['bench'].fillna(0))

    print("\n" + "=" * 56)
    print(f"  評価期間: {df.index[0].date()} ～ {df.index[-1].date()}")
    print(f"  設定: L={L}, K={K}, λ={LAM}, 平均{AVG}日, 週次, TOP{TOP}セクター")
    print(f"  コスト: 片道{TC*100:.2f}% (往復{TC*2*100:.2f}%/回, 週次)")
    print("=" * 56)
    print(f"{'指標':<16} {'改良版PCA-SUB':>14} {'日経225':>10}")
    print("-" * 56)
    print(f"{'年率リターン':<16} {s['ar']*100:>13.2f}%  {b['ar']*100:>8.2f}%")
    print(f"{'年率リスク':<16} {s['risk']*100:>13.2f}%  {b['risk']*100:>8.2f}%")
    print(f"{'R/R (Sharpe)':<16} {s['rr']:>15.3f}  {b['rr']:>10.3f}")
    print(f"{'最大DD':<16} {s['mdd']*100:>13.2f}%  {b['mdd']*100:>8.2f}%")
    print(f"{'累積リターン':<16} {s['tot']*100:>13.1f}%  {b['tot']*100:>8.1f}%")
    print("=" * 56)

    # 年別
    df2 = df.copy()
    df2.index = pd.to_datetime(df2.index)
    ann = df2.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    print(f"\n{'年':<5} {'改良版':>9} {'日経225':>9} {'超過':>8}  勝敗")
    print("-" * 40)
    wins = 0
    for yr in ann.index:
        s_ = float(ann.loc[yr, 'ret'])
        b_ = float(ann.loc[yr, 'bench'])
        ex  = s_ - b_
        mk  = "◎" if ex > 0.05 else ("○" if ex > 0 else "×")
        if ex > 0: wins += 1
        print(f"{yr.year:<5} {s_*100:>+8.1f}%  {b_*100:>+8.1f}%  {ex*100:>+7.1f}%  {mk}")
    total_yrs = len(ann)
    print(f"\n  日経225超過: {wins}/{total_yrs}年 ({wins/total_yrs*100:.0f}%)")

    # 参考: 論文オリジナルの数値
    print("\n─── 論文オリジナル PCA SUB（参考・コスト除外・ショートあり）────")
    print("  年率リターン: 23.79%  リスク: 10.70%  R/R: 2.22  最大DD: -9.58%")
    print("  ※ 改良版との差異 = ①ロング片脚のみ②コスト考慮③週次による")


# ─── 追加: 感応度分析（TOPの違い）─────────────────
def sensitivity_top(ret):
    print("\n─── 感応度: TOP N セクター数の影響 ──────────────────")
    print(f"{'TOP_N':<8} {'年率':>9} {'R/R':>8} {'最大DD':>9}")
    print("-" * 38)
    for top_n in [2, 3, 4, 5, 6]:
        global TOP
        TOP = top_n
        df_ = backtest(ret)
        s_  = stats(df_['ret'].fillna(0))
        print(f"TOP={top_n:<4} {s_['ar']*100:>+8.2f}%  {s_['rr']:>7.3f}  {s_['mdd']*100:>8.2f}%")
    TOP = 4  # 元に戻す


if __name__ == '__main__':
    print("=" * 60)
    print("  個人投資家向け改良版 日米業種リードラグ戦略バックテスト")
    print("  改良: ①ロングのみ ②週次 ③5日平均 ④コスト0.15%/片道")
    print("=" * 60)

    ret = download()
    df  = backtest(ret)
    report(df)
    sensitivity_top(ret)
