import json, os
import numpy as np
import pandas as pd
import yfinance as yf

from src.data.market import STOCK_TO_SECTOR_ETF, _JP_SECTOR_ETFS


def _load_universe() -> list[dict]:
    for path in ['data/nikkei225.json', 'data/universe.json']:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            print(f"  スクリーニング対象: {os.path.basename(path)} ({len(data)}銘柄)")
            return data
    return []


def screen_gekioshi_candidates(n_candidates: int = 5, existing_codes: set = None) -> list[dict]:
    """
    全ユニバース銘柄をPythonでスクリーニングして劇おすすめ候補を返す。
    Claudeへは上位n_candidates件のみ渡す（APIコスト増なし）。
    """
    universe = _load_universe()
    if not universe:
        return []

    existing_codes = existing_codes or set()
    targets  = [s for s in universe if s['code'] not in existing_codes]
    if not targets:
        return []

    codes    = [s['code'] for s in targets]
    code_map = {s['code']: s for s in targets}
    tickers  = [f"{c}.T" for c in codes]

    try:
        raw    = yf.download(tickers, period="6mo", auto_adjust=True,
                             progress=False, threads=True)
        closes = raw["Close"].ffill()
        highs  = raw["High"].ffill()
        lows   = raw["Low"].ffill()
        vols   = raw["Volume"].ffill()
    except Exception as e:
        print(f"  スクリーナーDLエラー: {e}")
        return []

    try:
        sec_raw = yf.download(list(_JP_SECTOR_ETFS.keys()), period="3mo",
                              auto_adjust=True, progress=False,
                              threads=True)["Close"].ffill()
    except Exception:
        sec_raw = pd.DataFrame()

    def rsi(s, p=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    def atr_latest(h, lo, c, p=14):
        tr = pd.concat([(h - lo),
                        (h - c.shift(1)).abs(),
                        (lo - c.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(p).mean().iloc[-1]

    candidates = []

    for code in codes:
        ticker = f"{code}.T"
        if ticker not in closes.columns:
            continue

        c = closes[ticker].dropna()
        if len(c) < 80:
            continue

        h  = highs[ticker].reindex(c.index).ffill()
        lo = lows[ticker].reindex(c.index).ffill()
        v  = vols[ticker].reindex(c.index).ffill()

        ma25    = c.rolling(25).mean()
        ma75    = c.rolling(75).mean()
        r_ser   = rsi(c)
        vol_avg = v.rolling(25).mean()

        price = c.iloc[-1]
        m25   = ma25.iloc[-1]
        m75   = ma75.iloc[-1]
        r     = r_ser.iloc[-1]
        chg5  = (price / c.iloc[-6] - 1) * 100 if len(c) >= 6 else 0
        va    = vol_avg.iloc[-1]
        vr    = float(v.iloc[-1] / va) if va > 0 else 1.0

        if any(pd.isna(x) for x in [price, m25, m75, r, vr]):
            continue

        # ── 定量スクリーニング条件 ──────────────────────────────
        if not (m25 > m75):        continue   # 上昇トレンド
        if not (40 <= r <= 60):    continue   # RSI適正ゾーン（劇おすすめ専用）
        if chg5 < -7:              continue   # 急落なし
        if not (0.7 <= vr <= 3.0): continue   # 出来高適正

        # セクター（✗のみ除外、不明は通す）
        etf = STOCK_TO_SECTOR_ETF.get(code)
        if etf and not sec_raw.empty and etf in sec_raw.columns:
            sec21 = sec_raw[etf].pct_change(21).iloc[-1]
            if not pd.isna(sec21) and sec21 < -0.03:
                continue

        # ATR・R/R
        try:
            atr_val = atr_latest(h, lo, c)
        except Exception:
            continue
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        stop     = max(price - atr_val * 2.5, price * 0.92)
        target   = price * 1.20
        downside = price - stop
        upside   = target - price
        if downside <= 0 or upside / downside < 2.0:
            continue

        # スコア（RSI位置・トレンド強度・R/R・出来高）
        rsi_score   = 1 - abs(r - 50) / 10
        trend_score = min((m25 - m75) / m75 * 100 / 5, 1.0)
        rr_score    = min(upside / downside / 4, 1.0)
        vol_score   = min(vr, 2.0) / 2.0
        score = (rsi_score * 0.35 + trend_score * 0.30
                 + rr_score * 0.20 + vol_score * 0.15)

        candidates.append({
            'code':               code,
            'name':               code_map[code].get('name', code),
            'latest':             round(price, 0),
            'ma25':               round(m25, 0),
            'ma75':               round(m75, 0),
            'rsi14':              round(r, 1),
            'atr14':              round(atr_val, 1),
            'vol_ratio':          round(vr, 2),
            'change_5d':          round(chg5, 2),
            'stop_est':           round(stop, 0),
            'rr_ratio':           round(upside / downside, 1),
            'score':              round(score, 3),
            'gekioshi_candidate': True,
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top = candidates[:n_candidates]

    if top:
        print(f"  劇おすすめ候補 {len(top)}件: "
              f"{[c['code']+' '+c['name'] for c in top]}")
    else:
        print("  劇おすすめ候補: 条件通過なし")

    return top
