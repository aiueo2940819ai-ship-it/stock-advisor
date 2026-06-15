import yfinance as yf
from .indicators import calc_rsi, calc_mas, calc_change_pct


def get_stock_data(code: str) -> dict:
    try:
        df = yf.Ticker(f"{code}.T").history(period="6mo")
        if df.empty:
            return {"code": code, "error": "データなし"}

        prices  = df["Close"].tolist()
        volumes = df["Volume"].tolist()
        latest  = round(prices[-1], 1)
        prev    = round(prices[-2], 1) if len(prices) > 1 else latest

        avg_vol   = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        vol_ratio = round(volumes[-1] / avg_vol, 2) if avg_vol else 1.0

        # 52週高値・安値（または取得できる全期間）
        window   = prices[-252:] if len(prices) >= 252 else prices
        high_52w = round(max(window), 1)
        low_52w  = round(min(window), 1)

        return {
            "code":       code,
            "latest":     latest,
            "change_pct": round((latest - prev) / prev * 100, 2) if prev else 0,
            "change_5d":  calc_change_pct(prices, 5),
            "change_20d": calc_change_pct(prices, 20),
            **calc_mas(prices),
            "rsi14":     calc_rsi(prices),
            "vol_ratio": vol_ratio,
            "high_52w":  high_52w,
            "low_52w":   low_52w,
            "prices_20d": [round(p, 1) for p in prices[-20:]],
        }
    except Exception as e:
        return {"code": code, "error": str(e)}
