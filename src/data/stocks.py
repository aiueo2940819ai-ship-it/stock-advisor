import math
import yfinance as yf
from datetime import date
from .indicators import calc_rsi, calc_mas, calc_change_pct


def _f(val, ndigits: int = 1):
    """NaN・Inf を None に変換して安全な float を返す"""
    if val is None:
        return None
    try:
        v = round(float(val), ndigits)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _clean_prices(prices: list) -> list:
    """価格リストから NaN を除去する"""
    return [p for p in prices if p is not None and not math.isnan(p)]


def _get_next_earnings(ticker) -> tuple[str | None, int | None]:
    try:
        cal = ticker.calendar
        if not cal:
            return None, None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif hasattr(cal, "to_dict"):
            dates = cal.to_dict().get("Earnings Date", [])
        else:
            return None, None
        if not dates:
            return None, None
        d      = dates[0] if isinstance(dates, (list, tuple)) else dates
        ed_str = str(d)[:10]
        ed     = date.fromisoformat(ed_str)
        days   = (ed - date.today()).days
        return ed_str, (days if days >= 0 else None)
    except Exception:
        return None, None


def get_stock_data(code: str) -> dict:
    try:
        ticker = yf.Ticker(f"{code}.T")
        df     = ticker.history(period="6mo")
        if df.empty:
            return {"code": code, "error": "データなし"}

        prices  = _clean_prices(df["Close"].tolist())
        volumes = df["Volume"].tolist()

        if not prices:
            return {"code": code, "error": "有効な価格データなし"}

        latest = _f(prices[-1])
        prev   = _f(prices[-2]) if len(prices) > 1 else latest

        if latest is None:
            return {"code": code, "error": "最新価格がNaN"}

        avg_vol   = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        vol_ratio = _f(volumes[-1] / avg_vol, 2) if avg_vol else 1.0

        window   = prices[-252:] if len(prices) >= 252 else prices
        high_52w = _f(max(window))
        low_52w  = _f(min(window))

        change_pct = _f((latest - prev) / prev * 100, 2) if prev else 0

        earnings_date, days_to_earnings = _get_next_earnings(ticker)

        return {
            "code":             code,
            "latest":           latest,
            "change_pct":       change_pct,
            "change_5d":        calc_change_pct(prices, 5),
            "change_20d":       calc_change_pct(prices, 20),
            **calc_mas(prices),
            "rsi14":            calc_rsi(prices),
            "vol_ratio":        vol_ratio,
            "high_52w":         high_52w,
            "low_52w":          low_52w,
            "earnings_date":    earnings_date,
            "days_to_earnings": days_to_earnings,
            "prices_20d":       [_f(p) for p in prices[-20:]],
        }
    except Exception as e:
        return {"code": code, "error": str(e)}
