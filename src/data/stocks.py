import yfinance as yf
from datetime import date
from .indicators import calc_rsi, calc_mas, calc_change_pct


def _get_next_earnings(ticker) -> tuple[str | None, int | None]:
    """次回決算日と残り営業日数を返す。取得できなければ (None, None)"""
    try:
        cal = ticker.calendar
        if not cal:
            return None, None

        # yfinance は dict または DataFrame を返すバージョンがある
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif hasattr(cal, "to_dict"):
            dates = cal.to_dict().get("Earnings Date", [])
        else:
            return None, None

        if not dates:
            return None, None

        d = dates[0] if isinstance(dates, (list, tuple)) else dates
        ed_str = str(d)[:10]
        ed     = date.fromisoformat(ed_str)
        days   = (ed - date.today()).days
        return ed_str, days if days >= 0 else None
    except Exception:
        return None, None


def get_stock_data(code: str) -> dict:
    try:
        ticker = yf.Ticker(f"{code}.T")
        df     = ticker.history(period="6mo")
        if df.empty:
            return {"code": code, "error": "データなし"}

        prices  = df["Close"].tolist()
        volumes = df["Volume"].tolist()
        latest  = round(prices[-1], 1)
        prev    = round(prices[-2], 1) if len(prices) > 1 else latest

        avg_vol   = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        vol_ratio = round(volumes[-1] / avg_vol, 2) if avg_vol else 1.0

        window   = prices[-252:] if len(prices) >= 252 else prices
        high_52w = round(max(window), 1)
        low_52w  = round(min(window), 1)

        earnings_date, days_to_earnings = _get_next_earnings(ticker)

        return {
            "code":             code,
            "latest":           latest,
            "change_pct":       round((latest - prev) / prev * 100, 2) if prev else 0,
            "change_5d":        calc_change_pct(prices, 5),
            "change_20d":       calc_change_pct(prices, 20),
            **calc_mas(prices),
            "rsi14":            calc_rsi(prices),
            "vol_ratio":        vol_ratio,
            "high_52w":         high_52w,
            "low_52w":          low_52w,
            "earnings_date":    earnings_date,
            "days_to_earnings": days_to_earnings,
            "prices_20d":       [round(p, 1) for p in prices[-20:]],
        }
    except Exception as e:
        return {"code": code, "error": str(e)}
