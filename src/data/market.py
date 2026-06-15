import yfinance as yf
from config import MACRO_INDICATORS


def get_macro_data() -> list[dict]:
    result = []
    for ticker, name in MACRO_INDICATORS.items():
        try:
            df = yf.Ticker(ticker).history(period="10d")
            if df.empty:
                continue
            prices = df["Close"].tolist()
            latest = round(prices[-1], 2)
            prev   = round(prices[-2], 2) if len(prices) > 1 else latest
            change_pct      = round((latest - prev) / prev * 100, 2) if prev else 0
            week_change_pct = round((latest - prices[0]) / prices[0] * 100, 2) if prices[0] else 0

            result.append({
                "name":            name,
                "ticker":          ticker,
                "latest":          latest,
                "change_pct":      change_pct,
                "week_change_pct": week_change_pct,
            })
            print(f"  {name}: {latest} ({change_pct:+.2f}%)")
        except Exception as e:
            print(f"  {name} エラー: {e}")
    return result
