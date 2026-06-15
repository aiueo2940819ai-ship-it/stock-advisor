def calc_rsi(prices: list, period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent  = changes[-period:]
    avg_gain = sum(max(c, 0)       for c in recent) / period
    avg_loss = sum(abs(min(c, 0))  for c in recent) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def calc_mas(prices: list) -> dict:
    result = {}
    for p in [5, 25, 75]:
        result[f"ma{p}"] = (
            round(sum(prices[-p:]) / p, 1) if len(prices) >= p else None
        )
    return result


def calc_change_pct(prices: list, days: int) -> float | None:
    if len(prices) < days + 1:
        return None
    base = prices[-days - 1]
    if base == 0:
        return None
    return round((prices[-1] - base) / base * 100, 2)
