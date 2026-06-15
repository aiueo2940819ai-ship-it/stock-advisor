def check_stop_loss(holdings: list, stock_map: dict, stop_loss_ratio: float) -> list[str]:
    alerts = []
    for h in holdings:
        code      = h.get("code")
        buy_price = h.get("buy_price", 0)
        if not code or not buy_price or code not in stock_map:
            continue
        current    = stock_map[code].get("latest", buy_price)
        loss_ratio = (current - buy_price) / buy_price
        if loss_ratio <= stop_loss_ratio:
            alerts.append(
                f"損切りアラート: {h.get('name','?')}({code}) "
                f"買値{buy_price:,.0f}円 → 現在{current:,.0f}円 "
                f"({loss_ratio*100:.1f}%)"
            )
    return alerts
