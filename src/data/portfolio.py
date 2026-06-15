import json
from pathlib import Path

_PORTFOLIO_FILE = Path("data/portfolio.json")
_WATCHLIST_FILE = Path("data/watchlist.json")


def load_portfolio() -> dict:
    with open(_PORTFOLIO_FILE, encoding="utf-8") as f:
        portfolio = json.load(f)

    with open(_WATCHLIST_FILE, encoding="utf-8") as f:
        portfolio["watch_list"] = json.load(f)

    # total_asset_jpy を保有株の時価 + 現金で自動計算（buy_price × shares で近似）
    holdings_value = sum(
        h.get("buy_price", 0) * h.get("shares", 0)
        for h in portfolio.get("holdings", [])
    )
    portfolio["total_asset_jpy"] = portfolio["cash_jpy"] + holdings_value

    print(f"保有株: {len(portfolio.get('holdings', []))}銘柄 | "
          f"現金: {portfolio['cash_jpy']:,}円 | "
          f"推定総資産: {portfolio['total_asset_jpy']:,}円 | "
          f"WL: {len(portfolio['watch_list'])}銘柄")
    return portfolio
