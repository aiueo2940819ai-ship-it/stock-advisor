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


def update_highest_prices(portfolio: dict, stock_map: dict) -> None:
    """
    保有銘柄の highest_price（保有後最高値）を更新して portfolio.json に保存。
    同時に portfolio["holdings"] の各エントリに分析用フィールドを追記する:
      - highest_price   : 保有後の最高値（円）
      - from_high_pct   : 現在値が最高値から何%下落しているか
      - unrealized_pct  : 買値からの損益率
      - trailing_alert  : トレイリングストップ警告レベル ("safe"/"caution"/"exit")
    """
    with open(_PORTFOLIO_FILE, encoding="utf-8") as f:
        saved = json.load(f)

    file_changed = False

    for h in portfolio.get("holdings", []):
        code    = h["code"]
        current = stock_map.get(code, {}).get("latest")
        buy     = h.get("buy_price", 0)

        # ── 最高値の更新（ファイル保存用） ──
        saved_h = next((s for s in saved.get("holdings", []) if s["code"] == code), None)
        if saved_h is not None:
            prev_high = saved_h.get("highest_price")
            if current is not None:
                new_high = max(filter(None, [prev_high, current, buy]))
                if new_high != prev_high:
                    saved_h["highest_price"] = new_high
                    file_changed = True
                h["highest_price"] = saved_h.get("highest_price", buy)
            else:
                h["highest_price"] = prev_high or buy
        else:
            h["highest_price"] = buy

        # ── stop_price の初回設定（未設定の場合のみ）──
        # ATRストップ（買値 - ATR×2.5）と ハードストップ（買値 - 8%）の厳しい方
        if saved_h is not None and saved_h.get("stop_price") is None and current is not None:
            atr = stock_map.get(code, {}).get("atr14")
            if atr and buy:
                atr_stop  = round(buy - atr * 2.5, 0)
                hard_stop = round(buy * (1 + -0.08), 0)
                stop_price = max(atr_stop, hard_stop)  # 高い方 = 厳しい方
                saved_h["stop_price"] = stop_price
                file_changed = True
                print(f"  {code} 損切りライン設定: {stop_price:.0f}円 "
                      f"(ATR止め:{atr_stop:.0f} / ハード:{hard_stop:.0f})")
        h["stop_price"] = (saved_h or {}).get("stop_price")

        # ── トレイリングストップの自動引き上げ ──
        # 含み益の最高値水準に応じて stop_price を段階的に引き上げる（絶対に下げない）
        if saved_h is not None and saved_h.get("stop_price") is not None and buy:
            high_now     = h["highest_price"] or buy
            current_stop = saved_h["stop_price"]
            new_stop     = current_stop

            if   high_now >= buy * 1.20:
                new_stop = max(current_stop, round(buy * 1.12, 0))
            elif high_now >= buy * 1.15:
                new_stop = max(current_stop, round(buy * 1.07, 0))
            elif high_now >= buy * 1.10:
                new_stop = max(current_stop, round(buy * 1.03, 0))

            if new_stop != current_stop:
                saved_h["stop_price"] = new_stop
                h["stop_price"]       = new_stop
                file_changed = True
                print(f"  {code} トレイリングストップ引き上げ: "
                      f"{current_stop:.0f}円 → {new_stop:.0f}円 "
                      f"(最高値{high_now:.0f}円 / 買値{buy:.0f}円)")

        # ── 分析用フィールドをメモリ上の holdings に追記 ──
        high = h["highest_price"] or buy
        if current and buy:
            h["unrealized_pct"] = round((current - buy) / buy * 100, 1)
            h["from_high_pct"]  = round((current - high) / high * 100, 1)

            unr = h["unrealized_pct"]
            fh  = h["from_high_pct"]

            # トレイリングストップ判定
            # +15%超えてから+7%以下に戻った → exit
            # +10%超えてから+3%以下に戻った → exit
            # 上記未満だが最高値から-5%以上下落 → caution
            if (high >= buy * 1.15 and unr <= 7) or (high >= buy * 1.10 and unr <= 3):
                h["trailing_alert"] = "exit"
            elif fh <= -5:
                h["trailing_alert"] = "caution"
            else:
                h["trailing_alert"] = "safe"
        else:
            h["unrealized_pct"] = None
            h["from_high_pct"]  = None
            h["trailing_alert"] = "safe"

    if file_changed:
        with open(_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        print("  highest_price 更新 → portfolio.json 保存")
