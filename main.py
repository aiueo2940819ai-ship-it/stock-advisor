from datetime import datetime

from src.data.market    import get_macro_data, get_us_sector_signal, get_jp_sector_trend
from src.data.stocks    import get_stock_data
from src.data.portfolio import load_portfolio, update_highest_prices
from src.data.history   import save_history, load_history
from src.data.screener  import screen_gekioshi_candidates
from src.analysis.analyzer          import analyze_daily
from src.analysis.risk              import check_stop_loss
from src.analysis.monthly_rotation  import is_first_business_day, load_universe, analyze_rotation
from src.notifications.gmail        import send_report


def main():
    print(f"=== AI投資判断ボット {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # ① ポートフォリオ読み込み
    print("\n[1/6] ポートフォリオ読み込み中...")
    portfolio = load_portfolio()

    # ② マクロデータ + 米国セクターシグナル
    print("\n[2/6] マクロ指標取得中...")
    macro_data = get_macro_data()
    print("  米国セクターシグナル取得中...")
    us_sector_signal = get_us_sector_signal()
    print("  日本セクタートレンド取得中...")
    jp_sector_trend = get_jp_sector_trend()

    # ③ 株価データ（ウォッチリスト + 保有銘柄の漏れを補完）
    watch_list    = portfolio.get("watch_list", [])
    watch_codes   = {s["code"] for s in watch_list}
    extra_holdings = [
        {"code": h["code"], "name": h.get("name", h["code"])}
        for h in portfolio.get("holdings", [])
        if h["code"] not in watch_codes
    ]
    all_targets = watch_list + extra_holdings

    print(f"\n[3/6] 株価データ取得中...({len(all_targets)}銘柄)")
    stock_data_list = []
    stock_map       = {}
    for s in all_targets:
        code = s["code"]
        data = get_stock_data(code)
        data["name"] = s["name"]
        stock_data_list.append(data)
        stock_map[code] = data
        if "error" in data:
            print(f"  ⚠  {code} {s['name']}: {data['error']}")
        else:
            rsi_str = f" RSI:{data['rsi14']}" if data.get("rsi14") else ""
            print(f"  ✓ {code} {s['name']}: {data['latest']:,.0f}円{rsi_str}")

    # ③-b 最高値更新（highest_price を portfolio.json に記録）
    print("  最高値チェック中...")
    update_highest_prices(portfolio, stock_map)

    # ③-b2 総資産を時価ベースで再計算（買値ベースのズレを修正）
    holdings_now = sum(
        stock_map.get(h["code"], {}).get("latest", h.get("buy_price", 0)) * h.get("shares", 0)
        for h in portfolio.get("holdings", [])
    )
    portfolio["total_asset_jpy"] = portfolio["cash_jpy"] + holdings_now
    print(f"  時価総資産: {portfolio['total_asset_jpy']:,}円")

    # ③-c 劇おすすめ広域スクリーニング（ウォッチリスト外の候補を抽出）
    print("\n  劇おすすめ広域スクリーニング中（日経225規模）...")
    gekioshi_candidates = screen_gekioshi_candidates(
        n_candidates=5,
        existing_codes=set(stock_map.keys()),
    )

    # ④ 損切りチェック
    stop_alerts = check_stop_loss(
        portfolio.get("holdings", []),
        stock_map,
        portfolio.get("stop_loss_ratio", -0.08),
    )

    # ⑤ 履歴読み込み → 日次Claude分析
    print("\n[4/6] 過去履歴読み込み中...")
    is_monday = datetime.now().weekday() == 0
    history_days = 14 if is_monday else 7   # 月曜は2週間・平日は1週間
    history = load_history(days=history_days)
    print(f"  {len(history)}日分の履歴を取得")

    print("\n[5/6] Claude分析中...")
    result = analyze_daily(stock_data_list, portfolio, macro_data, history, stock_map, us_sector_signal, jp_sector_trend, gekioshi_candidates)

    # ⑥ 履歴保存 → 日次メール送信
    print("\n[6/6] 履歴保存 & メール送信中...")
    save_history(result)

    gekioshi = result.get("gekioshi_code")
    subject = (
        f"{'🔥【劇おすすめ ' + gekioshi + '】' if gekioshi else '【AI投資判断】'}"
        f"{datetime.now().strftime('%m/%d(%a)')} "
        f"{result.get('summary', '本日の分析')}"
    )
    body_parts = []
    if stop_alerts:
        body_parts += ["=" * 50, "【緊急】損切りアラート", "=" * 50] + stop_alerts + [""]
    body_parts.append(result["analysis_text"])
    send_report(subject, "\n".join(body_parts))

    print(f"\n=== 完了 ===")
    print(f"市場状況: {result['market_condition']}")
    print(f"買い推奨: {result['buy_codes']}")

    # ⑦ 月次ローテーション（毎月第1営業日のみ）
    if is_first_business_day():
        history_30d = load_history(days=30)
        print(f"\n[月次] 過去30日履歴: {len(history_30d)}日分")
        print("\n[月次] ウォッチリスト入れ替え分析中...")
        universe      = load_universe()
        universe_codes = {s["code"] for s in universe}

        # ユニバース全銘柄の株価を取得（ウォッチリスト外の分だけ追加取得）
        universe_data = []
        for s in universe:
            code = s["code"]
            if code in stock_map:
                d = dict(stock_map[code])
            else:
                d = get_stock_data(code)
                print(f"  追加取得: {code} {s['name']}")
            d["name"]   = s["name"]
            d["sector"] = s.get("sector", "")
            universe_data.append(d)

        rotation_text = analyze_rotation(
            universe_data, watch_list, macro_data, portfolio, history_30d
        )
        rotation_subject = (
            f"【月次ローテーション】{datetime.now().strftime('%Y年%m月')} "
            "ウォッチリスト入れ替え提案"
        )
        send_report(rotation_subject, rotation_text)
        print("[月次] ローテーションメール送信完了")


if __name__ == "__main__":
    main()
