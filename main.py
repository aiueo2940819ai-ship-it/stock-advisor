from datetime import datetime

from src.data.market    import get_macro_data
from src.data.stocks    import get_stock_data
from src.data.portfolio import load_portfolio
from src.data.history   import save_history, load_history
from src.analysis.analyzer import analyze_daily
from src.analysis.risk      import check_stop_loss
from src.notifications.gmail import send_report


def main():
    print(f"=== AI投資判断ボット {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # ① ポートフォリオ読み込み
    print("\n[1/6] ポートフォリオ読み込み中...")
    portfolio = load_portfolio()

    # ② マクロデータ
    print("\n[2/6] マクロ指標取得中...")
    macro_data = get_macro_data()

    # ③ 株価データ（ウォッチリスト + 保有銘柄の漏れを補完）
    watch_list    = portfolio.get("watch_list", [])
    holding_codes = {h["code"] for h in portfolio.get("holdings", [])}
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

    # ④ 損切りチェック
    stop_alerts = check_stop_loss(
        portfolio.get("holdings", []),
        stock_map,
        portfolio.get("stop_loss_ratio", -0.08),
    )

    # ⑤ 履歴読み込み → Claude分析
    print("\n[4/6] 過去履歴読み込み中...")
    history = load_history(days=7)
    print(f"  {len(history)}日分の履歴を取得")

    print("\n[5/6] Claude分析中...")
    result = analyze_daily(stock_data_list, portfolio, macro_data, history, stock_map)

    # ⑥ 履歴保存 → メール送信
    print("\n[6/6] 履歴保存 & メール送信中...")
    save_history(result)

    subject = (
        f"【AI投資判断】{datetime.now().strftime('%m/%d(%a)')} "
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
    print(f"売り推奨: {result['sell_codes']}")


if __name__ == "__main__":
    main()
