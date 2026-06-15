from datetime import datetime

from src.data.market        import get_macro_data
from src.data.stocks        import get_stock_data
from src.data.portfolio     import load_portfolio
from src.analysis.evening   import analyze_evening
from src.notifications.gmail import send_report


def main():
    print(f"=== 夕方サマリー {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    portfolio  = load_portfolio()
    macro_data = get_macro_data()
    holdings   = portfolio.get("holdings", [])

    # 保有銘柄の現在値を取得
    print(f"\n保有銘柄取得中...({len(holdings)}銘柄)")
    holding_data = []
    for h in holdings:
        data = get_stock_data(h["code"])
        data["name"]      = h.get("name", h["code"])
        data["buy_price"] = h.get("buy_price", 0)
        data["shares"]    = h.get("shares", 0)
        holding_data.append(data)
        current = data.get("latest", "エラー")
        ed      = data.get("earnings_date", "不明")
        print(f"  {h['code']} {h.get('name','')}: {current}円 | 決算: {ed}")

    # ウォッチリストから上位10銘柄（RSIが良いもの）を取得
    watch_list = portfolio.get("watch_list", [])
    print(f"\nウォッチリスト取得中...(上位10銘柄)")
    watch_data = []
    for s in watch_list[:10]:
        data = get_stock_data(s["code"])
        data["name"]   = s["name"]
        data["sector"] = s.get("sector", "")
        watch_data.append(data)
        if "error" not in data:
            print(f"  {s['code']} {s['name']}: {data['latest']}円")

    print("\nClaude分析中...")
    analysis = analyze_evening(holding_data, watch_data, macro_data, portfolio)

    subject = f"【夕方サマリー】{datetime.now().strftime('%m/%d(%a)')} 本日の成績と明日の注目点"
    send_report(subject, analysis)
    print("=== 夕方サマリー送信完了 ===")


if __name__ == "__main__":
    main()
