import json
from datetime import datetime
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST as CLAUDE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def analyze_evening(
    holding_data: list[dict],
    watch_data: list[dict],
    macro_data: list[dict],
    portfolio: dict,
) -> str:
    # 保有銘柄の損益を計算してサマリーを作る
    holdings_summary = []
    for h in holding_data:
        if "error" in h:
            continue
        buy     = h.get("buy_price", 0)
        current = h.get("latest", buy)
        shares  = h.get("shares", 0)
        pnl_pct = round((current - buy) / buy * 100, 2) if buy else 0
        pnl_jpy = round((current - buy) * shares)
        ed      = h.get("earnings_date")
        days_ed = h.get("days_to_earnings")
        holdings_summary.append({
            "code":           h["code"],
            "name":           h["name"],
            "current":        current,
            "buy_price":      buy,
            "shares":         shares,
            "change_today":   h.get("change_pct", 0),
            "pnl_pct":        pnl_pct,
            "pnl_jpy":        pnl_jpy,
            "earnings_date":  ed,
            "days_to_earnings": days_ed,
        })

    today        = datetime.now().strftime("%Y年%m月%d日")
    holdings_str = json.dumps(holdings_summary, ensure_ascii=False, indent=2)
    macro_str    = json.dumps(macro_data,        ensure_ascii=False, indent=2)
    watch_str    = json.dumps(watch_data,         ensure_ascii=False, indent=2)
    cash         = portfolio.get("cash_jpy", 0)
    stop_loss    = portfolio.get("stop_loss_ratio", -0.08)

    prompt = f"""あなたは日本株スウィングトレードの専門家です。
今日（{today}）の東京市場大引け後のサマリーを作成してください。

## 本日の保有銘柄成績
{holdings_str}

## 損切りライン: 買値から{int(stop_loss*100)}%
## 現金残高: {cash:,}円

## 本日のマクロ指標
{macro_str}

## ウォッチリスト注目銘柄（本日）
{watch_str}

## ルール
- 決算まで3日以内の保有銘柄は「⚠️決算近接」と明記
- pnl_pct が損切りライン以下の銘柄は「🔴要確認」と明記
- pnl_pct が+20%以上の銘柄は「🟡利確検討」と明記

## 出力フォーマット（短く・スマホで読みやすく）

### 📊 本日の保有成績
（保有なしの場合は「現在保有なし・現金{cash:,}円」のみ）
銘柄名(コード)
  本日: X% ｜ 購入来: X% (±X円)
  ※ アラートがあれば追記

### 🌐 本日の相場（3行以内）
日米の主要指数と為替の動きを端的に

### 👀 明日の注目ポイント（2点のみ）
明日の取引で特に意識すべきことを具体的に

### 💡 明日の姿勢（1文）
「買い増し狙い」「様子見」「利確検討」のどれか＋理由
"""

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
