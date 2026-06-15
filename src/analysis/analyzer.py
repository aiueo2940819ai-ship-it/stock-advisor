import json
import re
from datetime import datetime
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_SMART as CLAUDE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── 履歴コンテキスト構築 ──────────────────────────────────────────────────────

def _history_context(history: list[dict], stock_map: dict) -> str:
    if not history:
        return "## 過去の分析履歴\n初回実行のため履歴なし。\n"

    lines = ["## 過去の分析履歴（直近7日間）\n"]
    for h in history:
        lines.append(f"### {h['date']} 市場状況: {h['market_condition']}")
        lines.append(f"サマリー: {h['summary']}")

        snapshot = h.get("stock_snapshot", {})

        if h["buy_codes"]:
            lines.append("買い推奨 → 現在のパフォーマンス:")
            for code in h["buy_codes"]:
                then = snapshot.get(code, {}).get("latest")
                now  = stock_map.get(code, {}).get("latest")
                if then and now:
                    pct = (now - then) / then * 100
                    mark = "▲" if pct > 0 else "▼"
                    lines.append(f"  {code}: 推奨時 {then:,.0f}円 → 現在 {now:,.0f}円 ({mark}{abs(pct):.1f}%)")
                else:
                    lines.append(f"  {code}: 価格追跡不可")

        if h["sell_codes"]:
            lines.append(f"売り推奨: {', '.join(h['sell_codes'])}")
        if h["hold_codes"]:
            lines.append(f"継続保有: {', '.join(h['hold_codes'])}")
        lines.append("")

    return "\n".join(lines)


# ── Claude分析 ────────────────────────────────────────────────────────────────

def _build_prompt(
    stock_data_list: list[dict],
    portfolio: dict,
    macro_data: list[dict],
    history: list[dict],
    stock_map: dict,
) -> str:
    cash      = portfolio["cash_jpy"]
    total     = portfolio["total_asset_jpy"]
    max_ratio = portfolio.get("max_single_stock_ratio", 0.15)
    stop_loss = portfolio.get("stop_loss_ratio", -0.08)
    today     = datetime.now().strftime("%Y年%m月%d日")
    is_monday = datetime.now().weekday() == 0

    history_ctx = _history_context(history, stock_map)

    weekly_instruction = (
        "\n⚠️ 今日は月曜日です。「先週のパフォーマンスレビュー」セクションを必ず追加し、"
        "先週推奨した銘柄の勝敗と学びを具体的に総括してください。\n"
    ) if is_monday else ""

    return f"""あなたは10年以上の経験を持つ日本株スウィングトレードの専門家です。
今日は{today}です。{weekly_instruction}

## 投資方針
- 保有期間: 1週間〜1ヶ月のスウィングトレード（デイトレードではない）
- 目標: トレンドに乗り、ノイズを無視して利益を最大化
- リスク管理最優先で、利益より損失回避を重視

## ポートフォリオ
- 現金: {cash:,}円
- 総資産: {total:,}円
- 1銘柄上限: 総資産の{int(max_ratio*100)}% = {int(total*max_ratio):,}円
- 損切りライン: 買値から{int(stop_loss*100)}%

## 本日のマクロ指標
{json.dumps(macro_data, ensure_ascii=False, indent=2)}

## 保有銘柄
{json.dumps(portfolio.get("holdings", []), ensure_ascii=False, indent=2)}

## ウォッチリスト（本日の株価・テクニカル）
テクニカル指標の見方:
- rsi14: 30以下=過売り、40-65=スウィング買い圏、70以上=過熱
- ma25 > ma75 → 中期上昇トレンド確定
- vol_ratio: 1.5以上=方向感のある動き、0.5以下=膠着
- change_5d: 直近5日の変化率（-5%以内の押し目は好機）

{json.dumps(stock_data_list, ensure_ascii=False, indent=2)}

{history_ctx}

## 絶対ルール（必ず守ること）

### マクロ環境チェック（最優先）
- VIX > 25: 新規買い禁止
- VIX 20-25: 厳選1銘柄のみ許可
- NYダウ/S&P500 前日-2%以下: リスクオフ、新規見送り
- ドル円 前日比±1.5%以上: 為替敏感銘柄は様子見

### 保有銘柄（デフォルト=継続保有）
以下のいずれかでのみ売り推奨:
- 損切りライン到達（-{abs(int(stop_loss*100))}%）
- 利確水準到達（+20%以上）
- デッドクロス発生（MA25 < MA75に転落）
- RSI 75以上 + 出来高急増（天井示唆）
- 買いの根拠が完全に崩れた場合

### 新規買い条件（全て満たす銘柄のみ）
1. MA25 > MA75（中期上昇トレンド確認）
2. RSI が 35-68 の範囲
3. 直近5日で-7%超の急落がない（急落は別途押し目判断）
4. vol_ratio が 0.7-3.0 の範囲
5. 現金比率が30%以上
6. セクター全体が下落基調でない
7. **決算3営業日以内（days_to_earnings が 0〜3）の銘柄は新規買い禁止**
   → 決算またぎは予測不可能なリスク。翌日以降に改めて判断する

### 一貫性ルール（最重要）
- 過去の推奨を確認し、根拠なく方針を変更しない
- 前日「継続保有」の銘柄を翌日「売り」にする場合は明確な変化を必ず記載
- 同じ銘柄を連日推奨する場合は「前回推奨から継続」と明記

## 出力フォーマット（この形式で必ず出力）

### 本日の市場環境
- VIX: X（リスクオン/警戒/リスクオフ）
- 米国市場: 前日比X%（概況1文）
- ドル円: X円（トレンド1文）
- 本日の基本姿勢: 【買い増し/様子見/利確局面/損切り実施】

### 保有銘柄の判断
（保有銘柄がある場合のみ記載）
- **コード 銘柄名**: ✅継続 / 🔶利確 / 🔴損切り
  現在値: X円 | 買値: X円 | 損益: ±X%
  RSI: X | MA25: X円 vs MA75: X円
  判断: （理由30字以内）

### 新規買い候補TOP3
（リスクオン環境かつ現金30%以上の場合のみ。条件未達なら「本日は新規買い推奨なし（理由）」と明記）

1位 **コード 銘柄名** [コア/サテライト]
   現在値: X円 | RSI: X | MA25: Xvs MA75: X
   次回決算: X月X日（あとX日）※ 3日以内なら「決算近接につき対象外」と明記
   エントリー目安: X円前後
   目標値: X円（+X%、想定X週間後）
   損切り: X円（-X%）
   推奨数量: X株（必要額: 約X万円）
   推奨理由: （スウィング根拠50字以内）
   前回との整合性: 初回推奨/前回継続/方針変更（変更理由）

2位 （同形式 or 「該当なし」）
3位 （同形式 or 「該当なし」）

### ポートフォリオ構成
- 現在: コアX% / サテライトX% / 現金X%
- 目標: コア60% / サテライト40%
- 調整方針: （1文）

### 今週の注目テーマ
（マクロ背景と絡めて2点）

### リスク管理
（現ポートフォリオで注意すべき点1-2点）

---
※ 以下のJSONブロックは必ず出力の最後に含めてください。フォーマットを厳守してください。
[STRUCTURED_DATA]
{{
  "market_condition": "risk-on または risk-neutral または risk-off",
  "buy_codes":  ["例: 7203"],
  "sell_codes": [],
  "hold_codes": [],
  "summary": "本日の一言サマリー（40字以内）"
}}
[/STRUCTURED_DATA]
"""


def _parse_structured(text: str) -> dict:
    m = re.search(r'\[STRUCTURED_DATA\](.*?)\[/STRUCTURED_DATA\]', text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1).strip())
    except Exception:
        return {}


def _stock_snapshot(stock_data_list: list[dict]) -> dict:
    return {
        s["code"]: {"latest": s.get("latest"), "ma25": s.get("ma25"), "rsi14": s.get("rsi14")}
        for s in stock_data_list
        if "error" not in s
    }


def analyze_daily(
    stock_data_list: list[dict],
    portfolio: dict,
    macro_data: list[dict],
    history: list[dict],
    stock_map: dict,
) -> dict:
    prompt = _build_prompt(stock_data_list, portfolio, macro_data, history, stock_map)

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    text       = message.content[0].text
    structured = _parse_structured(text)

    clean_text = re.sub(
        r'\[STRUCTURED_DATA\].*?\[/STRUCTURED_DATA\]', '', text, flags=re.DOTALL
    ).strip()

    return {
        "analysis_text":    clean_text,
        "market_condition": structured.get("market_condition", "risk-neutral"),
        "buy_codes":        structured.get("buy_codes",  []),
        "sell_codes":       structured.get("sell_codes", []),
        "hold_codes":       structured.get("hold_codes", []),
        "summary":          structured.get("summary",    ""),
        "stock_snapshot":   _stock_snapshot(stock_data_list),
    }
