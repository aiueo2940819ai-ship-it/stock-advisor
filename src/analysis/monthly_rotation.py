import json
import re
from datetime import date, datetime
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST as CLAUDE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def is_first_business_day() -> bool:
    """今月の第1営業日かどうかを判定する"""
    today = date.today()
    if today.day > 7:
        return False
    for d in range(1, today.day + 1):
        candidate = date(today.year, today.month, d)
        if candidate.weekday() < 5:   # 月〜金
            return candidate == today
    return False


def load_universe() -> list[dict]:
    with open("data/universe.json", encoding="utf-8") as f:
        return json.load(f)


def analyze_rotation(
    universe_data: list[dict],
    current_watchlist: list[dict],
    macro_data: list[dict],
    portfolio: dict,
    history: list[dict] | None = None,
) -> str:
    current_codes  = {s["code"] for s in current_watchlist}
    universe_str   = json.dumps(universe_data,    ensure_ascii=False, indent=2)
    macro_str      = json.dumps(macro_data,        ensure_ascii=False, indent=2)
    current_str    = json.dumps(current_watchlist, ensure_ascii=False, indent=2)
    month_label    = datetime.now().strftime("%Y年%m月")

    # 過去30日の推奨履歴サマリー（月次専用）
    history_summary = ""
    if history:
        lines = ["\n## 過去30日間の推奨履歴サマリー"]
        for h in history[-20:]:   # 最大20日分に絞る
            buy   = ", ".join(h.get("buy_codes",  [])) or "なし"
            sell  = ", ".join(h.get("sell_codes", [])) or "なし"
            lines.append(f"- {h['date']}: 買い={buy} / 売り={sell} / {h.get('summary','')}")
        history_summary = "\n".join(lines)

    prompt = f"""あなたは10年以上の経験を持つ日本株スウィングトレードの専門家です。
今日は{month_label}の第1営業日です。今月のウォッチリスト入れ替えを提案してください。
{history_summary}

## 前提
- 投資スタイル: 1週間〜1ヶ月のスウィングトレード
- 総資産: {portfolio.get('total_asset_jpy', 0):,}円
- ウォッチリストは常時25銘柄を維持する

## 現在のウォッチリスト（25銘柄）
{current_str}

## 投資ユニバース（50銘柄・本日の株価テクニカル込み）
{universe_str}

## 本日のマクロ指標
{macro_str}

## 分析の観点
今月のセクターローテーションと相場テーマを踏まえ、以下の基準でリストを評価してください：

### 外す候補の基準
- MA25 < MA75 かつ RSI < 40（中期下落トレンドが続いている）
- change_20d が -10% 以下（先月1ヶ月で大きく下落）
- セクター全体が今月の相場テーマと合わない
- vol_ratio が長期的に低位（市場の関心が薄れている）

### 入れる候補の基準
- MA25 > MA75（中期上昇トレンド）
- RSI が 40-65 の範囲（モメンタム良好かつ過熱なし）
- 今月の相場テーマ・セクターローテーションと合致
- 現在のウォッチリストでカバーできていないセクターを補完

## 出力フォーマット（必ずこの形式で）

# {month_label} ウォッチリスト入れ替え提案

## 今月の相場テーマ・セクター注目点
（今月特に注目すべきセクターや相場テーマを3点）

## 外し推奨銘柄
（最大5銘柄。なければ「入れ替えなし」）

| 銘柄コード | 銘柄名 | 外す理由 |
|-----------|--------|---------|
| XXXX | 〇〇 | MA25<MA75でトレンド崩れ、等 |

## 追加推奨銘柄
（外した分だけ追加。なければ「追加なし」）

| 銘柄コード | 銘柄名 | セクター | 追加理由 |
|-----------|--------|---------|---------|
| XXXX | 〇〇 | 〇〇 | RSI50台・MA25>MA75・今月テーマ合致、等 |

## 来月のウォッチリスト25銘柄（推奨）
（コードと銘柄名のみ、番号付きリストで）

## 入れ替えの総括
（今月の方針を3文以内で）
"""

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
