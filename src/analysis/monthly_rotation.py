import json
import re
from datetime import date, datetime
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST as CLAUDE_MODEL
from src.utils.usage_tracker import log_usage

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


def _performance_summary(portfolio: dict) -> str:
    """sell_historyから先月の勝率・期待値を集計して文字列で返す"""
    sells = portfolio.get("sell_history", [])
    if not sells:
        return ""

    last_month = (datetime.now().replace(day=1) - __import__('datetime').timedelta(days=1)).strftime("%Y-%m")
    month_sells = [s for s in sells if s.get("date", "").startswith(last_month)]
    all_sells   = [s for s in sells if s.get("realized_pnl") is not None]

    lines = ["\n## 先月の実績（sell_historyより自動集計）"]

    for label, data in [("先月", month_sells), ("累計", all_sells)]:
        if not data:
            lines.append(f"{label}: データなし")
            continue
        wins   = [s for s in data if s["realized_pnl"] > 0]
        losses = [s for s in data if s["realized_pnl"] < 0]
        rate   = round(len(wins) / len(data) * 100, 1)
        avg_w  = round(sum(s["realized_pnl"] for s in wins)   / len(wins),   0) if wins   else 0
        avg_l  = round(sum(s["realized_pnl"] for s in losses) / len(losses), 0) if losses else 0
        ev     = round(sum(s["realized_pnl"] for s in data)   / len(data),   0)
        total_w = sum(s["realized_pnl"] for s in wins)
        total_l = abs(sum(s["realized_pnl"] for s in losses))
        pf = round(total_w / total_l, 2) if total_l > 0 else float('inf')
        lines.append(
            f"{label}: {len(data)}回 勝率{rate}%（{len(wins)}勝{len(losses)}敗）"
            f" 平均利益+{avg_w:,.0f}円 平均損失{avg_l:,.0f}円 PF={pf} 期待値/回{ev:+,.0f}円"
        )
        if data:
            lines.append("  銘柄別: " + " / ".join(
                f"{s['code']}({s['realized_pnl']:+,.0f}円)" for s in sorted(data, key=lambda x: x["date"])
            ))

    return "\n".join(lines)


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
    perf_summary   = _performance_summary(portfolio)

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
{perf_summary}{history_summary}

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

## 先月の自己評価
先月の実績データをもとに、以下を簡潔に振り返ってください。
- 勝率・期待値は目標水準（勝率50%以上・PF1.5以上）に達しているか
- 負けトレードの共通パターン（セクター・エントリータイミング等）
- 今月のウォッチリスト選定・判断基準への反映点
"""

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    log_usage("rotation", CLAUDE_MODEL, message.usage.input_tokens, message.usage.output_tokens)
    return message.content[0].text
