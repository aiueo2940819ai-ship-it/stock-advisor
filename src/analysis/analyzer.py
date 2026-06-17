import json
import re
from datetime import datetime
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_SMART as CLAUDE_MODEL
from src.utils.usage_tracker import log_usage

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
    us_sector_signal: str = "",
    jp_sector_trend: str = "",
    gekioshi_candidates: list[dict] = None,
) -> str:
    cash      = portfolio["cash_jpy"]
    total     = portfolio["total_asset_jpy"]
    max_ratio = portfolio.get("max_single_stock_ratio", 0.15)
    stop_loss = portfolio.get("stop_loss_ratio", -0.08)
    today     = datetime.now().strftime("%Y年%m月%d日")
    is_monday = datetime.now().weekday() == 0

    history_ctx = _history_context(history, stock_map)

    cands     = gekioshi_candidates or []
    n_cand    = len(cands)
    cand_json = json.dumps(cands, ensure_ascii=False, indent=2) if cands else "（本日は条件通過銘柄なし）"

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

保有銘柄フィールドの説明:
- highest_price  : 保有開始後の最高値（円）
- unrealized_pct : 買値からの現在損益率（%）
- from_high_pct  : 最高値から現在値への下落率（%、マイナスが大きいほど戻り大）
- trailing_alert : "exit"=トレイリングストップ到達→売り推奨 / "caution"=要注意 / "safe"=問題なし
  ※ trailing_alert="exit" の銘柄は、含み益が一度大きく乗った後に失われており、
     損切りラインに達していなくても売りを推奨する

{us_sector_signal}

{jp_sector_trend}

## 劇おすすめ広域候補（ウォッチリスト外・Python事前スクリーニング通過）
※ 日経225規模の銘柄からMA/RSI/ATR/R/Rの定量条件を通過した上位{n_cand}件。
  劇おすすめ選定時はウォッチリスト銘柄と同等に検討すること。
  stop_est=ATRベース損切り目安、rr_ratio=リスクリワード比

{cand_json}

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

### 保有銘柄の売り判断（デフォルト=継続保有）

**【絶対売り】以下のいずれかで必ず売る:**
- 損切りライン到達（各銘柄の stop_price を下回った場合）
  ※ stop_price = ATRストップ（買値-ATR×2.5）とハードストップ（買値-8%）の厳しい方
  ※ stop_price が未設定の場合は買値から-{abs(int(stop_loss*100))}%をハードストップとして使う
- 利確水準到達（+20%以上）
- デッドクロス発生（MA25 < MA75に転落）
- RSI 75以上 + 出来高急増（天井示唆）

**【トレイリングストップ】利益を守るための引き上げライン:**
- 含み益が+10%を超えたことがある銘柄が、その後+3%以下まで戻った → 売り
  （例: +15%まで上がり→+3%まで戻った時点で撤退、利益を守る）
- 含み益が+15%を超えたことがある銘柄が、その後+7%以下まで戻った → 売り
- ※ 「以前の高値から◯%戻した」はchange_5dと直近推移から推定すること

**【タイムストップ】塩漬け防止:**
- 保有期間が3週間（15営業日）を超え、かつ含み益が±5%以内 → 売り検討
  （理由: 動かない銘柄は機会損失。別の銘柄に資金を回す）

**【テーマ崩壊】買い根拠が消えたら保有継続しない:**
- 購入理由（セクター上昇期待・トレンド転換等）が否定される材料が出た → 損益にかかわらず売り

### 新規買い条件（全て満たす銘柄のみ）
1. MA25 > MA75（中期上昇トレンド確認）
2. RSI が 35-68 の範囲
3. 直近5日で-7%超の急落がない（急落は別途押し目判断）
4. vol_ratio が 0.7-3.0 の範囲
5. 現金比率が30%以上
6. セクター全体が下落基調でない
7. **決算3営業日以内（days_to_earnings が 0〜3）の銘柄は新規買い禁止**
   → 決算またぎは予測不可能なリスク。翌日以降に改めて判断する

### ポジションサイジング（推奨数量の必須計算方法）
固定%ではなくATRベースのリスク額固定で算出すること。

  リスク額 = 総資産 × 1%
  損切り幅 = エントリー価格 - stop_price（stop_price未設定時はエントリー価格 × 8%）
  ATR株数  = floor(リスク額 ÷ 損切り幅)
  上限株数 = floor(総資産 × {int(max_ratio*100)}% ÷ エントリー価格)
  推奨株数 = min(ATR株数, 上限株数)

例: 総資産100万円、エントリー2,000円、stop_price 1,850円（損切り幅150円）
  → 10,000円 ÷ 150円 = 66株（約132,000円）

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

### 🔥 劇おすすめ（最高確信度の1銘柄）
以下の**全条件**を満たす銘柄が存在する場合のみ出力。なければ「本日は劇おすすめ該当なし」と1行で終わる。

【劇おすすめ選定基準 — 全て満たすこと】
1. 通常の新規買い7条件を全てクリア
2. 日本セクタートレンドで該当セクターが「◎ 強い上昇」または「○ 上昇」
3. 米国セクターシグナルで対応する米国セクターがプラス（追い風あり）
4. リスクリワード比が2倍以上（目標値までの上昇幅 ÷ 損切りまでの下落幅 ≥ 2）
5. RSIが40〜60の範囲（過熱でも売られすぎでもない、最も伸びやすい位置）

**コード 銘柄名** 🔥劇おすすめ
   現在値: X円 | RSI: X | セクター: XX（トレンド: ◎/○）
   米国シグナル: XXXETF X%（追い風/逆風）
   エントリー: X円 | 目標: X円（+X%） | 損切り（stop_price）: X円（-X%）
   リスクリワード: X倍
   推奨株数: X株（約X万円）
   一言: （なぜ今これなのか30字以内）

### 新規買い候補TOP2
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
  "summary": "本日の一言サマリー（40字以内）",
  "gekioshi_code": "劇おすすめが存在する場合はその銘柄コード（例: 7203）、存在しない場合は null"
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
    us_sector_signal: str = "",
    jp_sector_trend: str = "",
    gekioshi_candidates: list[dict] = None,
) -> dict:
    prompt = _build_prompt(stock_data_list, portfolio, macro_data, history, stock_map, us_sector_signal, jp_sector_trend, gekioshi_candidates)

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    log_usage("morning", CLAUDE_MODEL, message.usage.input_tokens, message.usage.output_tokens)
    text = next(b.text for b in message.content if b.type == "text")
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
        "gekioshi_code":    structured.get("gekioshi_code"),
        "stock_snapshot":   _stock_snapshot(stock_data_list),
    }
