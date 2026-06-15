import json
from datetime import datetime, timedelta
from pathlib import Path

from config import HISTORY_KEEP_DAYS

_HISTORY_FILE = Path("data/history.json")


def save_history(result: dict) -> None:
    history = _load_all()
    today   = datetime.now().strftime("%Y-%m-%d")

    # 今日の既存エントリを除去して上書き
    history = [h for h in history if h["date"] != today]
    history.append({
        "date":             today,
        "market_condition": result.get("market_condition", ""),
        "buy_codes":        result.get("buy_codes",  []),
        "sell_codes":       result.get("sell_codes", []),
        "hold_codes":       result.get("hold_codes", []),
        "summary":          result.get("summary", ""),
        "analysis_text":    result.get("analysis_text", "")[:5000],
        "stock_snapshot":   result.get("stock_snapshot", {}),
    })

    # 7日より古いエントリを削除
    cutoff  = datetime.now() - timedelta(days=HISTORY_KEEP_DAYS)
    history = [
        h for h in history
        if datetime.strptime(h["date"], "%Y-%m-%d") >= cutoff
    ]
    history.sort(key=lambda x: x["date"])

    _HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"履歴保存完了（{len(history)}日分）")


def load_history(days: int = 7) -> list[dict]:
    history = _load_all()
    cutoff  = datetime.now() - timedelta(days=days)
    return [
        h for h in history
        if datetime.strptime(h["date"], "%Y-%m-%d") >= cutoff
    ]


def _load_all() -> list[dict]:
    if not _HISTORY_FILE.exists():
        return []
    with open(_HISTORY_FILE, encoding="utf-8") as f:
        return json.load(f)
