import json
from datetime import datetime, timedelta
from pathlib import Path

_USAGE_FILE = Path("data/usage.json")

# 料金単価（$/token）
_PRICES = {
    "claude-sonnet-4-6":         {"input": 3.0  / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.0  / 1_000_000},
}

_TASK_LABELS = {
    "morning":  "朝の分析",
    "evening":  "夕方サマリー",
    "rotation": "月次ローテーション",
}


def log_usage(task: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """API使用状況を data/usage.json に追記する"""
    try:
        price  = _PRICES.get(model, {"input": 0, "output": 0})
        cost   = input_tokens * price["input"] + output_tokens * price["output"]
        record = {
            "date":          datetime.now().strftime("%Y-%m-%d"),
            "time":          datetime.now().strftime("%H:%M"),
            "task":          task,
            "task_label":    _TASK_LABELS.get(task, task),
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost, 6),
        }

        logs = _load_all()
        logs.append(record)

        # 30日より古いものを削除
        cutoff = datetime.now() - timedelta(days=30)
        logs   = [
            r for r in logs
            if datetime.strptime(r["date"], "%Y-%m-%d") >= cutoff
        ]

        _USAGE_FILE.parent.mkdir(exist_ok=True)
        with open(_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"使用状況ログ保存失敗: {e}")


def _load_all() -> list[dict]:
    if not _USAGE_FILE.exists():
        return []
    with open(_USAGE_FILE, encoding="utf-8") as f:
        return json.load(f)
