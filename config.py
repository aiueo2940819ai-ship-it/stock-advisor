import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GMAIL_ADDRESS     = os.environ["GMAIL_ADDRESS"]
GMAIL_PASSWORD    = os.environ["GMAIL_APP_PASSWORD"]

CLAUDE_MODEL_SMART = "claude-sonnet-4-6"          # 朝の売買判断（精度優先）
CLAUDE_MODEL_FAST  = "claude-haiku-4-5-20251001"  # 夕方サマリー・月次（コスト優先）

MACRO_INDICATORS = {
    "^N225":    "日経平均",
    "USDJPY=X": "ドル円",
    "^DJI":     "NYダウ",
    "^IXIC":    "ナスダック",
    "^GSPC":    "S&P500",
    "^VIX":     "VIX恐怖指数",
    "^SOX":     "SOX半導体指数",
}

HISTORY_KEEP_DAYS = 30
