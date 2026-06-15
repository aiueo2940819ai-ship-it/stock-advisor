import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GMAIL_ADDRESS     = os.environ["GMAIL_ADDRESS"]
GMAIL_PASSWORD    = os.environ["GMAIL_APP_PASSWORD"]

CLAUDE_MODEL = "claude-sonnet-4-6"

MACRO_INDICATORS = {
    "^N225":    "日経平均",
    "USDJPY=X": "ドル円",
    "^DJI":     "NYダウ",
    "^IXIC":    "ナスダック",
    "^GSPC":    "S&P500",
    "^VIX":     "VIX恐怖指数",
    "^SOX":     "SOX半導体指数",
}

HISTORY_KEEP_DAYS = 7
