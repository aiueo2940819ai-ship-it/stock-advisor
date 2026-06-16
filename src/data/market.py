import yfinance as yf
from config import MACRO_INDICATORS

# 米国セクターETF → 日本業種・代表銘柄の対応
_US_SECTOR_MAP = {
    "XLF":  ("金融",       ["8306三菱UFJ", "8316三井住友", "8411みずほ"]),
    "XLK":  ("IT・電機",   ["6758ソニー", "6861キーエンス", "9984ソフトバンクG"]),
    "XLI":  ("産業機械",   ["6367ダイキン", "7011三菱重工", "6301コマツ"]),
    "XLE":  ("エネルギー", ["1605INPEX", "5020ENEOS"]),
    "XLY":  ("自動車",     ["7203トヨタ", "7267ホンダ", "7269スズキ"]),
    "XLB":  ("素材・化学", ["4063信越化学", "3407旭化成"]),
    "XLV":  ("医薬品",     ["4502武田薬品", "4568第一三共"]),
    "XLP":  ("食品・小売", ["2802味の素", "3382セブン&アイ"]),
    "XLC":  ("通信",       ["9432NTT", "9433KDDI"]),
    "XLU":  ("電力・ガス", ["9501東京電力HD", "9503関西電力"]),
    "XLRE": ("不動産",     ["3003ヒューリック", "8951日本ビルファンド"]),
}


def get_us_sector_signal() -> str:
    """
    米国11セクターETFの前日騰落率を取得し、
    日本市場への波及シグナルとして整形して返す。

    論文「日米業種リードラグ」の簡易版:
      米国終値（5:00 JST確定） → 日本翌日寄り付きへの情報伝播
    """
    tickers = list(_US_SECTOR_MAP.keys())
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True,
                          progress=False, threads=True)["Close"]
        raw = raw.ffill().dropna(how="all")
        if len(raw) < 2:
            return ""
        changes = ((raw.iloc[-1] - raw.iloc[-2]) / raw.iloc[-2] * 100).dropna()
    except Exception as e:
        print(f"  US sector signal エラー: {e}")
        return ""

    # 騰落率でソート
    ranked = changes.sort_values(ascending=False)

    lines = ["## 米国昨日セクター動向（日本市場への波及参考）",
             "※ 米国終値が日本翌日寄り付きにリードラグ効果をもたらす傾向あり\n"]

    lines.append("**強いセクター（日本関連株に追い風）:**")
    for ticker in ranked.index[:4]:
        if ranked[ticker] > 0 and ticker in _US_SECTOR_MAP:
            jp_sector, jp_stocks = _US_SECTOR_MAP[ticker]
            lines.append(
                f"  {ticker}({jp_sector}): {ranked[ticker]:+.1f}%"
                f" → {', '.join(jp_stocks[:2])}"
            )

    lines.append("\n**弱いセクター（関連株は慎重）:**")
    for ticker in ranked.index[-3:]:
        if ranked[ticker] < 0 and ticker in _US_SECTOR_MAP:
            jp_sector, jp_stocks = _US_SECTOR_MAP[ticker]
            lines.append(
                f"  {ticker}({jp_sector}): {ranked[ticker]:+.1f}%"
                f" → {', '.join(jp_stocks[:2])}"
            )

    lines.append("\n※ このシグナルは当日のエントリータイミング判断補助。スウィング目線では複数日の傾向継続を確認してから判断。")
    return "\n".join(lines)


_JP_SECTOR_ETFS = {
    '1617.T': '食品',       '1618.T': 'エネルギー', '1619.T': '建設・資材',
    '1620.T': '素材・化学', '1621.T': '医薬品',     '1622.T': '自動車',
    '1623.T': '鉄鋼・非鉄', '1624.T': '機械',      '1625.T': '電機・精密',
    '1626.T': '情報通信',   '1627.T': '電力・ガス', '1628.T': '運輸・物流',
    '1629.T': '商社',       '1630.T': '小売',       '1631.T': '銀行',
    '1632.T': '金融',       '1633.T': '不動産',
}

# 個別銘柄コード → 対応セクターETF
STOCK_TO_SECTOR_ETF = {
    '8306': '1631.T', '8316': '1631.T', '8411': '1631.T',  # 銀行
    '6758': '1625.T', '6861': '1625.T', '6501': '1625.T',  # 電機・精密
    '7203': '1622.T', '7267': '1622.T', '7269': '1622.T',  # 自動車
    '9432': '1626.T', '9433': '1626.T', '9984': '1626.T',  # 情報通信
    '8058': '1629.T', '8001': '1629.T', '8031': '1629.T',  # 商社
    '4502': '1621.T', '4568': '1621.T', '4519': '1621.T',  # 医薬品
    '1605': '1618.T', '5020': '1618.T',                     # エネルギー
    '4063': '1620.T', '3407': '1620.T',                     # 素材・化学
    '6367': '1624.T', '7011': '1624.T', '6301': '1624.T',  # 機械
}


def get_jp_sector_trend() -> str:
    """
    TOPIX-17 セクターETF の1ヶ月・3ヶ月トレンドを取得。
    個別株のセクター環境確認に使う。
    """
    tickers = list(_JP_SECTOR_ETFS.keys())
    try:
        raw = yf.download(tickers, period="4mo", auto_adjust=True,
                          progress=False, threads=True)["Close"]
        raw = raw.ffill().dropna(how="all")
        if len(raw) < 5:
            return ""

        def pct(days):
            if len(raw) <= days:
                return None
            base = raw.iloc[-days - 1]
            last = raw.iloc[-1]
            return ((last - base) / base * 100).round(1)

        r1m = pct(21)   # 1ヶ月
        r3m = pct(63)   # 3ヶ月

    except Exception as e:
        print(f"  JP sector trend エラー: {e}")
        return ""

    lines = ["## 日本セクタートレンド（TOPIX-17 ETF）",
             "※ 個別株のセクター環境確認用。上昇トレンドのセクターは買い環境◎\n"]

    rows = []
    for ticker, name in _JP_SECTOR_ETFS.items():
        m1 = float(r1m.get(ticker, 0) or 0)
        m3 = float(r3m.get(ticker, 0) or 0) if r3m is not None else 0
        rows.append((ticker, name, m1, m3))

    rows.sort(key=lambda x: x[2], reverse=True)

    lines.append("セクター          1ヶ月    3ヶ月   判定")
    lines.append("-" * 44)
    for ticker, name, m1, m3 in rows:
        if m1 > 3 and m3 > 5:
            mark = "◎ 強い上昇"
        elif m1 > 0:
            mark = "○ 上昇"
        elif m1 > -3:
            mark = "△ 横ばい"
        else:
            mark = "✗ 下落"
        lines.append(f"{name:<10} {m1:>+6.1f}%  {m3:>+6.1f}%  {mark}")

    return "\n".join(lines)


def get_macro_data() -> list[dict]:
    result = []
    for ticker, name in MACRO_INDICATORS.items():
        try:
            df = yf.Ticker(ticker).history(period="10d")
            if df.empty:
                continue
            prices = df["Close"].tolist()
            latest = round(prices[-1], 2)
            prev   = round(prices[-2], 2) if len(prices) > 1 else latest
            change_pct      = round((latest - prev) / prev * 100, 2) if prev else 0
            week_change_pct = round((latest - prices[0]) / prices[0] * 100, 2) if prices[0] else 0

            result.append({
                "name":            name,
                "ticker":          ticker,
                "latest":          latest,
                "change_pct":      change_pct,
                "week_change_pct": week_change_pct,
            })
            print(f"  {name}: {latest} ({change_pct:+.2f}%)")
        except Exception as e:
            print(f"  {name} エラー: {e}")
    return result
