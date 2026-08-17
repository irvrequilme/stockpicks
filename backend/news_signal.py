"""
Live-only news sentiment, using yfinance's free Yahoo Finance news feed plus
a small hand-built finance keyword scorer.

Why not a general sentiment library: testing showed VADER (a well-known
general-purpose sentiment tool) scores an obviously bullish headline like
"beats earnings expectations, raises guidance" as perfectly neutral (0.0) —
general lexicons don't recognize finance-specific jargon. A small, explicit
keyword list is more transparent and actually catches the relevant terms,
even though it's cruder than real NLP.

This CANNOT be backtested: Yahoo's news feed only returns current headlines,
not a historical point-in-time archive, so there's no way to know what
headlines existed 18 months ago without look-ahead bias. This is informational
context only — it does not feed into the backtested Win Rate score.
"""
import yfinance as yf

BULLISH_TERMS = [
    "beats", "beat estimates", "beat expectations", "tops estimates", "raises guidance",
    "raised guidance", "record profit", "record revenue", "upgrade", "upgraded",
    "outperform", "strong buy", "buyback", "share buyback", "surge", "soars", "soared",
    "rally", "rallied", "exceeds expectations", "better-than-expected", "all-time high",
    "acquisition", "expands", "expansion", "partnership", "breakthrough",
]

BEARISH_TERMS = [
    "misses", "miss estimates", "missed expectations", "cuts guidance", "cut guidance",
    "lowered guidance", "downgrade", "downgraded", "underperform", "layoffs", "lawsuit",
    "investigation", "probe", "recall", "plunge", "plunged", "slumps", "slumped",
    "sell-off", "selloff", "bankruptcy", "fraud", "warns", "warning", "weak outlook",
    "guidance cut", "delisting", "scandal", "resigns", "resignation",
]


def _score_text(text):
    text = text.lower()
    bull_hits = [term for term in BULLISH_TERMS if term in text]
    bear_hits = [term for term in BEARISH_TERMS if term in text]
    return bull_hits, bear_hits


def get_news_sentiment(ticker, max_items=8):
    """Returns {"label", "headline_count", "sample_headline", "bull_hits", "bear_hits"}
    or None if no news / fetch failed. Live-only — see module docstring."""
    try:
        news = yf.Ticker(ticker).news
    except Exception:
        return None

    if not news:
        return None

    items = news[:max_items]
    total_bull, total_bear = 0, 0
    sample_headline = None
    all_bull_hits, all_bear_hits = [], []

    for item in items:
        content = item.get("content", {})
        title = content.get("title", "")
        summary = content.get("summary", "")
        if sample_headline is None and title:
            sample_headline = title

        bull_hits, bear_hits = _score_text(f"{title} {summary}")
        total_bull += len(bull_hits)
        total_bear += len(bear_hits)
        all_bull_hits.extend(bull_hits)
        all_bear_hits.extend(bear_hits)

    if total_bull > total_bear:
        label = "bullish"
    elif total_bear > total_bull:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "label": label,
        "headline_count": len(items),
        "sample_headline": sample_headline,
        "bull_hits": sorted(set(all_bull_hits)),
        "bear_hits": sorted(set(all_bear_hits)),
    }
