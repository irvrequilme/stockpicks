"""Tests for _news_conflicts_signal() — the live-only soft warning that flags
when news sentiment contradicts a pick's backtested signal. This never changes
the signal/score/ranking itself, only whether the flag is set."""
from picks import _news_conflicts_signal


def test_no_conflict_when_news_sentiment_is_none():
    assert _news_conflicts_signal("BUY", None) is False


def test_buy_conflicts_with_strongly_bearish_news():
    news = {"label": "bearish", "bear_hits": ["misses", "downgrade"], "bull_hits": []}
    assert _news_conflicts_signal("BUY", news) is True


def test_sell_conflicts_with_strongly_bullish_news():
    news = {"label": "bullish", "bull_hits": ["beats", "upgrade"], "bear_hits": []}
    assert _news_conflicts_signal("SELL", news) is True


def test_buy_does_not_conflict_with_bullish_news():
    news = {"label": "bullish", "bull_hits": ["beats", "upgrade"], "bear_hits": []}
    assert _news_conflicts_signal("BUY", news) is False


def test_weak_single_hit_does_not_count_as_conflict():
    # A single loose keyword match is noise, not a real contradiction.
    news = {"label": "bearish", "bear_hits": ["warns"], "bull_hits": []}
    assert _news_conflicts_signal("BUY", news) is False


def test_neutral_news_never_conflicts():
    news = {"label": "neutral", "bear_hits": [], "bull_hits": []}
    assert _news_conflicts_signal("BUY", news) is False
    assert _news_conflicts_signal("SELL", news) is False
