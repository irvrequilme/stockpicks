"""
Live-only fundamentals context: analyst consensus rating and analyst target
price vs. current price, via yfinance.get_info().

This CANNOT be backtested either: get_info() only returns yfinance's current
snapshot (today's analyst consensus, today's P/E), not point-in-time history.
Using today's analyst rating to "explain" a signal from 18 months ago would
be look-ahead bias — the rating didn't exist yet at that point in time. So,
like news_signal.py, this is informational context only, not folded into the
backtested Win Rate score. Not available for crypto (no analysts/earnings).
"""
import yfinance as yf

BULLISH_RATINGS = {"strong_buy", "buy"}
BEARISH_RATINGS = {"sell", "strong_sell"}


def get_fundamentals(ticker, info=None):
    """Returns {"analyst_rating", "target_price", "upside_pct"} or None if
    unavailable (e.g. crypto, or a ticker yfinance has no analyst data for).

    Pass a pre-fetched `info` dict (from yf.Ticker(ticker).get_info()) when
    the caller already has one, to avoid a second Yahoo Finance round trip
    for the same ticker — get_categorized_picks() already fetches it for
    the display name, and doubling API calls per pick was contributing to
    Yahoo's rate limiting."""
    if info is None:
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            return None

    rating = info.get("recommendationKey")
    target = info.get("targetMeanPrice")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    if not rating and not target:
        return None

    upside_pct = None
    if target and price:
        upside_pct = round((target - price) / price * 100, 1)

    return {
        "analyst_rating": rating,
        "target_price": target,
        "upside_pct": upside_pct,
    }
