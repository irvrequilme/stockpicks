"""
Scans the universe via live yfinance data, scores each ticker on the same
indicator set as the stock-predictor CLI tool (9-RSI, 8/17/9 MACD, 9/20-EMA,
14-period Bollinger Bands, volume vs 10-day avg) plus chart-pattern detection
(double top/bottom, head & shoulders, triangles), and turns the top scorers
into dashboard-ready picks with entry/target/stop levels and a plain-English
thesis built from the triggered signals.

Pure technicals + chart patterns feed the actual signal/score/Win Rate — no
fundamentals or news/sentiment are folded into that (see the note above
CONFIDENCE_CALIBRATION for why). They ARE fetched and attached to each final
pick as separate, clearly-labeled "Context" fields (fundamentals_signal.py,
news_signal.py) purely for the user's information.

The scoring rules live in `evaluate()`, indexed at an arbitrary bar `i`
rather than always "the last bar," so the exact same rules that power the
live dashboard can be replayed bar-by-bar in backtest.py without drifting
into a second, silently-different implementation.
"""
import math
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from universe import UNIVERSE
from patterns import detect_patterns
from crypto_universe import CRYPTO_ALIASES
from news_signal import get_news_sentiment
from fundamentals_signal import get_fundamentals

# Live news sentiment stays out of the actual score (see the module docstring
# and news_signal.py) — but a pick whose live news flatly contradicts the
# backtested signal is still worth flagging for the user, as a soft warning
# rather than a change to which picks qualify or how they're ranked.
NEWS_CONFLICT_MIN_HITS = 2


def _news_conflicts_signal(signal, news_sentiment):
    if not news_sentiment:
        return False
    label = news_sentiment.get("label")
    if signal == "BUY" and label == "bearish":
        return len(news_sentiment.get("bear_hits", [])) >= NEWS_CONFLICT_MIN_HITS
    if signal == "SELL" and label == "bullish":
        return len(news_sentiment.get("bull_hits", [])) >= NEWS_CONFLICT_MIN_HITS
    return False

# Bars-to-resolution assumed for each category — this is the same horizon
# backtest.py uses to check whether a simulated trade hit target/stop, so
# the "estimated time to finish" shown live matches what was actually tested
# (see the calibration note above CONFIDENCE_CALIBRATION).
#
# day_trade/one_week/swing_trade are mapped 1:1 from the existing
# High/Medium/Low risk bands on DAILY candles, so no new detection logic was
# needed there: High-risk (wide-band) setups resolve fastest, Low-risk
# (tight-band) setups get the longest runway. one_hour is a distinct fourth
# category tied specifically to 5-minute candles (12 bars = 60 minutes) —
# it doesn't fit the daily sqrt(time) scaling in horizon_variants() below,
# since it's a genuinely different price series (5m bars), not a rescaled
# view of the same daily signal. See _lookup_ticker()'s interval=="5m" branch.
HORIZON_BARS = {"day_trade": 2, "one_week": 5, "swing_trade": 15, "one_hour": 12}
CATEGORY_LABELS = {
    "day_trade": "day trade", "one_week": "1 week", "swing_trade": "swing (days–weeks)",
    "one_hour": "1 hour",
}
DAILY_CATEGORIES = ("day_trade", "one_week", "swing_trade")

# Which categories a given candle interval can meaningfully produce/test —
# shared by the live universe scan below and backtest.py's simulate_ticker(),
# so both always agree on what "one_hour" means and where it's calibrated
# from. 5-minute candles only ever produce one_hour, never the daily-scale
# categories (see the HORIZON_BARS comment above for why).
CATEGORIES_BY_INTERVAL = {"1d": DAILY_CATEGORIES, "5m": ("one_hour",)}

INTERVAL_MINUTES = {"1m": 1, "5m": 5, "30m": 30}


def estimate_duration(category, interval):
    bars = HORIZON_BARS[category]
    if interval == "1d":
        if bars < 5:
            return f"~{bars} trading days"
        weeks = round(bars / 5, 1)
        return f"~{bars} trading days (~{weeks:g} wk)"

    minutes = bars * INTERVAL_MINUTES[interval]
    if minutes < 60:
        return f"~{minutes} min"
    hours = round(minutes / 60, 1)
    return f"~{hours:g} hr"


def move_for_category(category, atr, band_width_pct):
    """Target/stop distance for `category`.

    one_hour (5-minute candles, 12-bar resolution) is sized from `atr` — the
    per-bar ATR-as-fraction-of-price at whatever candle size produced it
    (see atr_pct()) — scaled by sqrt(bars): the standard random-walk
    assumption that expected price range grows with the square root of
    elapsed time. The three daily categories (day_trade/one_week/
    swing_trade) instead use the fixed 14-period Bollinger-band-width floor,
    unscaled by bars — this was ALSO tried as ATR-scaled-by-sqrt(bars), to
    generalize the fix that helped one_hour, and re-backtested (2-year/
    99-ticker). It clearly helped one_hour (18.8%->26.3% win rate, 85%->40%
    timeout) but made every daily category WORSE, not better (28.4/29.1/
    30.2% -> 25.0/25.3/26.1% win rate — see CONFIDENCE_CALIBRATION's
    comment). So it's kept only where backtesting actually showed it
    helped, not applied everywhere on principle.
    """
    if category == "one_hour":
        floor = 0.002
        if atr is None or math.isnan(atr):
            return floor
        return max(atr * math.sqrt(HORIZON_BARS["one_hour"]), floor)
    return max(band_width_pct / 2, 0.02)


def horizon_variants(price, signal, atr, band_width_pct, score, categories=DAILY_CATEGORIES):
    """Entry/target/stop/confidence for a search result under alternate time
    horizons — every one of them backtested, not just the ticker's natural one.

    Each category's move comes from move_for_category() — see its docstring
    for why one_hour and the three daily categories are sized differently.
    backtest.py's simulate_ticker() applies this exact same sizing to every
    historical signal and tests all resulting targets/stops against their
    own HORIZON_BARS window, so CONFIDENCE_CALIBRATION's per-category Win
    Rate (used below) reflects a real backtested outcome for whichever
    horizon is selected.

    `categories` defaults to the three daily-candle horizons; one_hour (a
    5-minute-candle category) is only ever passed on its own, from a fresh
    intraday analysis, since its `atr` comes from a different price series
    entirely.
    """
    variants = {}
    for category in categories:
        move = move_for_category(category, atr, band_width_pct)
        if signal == "BUY":
            target = price * (1 + move)
            stop = price * (1 - move / 2)
        else:
            target = price * (1 - move)
            stop = price * (1 + move / 2)
        variants[category] = {
            "entry": fmt_price(price),
            "target": fmt_price(target),
            "stop": fmt_price(stop),
            "confidence": calibrated_confidence(score, category),
        }
    return variants

warnings.filterwarnings("ignore")


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=9):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # avg_loss of exactly 0 (the lookback window had zero down-bars) makes the
    # division above NaN via the guard on the line before — resolve it to
    # RSI's actual mathematical limit instead of leaving it undefined: 100 if
    # there was real upward movement, 50 if price didn't move at all either way.
    no_loss = avg_loss == 0
    return result.where(~no_loss, np.where(avg_gain > 0, 100.0, 50.0))


def macd(series, fast=8, slow=17, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(series, period=14, num_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def atr_pct(high, low, close, period=14):
    """Average True Range as a fraction of price — a bar-by-bar measure of
    how far this instrument actually moves, at whatever candle size `high`/
    `low`/`close` are sampled at. Used to size the one_hour target/stop (see
    evaluate()): the daily-tuned Bollinger-band move badly overshoots a
    12-bar/5-minute window (an 85% timeout rate in backtesting — see
    CONFIDENCE_CALIBRATION's comment), because it assumes daily-scale price
    action. ATR measured directly on 5-minute bars doesn't carry that
    assumption over."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean() / close


def fmt_price(x):
    """Rounds to 2 decimals for normal-priced assets, but keeps enough
    precision for sub-$1 assets (many crypto tokens) so they don't collapse to $0."""
    x = float(x)
    if abs(x) >= 1:
        return round(x, 2)
    if abs(x) >= 0.01:
        return round(x, 4)
    return round(x, 6)


# Real historical win rate per (category, score), from a 2-year/99-ticker
# backtest that tests EVERY signal against all three horizons, not just its
# risk-derived natural one (see backtest.py's simulate_ticker(), which sizes
# entry/target/stop per category via horizon_variants() and checks whether
# THAT category's target/stop is hit within THAT category's own HORIZON_BARS
# window). This is what makes every horizon on the search card "accurate":
# Win Rate reflects an outcome actually tested for that category, not the
# ticker's natural bucket extrapolated onto a horizon it was never sized for.
#
# walk_forward_report()'s by_category_and_score confirms the same
# small-sample-noise pattern seen before the cross-horizon split: score 4
# swings roughly 8-10 points between the calibrate/validate folds in every
# category (smallest sample, ~690 trades/category full-period) but its
# full-period average still lands within ~1 point of score 3's stable
# estimate in all three categories, so — as before — no separate capping
# was applied. Scores 1-3 held up within ~1 point across both folds, with
# one exception (swing_trade score 1, ~3.5 points) still well short of
# score 4's swing. This system has not been shown to have a real edge —
# confidence reflects that.
#
# These three keep the original Bollinger-band-width sizing (see
# move_for_category()) — ATR-scaled-by-sqrt(bars), the fix that helped
# one_hour, was also tried here and re-backtested, but made every one of
# them worse (28.4/29.1/30.2% -> 25.0/25.3/26.1% win rate that same day),
# so it was reverted for these three rather than applied on principle.
#
# A second, honest finding from that same re-backtest: re-running the
# identical (reverted) formula on a freshly-fetched 2-year window, just
# hours later, moved day_trade's numbers by ~5 points (28/29/29/28 ->
# 23/24/24/24) even though the formula was verified byte-for-byte
# unchanged (checked directly against individual evaluate() calls, not
# just the aggregate stats). one_week/swing_trade drifted less. This is
# real day-to-day variance in which 2 years of market history gets tested,
# not a bug — and is itself more evidence this system's edge, where it
# exists at all, is fragile enough to be sensitive to exactly which window
# you happen to backtest.
#
# one_hour is calibrated separately, from its own intraday (5-minute-candle,
# 12-bar) backtest — Yahoo only retains ~60 days of 5m history, vs. 2 years
# of daily history for the other three, so this sample is still much smaller
# and noisier than the daily categories.
#
# It originally sized its move the same way as the daily categories (half
# the 14-period Bollinger Band width, floored at 2%) — a daily-tuned floor
# that badly overshot a 12-bar/5-minute window: 85.4% of trades timed out
# and the win rate was 18.8%, well below the 33.3% breakeven. Switching to
# an ATR-based move (see atr_pct()), sized from actual recent 5-minute-bar
# range rather than a daily assumption, cut the timeout rate to 40.0% and
# lifted the win rate to 26.3% overall — still below breakeven, but a real,
# stable improvement: walk-forward swings shrank to ≤1 point per score
# (calibrate-fold 25.3-28.5% vs. validate-fold 25.9-28.5%), vs. up to 5.7
# points before. Still the least reliable of the four horizons — just no
# longer badly miscalibrated on top of that. `python3 backtest.py intraday`.
CONFIDENCE_CALIBRATION = {
    "day_trade": {1: 23, 2: 24, 3: 24, 4: 24},
    "one_week": {1: 28, 2: 29, 3: 30, 4: 31},
    "swing_trade": {1: 32, 2: 33, 3: 33, 4: 33},
    "one_hour": {1: 26, 2: 26, 3: 27, 4: 29},
}


def calibrated_confidence(score, category):
    return CONFIDENCE_CALIBRATION.get(category, {}).get(score, 30)


def compute_indicators(close, high, low, volume):
    """Vectorized, computed once over the full series. evaluate() then just
    indexes into these at bar `i` — critical for backtest performance, since
    recomputing from scratch at every bar would make a multi-year backtest
    across dozens of tickers far too slow."""
    macd_line, signal_line, hist_line = macd(close)
    return {
        "rsi": rsi(close),
        "macd_hist": hist_line,
        "ema9": ema(close, 9),
        "ema20": ema(close, 20),
        "bb_upper": bollinger(close)[0],
        "bb_lower": bollinger(close)[2],
        "vol_avg10": volume.rolling(10).mean(),
        "atr_pct": atr_pct(high, low, close),
    }


def evaluate(ticker, close, volume, ind, i, interval="1d"):
    """Applies the scoring rules as of bar `i` (0-indexed into `close`/`volume`).
    Returns the same shape as a live pick, plus a `components` dict recording
    which named signal fired bull/bear/None — used by backtest.py to measure
    each signal's individual predictive value, not just the combined score.
    `interval` selects the candle size, used both to scale chart-pattern
    detection (see patterns.INTERVAL_CONFIG) and to pick which category/move-
    sizing this bar produces (one_hour for "5m", the risk-derived daily
    category for "1d") — backtest.py's 2-year run always uses daily bars, so
    those calls rely on the "1d" default rather than passing this explicitly."""
    if i < 25:
        return None
    price = close.iloc[i]
    if price is None or np.isnan(price) or price <= 0:
        return None

    r = ind["rsi"].iloc[i]
    hist_val, hist_prev = ind["macd_hist"].iloc[i], ind["macd_hist"].iloc[i - 1]
    ema9, ema20 = ind["ema9"].iloc[i], ind["ema20"].iloc[i]
    ema9_prev, ema20_prev = ind["ema9"].iloc[i - 1], ind["ema20"].iloc[i - 1]
    bb_upper, bb_lower = ind["bb_upper"].iloc[i], ind["bb_lower"].iloc[i]
    vol_avg10 = ind["vol_avg10"].iloc[i]
    vol_ratio = volume.iloc[i] / vol_avg10 if vol_avg10 else np.nan

    reasons_bull, reasons_bear = [], []
    components = {"macd": None, "ema": None, "pattern": None}
    bull, bear = 0, 0

    # RSI and Bollinger-band mean-reversion votes were removed after backtest.py
    # showed they were the two weakest signals (24.8% and 27.4% win rate when
    # they fired, vs. a 33.3% breakeven bar) — see backtest_results.json.
    # RSI is still used below as a filter, not a standalone vote.

    if hist_val > 0 and hist_val > hist_prev:
        bull += 1
        components["macd"] = "bull"
        reasons_bull.append("MACD histogram is positive and rising")
    elif hist_val < 0 and hist_val < hist_prev:
        bear += 1
        components["macd"] = "bear"
        reasons_bear.append("MACD histogram is negative and falling")

    crossed_up = ema9_prev <= ema20_prev and ema9 > ema20
    crossed_down = ema9_prev >= ema20_prev and ema9 < ema20
    if crossed_up:
        bull += 1
        components["ema"] = "bull"
        reasons_bull.append("9-EMA just crossed above the 20-EMA")
    elif ema9 > ema20 and r < 75:
        bull += 1
        components["ema"] = "bull"
        reasons_bull.append("price trending above the 20-EMA")
    if crossed_down:
        bear += 1
        components["ema"] = "bear"
        reasons_bear.append("9-EMA just crossed below the 20-EMA")
    elif ema9 < ema20 and r > 25:
        bear += 1
        components["ema"] = "bear"
        reasons_bear.append("price trending below the 20-EMA")

    detected_patterns = detect_patterns(close.iloc[: i + 1], interval=interval)
    if any(p["direction"] == "bullish" for p in detected_patterns):
        bull += 1
        components["pattern"] = "bull"
        for p in detected_patterns:
            if p["direction"] == "bullish":
                reasons_bull.append(f"{p['name']} pattern ({p['detail']})")
    if any(p["direction"] == "bearish" for p in detected_patterns):
        bear += 1
        components["pattern"] = "bear"
        for p in detected_patterns:
            if p["direction"] == "bearish":
                reasons_bear.append(f"{p['name']} pattern ({p['detail']})")

    volume_confirmed = bool(vol_ratio >= 1.5) if not np.isnan(vol_ratio) else False
    if volume_confirmed:
        reasons_bull.append(f"volume is {vol_ratio:.1f}x the 10-day average")
        reasons_bear.append(f"volume is {vol_ratio:.1f}x the 10-day average")

    if bull == bear:
        return None  # no clear direction, skip

    signal = "BUY" if bull > bear else "SELL"
    score = max(bull, bear) + (1 if volume_confirmed else 0)
    reasons = reasons_bull if signal == "BUY" else reasons_bear

    band_width_pct = float((bb_upper - bb_lower) / price)
    if band_width_pct < 0.08:
        risk = "Low"
    elif band_width_pct < 0.16:
        risk = "Medium"
    else:
        risk = "High"

    if interval == "5m":
        category = "one_hour"
    elif interval != "1d":
        # 1m/30m: not exposed in the current UI, and never separately
        # backtested — keep the pre-existing "treat it as day_trade" fallback
        # rather than claiming a horizon that isn't calibrated.
        category = "day_trade"
    else:
        category = {"High": "day_trade", "Medium": "one_week", "Low": "swing_trade"}[risk]

    atr = float(ind["atr_pct"].iloc[i])
    move = move_for_category(category, atr, band_width_pct)
    if signal == "BUY":
        target = price * (1 + move)
        stop = price * (1 - move / 2)
    else:
        target = price * (1 - move)
        stop = price * (1 + move / 2)

    confidence = calibrated_confidence(score, category)
    sparkline = [round(float(x), 6) for x in close.iloc[max(0, i - 19): i + 1].tolist()]

    return {
        "ticker": ticker,
        "price": fmt_price(price),
        "signal": signal,
        "confidence": confidence,
        "entry": fmt_price(price),
        "target": fmt_price(target),
        "stop": fmt_price(stop),
        "risk": risk,
        "category": category,
        "time_horizon": CATEGORY_LABELS[category],
        "thesis": "; ".join(reasons).capitalize() + ".",
        "score": score,
        "components": components,
        "volume_confirmed": volume_confirmed,
        "sparkline": sparkline,
        "atr_pct": atr,
        "band_width_pct": band_width_pct,
    }


def analyze(ticker, hist, interval="1d"):
    close = hist["Close"]
    volume = hist["Volume"]
    if len(close) < 25:
        return None
    ind = compute_indicators(close, hist["High"], hist["Low"], volume)
    return evaluate(ticker, close, volume, ind, len(close) - 1, interval=interval)


# Yahoo Finance caps how far back intraday candles are available at all —
# these periods are the maximum each interval supports (1m: ~7d, 5m/30m: ~60d).
INTERVAL_PERIODS = {
    "1d": "6mo",
    "1m": "5d",
    "5m": "60d",
    "30m": "60d",
}


_PICKS_CACHE = {}
PICKS_CACHE_TTL_SECONDS = 45  # absorbs rapid double-clicks / multiple open tabs


def get_categorized_picks(top_n=10, tickers=None, interval="1d"):
    """Cached wrapper around _get_categorized_picks_uncached(). A full scan
    already takes up to a minute and hits Yahoo Finance for ~20-100 tickers
    plus fundamentals/news for the top picks — a short TTL cache means
    clicking Refresh twice, or having the dashboard open in two tabs, doesn't
    double that load. 45s is short enough that data still feels live."""
    cache_key = (tuple(tickers) if tickers else "default", top_n, interval)
    cached = _PICKS_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < PICKS_CACHE_TTL_SECONDS:
        return cached[1]

    result = _get_categorized_picks_uncached(top_n=top_n, tickers=tickers, interval=interval)
    _PICKS_CACHE[cache_key] = (time.time(), result)
    return result


def _get_categorized_picks_uncached(top_n=10, tickers=None, interval="1d"):
    """Returns one bucket per category this interval supports (see
    CATEGORIES_BY_INTERVAL), each ranked by confidence — {"day_trade": [...],
    "one_week": [...], "swing_trade": [...]} for interval="1d",
    {"one_hour": [...]} for interval="5m".

    `interval` selects the candle size: "1d" (default), "1m", "5m", or "30m".
    one_hour (interval="5m") has its own separately-backtested calibration
    (see CONFIDENCE_CALIBRATION) — 1m/30m fall back to day_trade's daily
    calibration, an unvalidated carryover, since neither is separately
    tested or exposed in the current UI.
    """
    universe = tickers or UNIVERSE
    period = INTERVAL_PERIODS.get(interval, "6mo")
    data = yf.download(universe, period=period, interval=interval, group_by="ticker", threads=True, progress=False, auto_adjust=True)

    results = []
    for t in universe:
        try:
            # yf.download(list, group_by="ticker") returns MultiIndex columns
            # even for a single-element list — key off the actual column
            # structure rather than guessing from len(universe).
            hist = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
        except Exception:
            continue
        r = analyze(t, hist, interval=interval)
        if r:
            results.append(r)

    for r in results:
        r["estimated_duration"] = estimate_duration(r["category"], interval)

    categorized = {}
    for cat in CATEGORIES_BY_INTERVAL.get(interval, ("day_trade",)):
        bucket = sorted((r for r in results if r["category"] == cat), key=lambda x: x["confidence"], reverse=True)
        categorized[cat] = bucket[:top_n]

    for r in [item for bucket in categorized.values() for item in bucket]:
        # One get_info() call per ticker, reused for both the display name and
        # fundamentals — this used to be two separate Yahoo Finance round trips
        # per pick, which was needlessly doubling exposure to Yahoo's rate limits.
        try:
            info = yf.Ticker(r["ticker"]).get_info()
        except Exception:
            info = None
        r["name"] = (info or {}).get("shortName", r["ticker"])
        # Live-only context, fetched only for the final top-N picks (not the
        # whole universe scan) — neither can be backtested since yfinance
        # only exposes current snapshots, not point-in-time history, so
        # neither factors into the backtested Win Rate. See news_signal.py
        # and fundamentals_signal.py docstrings for why.
        r["fundamentals"] = get_fundamentals(r["ticker"], info=info)
        r["news_sentiment"] = get_news_sentiment(r["ticker"])
        r["news_conflict"] = _news_conflicts_signal(r["signal"], r["news_sentiment"])
        del r["score"]
        del r["components"]
        del r["volume_confirmed"]
        del r["atr_pct"]
        del r["band_width_pct"]

    return categorized


def _lookup_ticker(ticker, interval):
    """Attempts to pull data + build a pick for an exact ticker symbol.
    Returns None if that exact symbol has no usable data on Yahoo Finance."""
    period = INTERVAL_PERIODS.get(interval, "6mo")
    try:
        data = yf.download([ticker], period=period, interval=interval, group_by="ticker", progress=False, auto_adjust=True)
        hist = data[ticker].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
    except Exception:
        return None
    if hist.empty or len(hist["Close"]) < 25:
        return None

    try:
        name = yf.Ticker(ticker).get_info().get("shortName", ticker)
    except Exception:
        name = ticker

    pick = analyze(ticker, hist, interval=interval)
    if pick is None:
        last_price = hist["Close"].iloc[-1]
        return {
            "ticker": ticker,
            "name": name,
            "price": fmt_price(last_price),
            "signal": None,
            "thesis": "No clear signal right now — RSI/MACD/EMA/pattern don't agree on a direction.",
        }

    # For interval="1d" this offers all three daily horizons (Day Trade/1
    # Week/Swing Trade); for 5m (one_hour) or 1m/30m (day_trade fallback) —
    # each a single, standalone category — it's just that one category,
    # sized identically to what evaluate() already computed for it (same
    # move_for_category() call, so the two can never drift apart).
    categories = DAILY_CATEGORIES if interval == "1d" else (pick["category"],)
    pick["horizons"] = horizon_variants(pick["price"], pick["signal"], pick["atr_pct"], pick["band_width_pct"], pick["score"], categories=categories)

    pick["time_horizon"] = CATEGORY_LABELS[pick["category"]]
    pick["estimated_duration"] = estimate_duration(pick["category"], interval)
    pick["name"] = name
    del pick["score"]
    del pick["components"]
    del pick["volume_confirmed"]
    del pick["band_width_pct"]
    del pick["atr_pct"]
    return pick


def search_ticker(ticker, interval="1d"):
    """On-demand single-ticker lookup for the dashboard's search bar.

    Unlike get_categorized_picks (which only surfaces tickers with a clear
    signal), this always returns something: current price and name even when
    RSI/MACD/EMA/pattern don't agree, so a search never comes back empty.

    Checks the crypto alias table first (see crypto_universe.CRYPTO_ALIASES)
    since bare short codes like "BTC" or "LINK" silently collide with
    unrelated Yahoo Finance tickers rather than failing cleanly. If that
    doesn't match, falls back to Yahoo Finance's own search to find the
    closest real symbol for a typo or company name like "tesla". Returns
    None only if nothing — alias, exact input, or any suggestion — resolves.
    """
    typed = ticker.strip()

    alias = CRYPTO_ALIASES.get(typed.lower())
    if alias:
        aliased = _lookup_ticker(alias, interval)
        if aliased is not None:
            aliased["corrected_from"] = typed.upper() if alias != typed.upper() else None
            return aliased

    exact = _lookup_ticker(typed.upper(), interval)
    if exact is not None:
        exact["corrected_from"] = None
        return exact

    try:
        suggestions = yf.Search(typed, max_results=5).quotes
    except Exception:
        suggestions = []

    for s in suggestions:
        symbol = s.get("symbol")
        if not symbol or symbol.upper() == typed.upper():
            continue
        corrected = _lookup_ticker(symbol, interval)
        if corrected is not None:
            corrected["corrected_from"] = typed.upper()
            return corrected

    return None
