"""Tests for evaluate() — the core scoring rule that both the live dashboard
and backtest.py share. Uses synthetic price series, no network calls."""
import numpy as np
import pandas as pd
import pytest

from picks import compute_indicators, evaluate, HORIZON_BARS, CATEGORY_LABELS


def _make_hist(closes, volumes=None):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(closes, index=idx, dtype=float)
    # Synthetic high/low: a small fixed band around close, just enough for
    # atr_pct() to have real (non-zero, non-degenerate) input — the exact
    # width isn't load-bearing for these tests, only its presence is.
    high = close * 1.002
    low = close * 0.998
    volume = pd.Series(volumes or [1_000_000] * n, index=idx, dtype=float)
    return close, high, low, volume


def test_evaluate_returns_none_before_bar_25():
    close, high, low, volume = _make_hist([100.0] * 30)
    ind = compute_indicators(close, high, low, volume)
    assert evaluate("TEST", close, volume, ind, 10) is None


def test_evaluate_returns_none_for_non_positive_price():
    closes = [100.0] * 25 + [0.0]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    assert evaluate("TEST", close, volume, ind, len(close) - 1) is None


def test_evaluate_on_strong_uptrend_leans_bullish_or_is_none():
    # A clean, strong uptrend should never produce a SELL signal — either
    # BUY (if enough of the 3 signals agree) or None (if they don't), but
    # a sustained uptrend flipping out a SELL call would indicate a real bug.
    closes = [100 * (1.03 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1)
    if result is not None:
        assert result["signal"] == "BUY"
        assert result["target"] > result["entry"] > result["stop"]


def test_evaluate_on_strong_downtrend_leans_bearish_or_is_none():
    closes = [100 * (0.97 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1)
    if result is not None:
        assert result["signal"] == "SELL"
        assert result["stop"] > result["entry"] > result["target"]


def test_evaluate_returns_none_on_flat_series_with_no_clear_direction():
    # A perfectly flat series shouldn't manufacture a directional call.
    close, high, low, volume = _make_hist([100.0] * 60)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1)
    assert result is None


def test_confidence_is_always_within_calibrated_bounds():
    closes = [100 * (1.02 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1)
    if result is not None:
        # Calibration table only has entries for scores 1-4 — confidence
        # should never silently exceed what's actually been backtested.
        assert 0 < result["confidence"] <= 35


def test_category_and_horizon_are_consistent():
    closes = [100 * (1.02 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1)
    if result is not None:
        assert result["category"] in HORIZON_BARS
        assert result["time_horizon"] == CATEGORY_LABELS[result["category"]]


def test_risk_high_always_maps_to_day_trade_category():
    # This mapping is load-bearing for backtest.py's horizon lookup and the
    # live dashboard's tab split — a mismatch here would silently break both.
    from picks import evaluate as _evaluate  # re-import to keep this test self-contained
    closes = [100 * (1.05 ** i) for i in range(60)]  # very volatile -> High risk band
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = _evaluate("TEST", close, volume, ind, len(close) - 1)
    if result is not None and result["risk"] == "High":
        assert result["category"] == "day_trade"


def test_5m_interval_always_produces_one_hour_category():
    # Regardless of what the risk band would otherwise imply, a "5m" call
    # must land on one_hour — it's a standalone category, not one of the
    # three daily-scale ones (see CATEGORIES_BY_INTERVAL).
    closes = [100 * (1.02 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1, interval="5m")
    if result is not None:
        assert result["category"] == "one_hour"
        assert result["time_horizon"] == CATEGORY_LABELS["one_hour"]


def test_5m_interval_sizes_target_from_atr_not_the_daily_floor():
    # A very tight, low-volatility uptrend: band_width_pct/2 would floor at
    # the old daily 2% minimum, hugely oversized for a 12-bar/5-minute
    # window. The ATR-based one_hour sizing should produce a much tighter,
    # more plausible move instead of reusing that floor.
    closes = [100 * (1.0005 ** i) for i in range(60)]
    close, high, low, volume = _make_hist(closes)
    ind = compute_indicators(close, high, low, volume)
    result = evaluate("TEST", close, volume, ind, len(close) - 1, interval="5m")
    if result is not None:
        move_pct = abs(result["target"] - result["entry"]) / result["entry"]
        assert move_pct < 0.02  # tighter than the old daily 2% floor
