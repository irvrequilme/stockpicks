"""Unit tests for the pure indicator math in picks.py — no network calls."""
import numpy as np
import pandas as pd
import pytest

from picks import ema, rsi, macd, bollinger, fmt_price, atr_pct


def _series(values):
    return pd.Series(values, dtype=float)


def test_ema_converges_toward_a_flat_series():
    s = _series([100.0] * 30)
    result = ema(s, span=9)
    assert result.iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_rsi_is_100_when_every_bar_gains():
    s = _series([100 + i for i in range(30)])  # strictly increasing
    result = rsi(s, period=9)
    assert result.iloc[-1] == pytest.approx(100.0, abs=0.5)


def test_rsi_is_0_when_every_bar_loses():
    s = _series([130 - i for i in range(30)])  # strictly decreasing
    result = rsi(s, period=9)
    assert result.iloc[-1] == pytest.approx(0.0, abs=0.5)


def test_rsi_is_near_50_for_alternating_series():
    s = _series([100 + (1 if i % 2 == 0 else -1) for i in range(30)])
    result = rsi(s, period=9)
    assert 30 < result.iloc[-1] < 70


def test_macd_histogram_is_positive_when_price_accelerates_upward():
    # Strongly accelerating uptrend should push MACD histogram positive.
    s = _series([100 * (1.02 ** i) for i in range(40)])
    macd_line, signal_line, hist = macd(s)
    assert hist.iloc[-1] > 0


def test_bollinger_bands_bracket_a_stable_price():
    s = _series([100.0] * 20)
    upper, mid, lower = bollinger(s, period=14, num_std=2)
    assert lower.iloc[-1] <= mid.iloc[-1] <= upper.iloc[-1]
    assert mid.iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_atr_pct_is_zero_for_a_perfectly_flat_series():
    close = _series([100.0] * 20)
    high = _series([100.0] * 20)
    low = _series([100.0] * 20)
    result = atr_pct(high, low, close, period=14)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_atr_pct_reflects_actual_bar_range():
    # A steady 1-point high-low range on a $100 close should land near 1%,
    # not near the daily-tuned Bollinger floor (2%) — this is the whole
    # point of sizing one_hour from ATR instead of that floor.
    close = _series([100.0] * 20)
    high = _series([100.5] * 20)
    low = _series([99.5] * 20)
    result = atr_pct(high, low, close, period=14)
    assert result.iloc[-1] == pytest.approx(0.01, abs=0.001)


def test_atr_pct_grows_with_wider_bars():
    close = _series([100.0] * 20)
    tight_high, tight_low = _series([100.2] * 20), _series([99.8] * 20)
    wide_high, wide_low = _series([102.0] * 20), _series([98.0] * 20)
    tight = atr_pct(tight_high, tight_low, close, period=14).iloc[-1]
    wide = atr_pct(wide_high, wide_low, close, period=14).iloc[-1]
    assert wide > tight


class TestFmtPrice:
    def test_rounds_normal_prices_to_2_decimals(self):
        assert fmt_price(123.456789) == 123.46

    def test_keeps_4_decimals_for_sub_dollar_prices(self):
        assert fmt_price(0.073456) == 0.0735

    def test_keeps_6_decimals_for_sub_cent_prices(self):
        assert fmt_price(0.0000734) == 0.000073

    def test_handles_string_input(self):
        # evaluate()/backtest.py always pass floats, but fmt_price shouldn't
        # silently misbehave if a numeric string ever reaches it.
        result = fmt_price("5.25")
        assert isinstance(result, float)
        assert result == pytest.approx(5.25)
