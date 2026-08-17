"""Tests for move_for_category()/horizon_variants().

one_hour (5-minute candles, 12-bar resolution) is sized from ATR (per-bar
price range as a fraction of price), scaled by sqrt(bars) — this was
validated to help (backtested win rate 18.8% -> 26.3%, timeout 85% -> 40%).

The three daily categories (day_trade/one_week/swing_trade) keep the
original, unscaled Bollinger-band-width floor — the same ATR-scaled-by-
sqrt(bars) approach was tried here too (generalizing the one_hour fix) and
re-backtested, but it made every one of them worse, not better, so it was
reverted for these three specifically. See CONFIDENCE_CALIBRATION's comment
in picks.py for the before/after numbers on both sides of that finding."""
import math

import pytest

from picks import DAILY_CATEGORIES, HORIZON_BARS, fmt_price, horizon_variants, move_for_category


class TestMoveForCategory:
    def test_daily_categories_use_band_width_unscaled_by_bars(self):
        # day_trade (2 bars) and swing_trade (15 bars) should get the exact
        # same move for the same band_width_pct — no sqrt(bars) scaling
        # applies to the daily categories (that's the point being tested).
        day = move_for_category("day_trade", atr=0.01, band_width_pct=0.20)
        swing = move_for_category("swing_trade", atr=0.01, band_width_pct=0.20)
        assert day == swing == pytest.approx(0.10)

    def test_daily_categories_respect_the_2pct_floor(self):
        assert move_for_category("day_trade", atr=0.01, band_width_pct=0.001) == pytest.approx(0.02)

    def test_daily_categories_ignore_atr_entirely(self):
        # A huge ATR shouldn't move the daily categories' target at all —
        # they're driven purely by band_width_pct.
        move = move_for_category("swing_trade", atr=5.0, band_width_pct=0.10)
        assert move == pytest.approx(0.05)

    def test_one_hour_scales_with_sqrt_of_bars(self):
        atr = 0.03  # big enough that the 0.2% floor doesn't kick in
        move = move_for_category("one_hour", atr=atr, band_width_pct=0.10)
        assert move == pytest.approx(atr * math.sqrt(HORIZON_BARS["one_hour"]))

    def test_one_hour_respects_its_own_tighter_floor(self):
        assert move_for_category("one_hour", atr=0.00001, band_width_pct=0.10) == pytest.approx(0.002)

    def test_one_hour_ignores_band_width_entirely(self):
        move = move_for_category("one_hour", atr=0.003, band_width_pct=999)
        assert move == pytest.approx(0.003 * math.sqrt(HORIZON_BARS["one_hour"]))

    def test_one_hour_nan_atr_falls_back_to_its_floor(self):
        assert move_for_category("one_hour", atr=float("nan"), band_width_pct=0.10) == pytest.approx(0.002)


def test_wider_band_gives_a_bigger_daily_move_for_a_buy():
    variants = horizon_variants(100.0, "BUY", atr=0.01, band_width_pct=0.10, score=2)
    tight = variants["day_trade"]["target"] - 100.0
    variants_wide = horizon_variants(100.0, "BUY", atr=0.01, band_width_pct=0.30, score=2)
    wide = variants_wide["day_trade"]["target"] - 100.0
    assert wide > tight > 0


def test_daily_categories_all_get_the_same_move():
    # Since move_for_category() doesn't scale daily categories by bars,
    # day_trade/one_week/swing_trade should all land on the same target.
    variants = horizon_variants(100.0, "BUY", atr=0.01, band_width_pct=0.10, score=2)
    targets = {variants[c]["target"] for c in DAILY_CATEGORIES}
    assert len(targets) == 1


def test_sell_targets_are_below_entry_in_every_horizon():
    variants = horizon_variants(100.0, "SELL", atr=0.01, band_width_pct=0.10, score=2)
    for category in DAILY_CATEGORIES:
        assert variants[category]["target"] < 100.0
        assert variants[category]["stop"] > 100.0


def test_entry_is_identical_across_all_horizons():
    variants = horizon_variants(50.0, "BUY", atr=0.01, band_width_pct=0.12, score=2)
    entries = {variants[c]["entry"] for c in DAILY_CATEGORIES}
    assert entries == {fmt_price(50.0)}


def test_all_daily_categories_are_present_by_default():
    variants = horizon_variants(10.0, "BUY", atr=0.01, band_width_pct=0.05, score=2)
    assert set(variants.keys()) == set(DAILY_CATEGORIES)


def test_move_floor_still_applies_when_band_is_very_tight():
    variants = horizon_variants(100.0, "BUY", atr=0.01, band_width_pct=0.0001, score=2)
    assert variants["swing_trade"]["target"] == fmt_price(100.0 * 1.02)


def test_every_variant_carries_a_backtested_confidence():
    variants = horizon_variants(100.0, "BUY", atr=0.01, band_width_pct=0.10, score=2)
    for category in DAILY_CATEGORIES:
        assert isinstance(variants[category]["confidence"], int)
        assert 0 < variants[category]["confidence"] <= 100


def test_one_hour_can_be_requested_on_its_own():
    variants = horizon_variants(100.0, "BUY", atr=0.003, band_width_pct=0.10, score=2, categories=("one_hour",))
    assert set(variants.keys()) == {"one_hour"}
    assert variants["one_hour"]["target"] > 100.0
    assert isinstance(variants["one_hour"]["confidence"], int)
