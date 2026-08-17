"""
Detects classic technical-analysis chart patterns (double top/bottom, head &
shoulders, triangles) from a peak/trough decomposition of the closing price
series. This is heuristic pattern-matching against the textbook definitions
(similar-height peaks/troughs, converging/diverging trendlines within a
tolerance band) — not a proprietary pattern library or ML model.
"""
import numpy as np
from scipy.signal import find_peaks

# Prominence (minimum peak/trough height, as a % of mean price) and lookback
# (in bars) both need to scale with candle size — intraday price swings are
# far smaller in % terms than daily swings over the same bar count, and a
# fixed 90-bar/3% setting (right for daily bars) finds almost no extrema at
# all on 1m/5m/30m candles. Values below were empirically chosen so each
# interval finds a comparable handful of peaks/troughs (~2-6) over a roughly
# similar wall-clock window (~1-2 days), not just copied from daily bars.
INTERVAL_CONFIG = {
    "1d": {"lookback": 90, "prominence_pct": 0.03},
    "30m": {"lookback": 90, "prominence_pct": 0.01},
    "5m": {"lookback": 540, "prominence_pct": 0.01},
    "1m": {"lookback": 1500, "prominence_pct": 0.01},
}


def _fmt(x):
    """Local price formatter (mirrors picks.fmt_price) — kept separate to
    avoid a circular import between picks.py and patterns.py."""
    x = float(x)
    if abs(x) >= 1:
        return f"{x:.2f}"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.6f}"


def _find_extrema(close, prominence_pct=0.03, distance=5):
    prices = close.values
    prominence = prices.mean() * prominence_pct
    peak_idx, _ = find_peaks(prices, prominence=prominence, distance=distance)
    trough_idx, _ = find_peaks(-prices, prominence=prominence, distance=distance)
    return peak_idx, trough_idx


def _similar(a, b, tol=0.03):
    return abs(a - b) / max(a, b) <= tol


def detect_patterns(close, interval="1d"):
    """Returns a list of {"name", "direction", "detail"} for patterns found
    in the most recent bars of a Close price Series, using the lookback and
    prominence tuned for the given candle `interval`."""
    cfg = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1d"])
    close = close.iloc[-cfg["lookback"]:]
    if len(close) < 20:
        return []

    peak_idx, trough_idx = _find_extrema(close, prominence_pct=cfg["prominence_pct"])
    prices = close.values
    last_price = prices[-1]
    patterns = []

    # --- Double top / double bottom: last two peaks or troughs at similar height ---
    if len(peak_idx) >= 2:
        p1, p2 = peak_idx[-2], peak_idx[-1]
        if _similar(prices[p1], prices[p2]) and last_price < min(prices[p1], prices[p2]):
            patterns.append({
                "name": "double top",
                "direction": "bearish",
                "detail": f"two peaks near ${_fmt(prices[p1])}/${_fmt(prices[p2])}, price has broken down since",
            })

    if len(trough_idx) >= 2:
        t1, t2 = trough_idx[-2], trough_idx[-1]
        if _similar(prices[t1], prices[t2]) and last_price > max(prices[t1], prices[t2]):
            patterns.append({
                "name": "double bottom",
                "direction": "bullish",
                "detail": f"two troughs near ${_fmt(prices[t1])}/${_fmt(prices[t2])}, price has broken out since",
            })

    # --- Head & shoulders / inverse: 3 peaks + 2 troughs (or vice versa), interleaved ---
    if len(peak_idx) >= 3 and len(trough_idx) >= 2:
        p, t = peak_idx[-3:], trough_idx[-2:]
        if p[0] < t[0] < p[1] < t[1] < p[2]:
            left, head, right = prices[p[0]], prices[p[1]], prices[p[2]]
            neckline = (prices[t[0]] + prices[t[1]]) / 2
            if head > left and head > right and _similar(left, right, tol=0.05) and last_price < neckline:
                patterns.append({
                    "name": "head and shoulders",
                    "direction": "bearish",
                    "detail": f"head at ${_fmt(head)} above shoulders ~${_fmt(left)}/${_fmt(right)}, broke neckline ~${_fmt(neckline)}",
                })

    if len(trough_idx) >= 3 and len(peak_idx) >= 2:
        t, p = trough_idx[-3:], peak_idx[-2:]
        if t[0] < p[0] < t[1] < p[1] < t[2]:
            left, head, right = prices[t[0]], prices[t[1]], prices[t[2]]
            neckline = (prices[p[0]] + prices[p[1]]) / 2
            if head < left and head < right and _similar(left, right, tol=0.05) and last_price > neckline:
                patterns.append({
                    "name": "inverse head and shoulders",
                    "direction": "bullish",
                    "detail": f"head at ${_fmt(head)} below shoulders ~${_fmt(left)}/${_fmt(right)}, broke neckline ~${_fmt(neckline)}",
                })

    # --- Triangles: fit trendlines to the last 3 peaks and last 3 troughs ---
    if len(peak_idx) >= 3 and len(trough_idx) >= 3:
        p_idx, t_idx = peak_idx[-3:], trough_idx[-3:]
        peak_slope = np.polyfit(p_idx, prices[p_idx], 1)[0]
        trough_slope = np.polyfit(t_idx, prices[t_idx], 1)[0]
        flat_threshold = last_price * 0.001  # slope-per-bar treated as "flat"

        if abs(peak_slope) < flat_threshold and trough_slope > flat_threshold:
            patterns.append({
                "name": "ascending triangle",
                "direction": "bullish",
                "detail": "flat resistance overhead with rising support (higher lows)",
            })
        elif abs(trough_slope) < flat_threshold and peak_slope < -flat_threshold:
            patterns.append({
                "name": "descending triangle",
                "direction": "bearish",
                "detail": "flat support underneath with falling resistance (lower highs)",
            })

    return patterns
