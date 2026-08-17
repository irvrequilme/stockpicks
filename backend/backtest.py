"""
Backtests the exact scoring rules in picks.py (via evaluate()) against
historical daily bars, to answer the question the live dashboard can't
answer on its own: does this scoring system have any real predictive value,
and which of the 6 signals are actually pulling their weight?

Known limitations — read before trusting the numbers:
- Daily-bar granularity can't properly validate the "day_trade" category:
  intraday path (which of target/stop got hit first within a day) isn't
  observable from daily OHLC, so day-trade results are the least trustworthy.
- Trades are independent hypothetical positions, not a capital-constrained
  portfolio — this measures per-signal hit rate/return, not account P&L.
- Overlapping trades on the same ticker aren't excluded, so samples are
  autocorrelated, not fully independent.
- No transaction costs, slippage, spread, or borrow cost modeled.
- If a bar's range crosses both target and stop, it's conservatively scored
  as a loss (can't know which was hit first from daily OHLC).
"""
import json
import sys
import warnings
from collections import defaultdict

import yfinance as yf

from universe import UNIVERSE
from crypto_universe import CRYPTO_UNIVERSE
from picks import compute_indicators, evaluate, horizon_variants, HORIZON_BARS, CATEGORIES_BY_INTERVAL

warnings.filterwarnings("ignore")


def simulate_ticker(ticker, hist, interval="1d"):
    """For every signal, tests every horizon this interval supports — not
    just the ticker's risk-derived natural one (for daily candles) — using
    the exact same target/stop sizing the live app offers on the search card
    (horizon_variants(); see move_for_category() in picks.py for how each
    category is sized). Each (signal, category) combination becomes its own
    trade, tagged with which category it was tested as (`category`) and
    which one the signal's own risk band naturally picked
    (`natural_category`), so the report can calibrate a real, backtested Win
    Rate per category — not just for the natural bucket — letting a user
    pick any horizon and get an accurate, tested number back instead of an
    untested extrapolation."""
    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    n = len(close)
    if n < 120:
        return []

    ind = compute_indicators(close, high, low, volume)
    trades = []
    categories = CATEGORIES_BY_INTERVAL[interval]

    for i in range(30, n - 1):
        r = evaluate(ticker, close, volume, ind, i, interval=interval)
        if r is None:
            continue

        variants = horizon_variants(r["entry"], r["signal"], r["atr_pct"], r["band_width_pct"], r["score"], categories=categories)

        for category in categories:
            bars = HORIZON_BARS[category]
            end = min(i + bars, n - 1)
            if end <= i:
                continue

            entry = variants[category]["entry"]
            target = variants[category]["target"]
            stop = variants[category]["stop"]

            outcome = "timeout"
            exit_price, exit_idx = close.iloc[end], end
            for j in range(i + 1, end + 1):
                if r["signal"] == "BUY":
                    hit_target, hit_stop = high.iloc[j] >= target, low.iloc[j] <= stop
                else:
                    hit_target, hit_stop = low.iloc[j] <= target, high.iloc[j] >= stop
                if hit_stop:
                    outcome, exit_price, exit_idx = "loss", stop, j
                    break
                if hit_target:
                    outcome, exit_price, exit_idx = "win", target, j
                    break

            ret = (exit_price - entry) / entry * (1 if r["signal"] == "BUY" else -1)
            risk_pct = abs(entry - stop) / entry

            trades.append({
                "ticker": ticker,
                "date": str(close.index[i].date()),
                "exit_date": str(close.index[exit_idx].date()),
                "signal": r["signal"],
                "score": r["score"],
                "category": category,
                "natural_category": r["category"],
                "components": r["components"],
                "outcome": outcome,
                "return_pct": round(ret * 100, 2),
                "risk_pct": round(risk_pct, 5),
            })

    return trades


def run_backtest(tickers, period="2y", interval="1d"):
    print(f"Downloading {period} of {interval} history for {len(tickers)} tickers...", file=sys.stderr)
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False, auto_adjust=True)

    all_trades = []
    for t in tickers:
        try:
            hist = data[t].dropna() if len(tickers) > 1 else data.dropna()
        except Exception:
            continue
        try:
            all_trades.extend(simulate_ticker(t, hist, interval=interval))
        except Exception as e:
            print(f"  skipped {t}: {e}", file=sys.stderr)

    return all_trades


def stats(subset):
    decided = [t for t in subset if t["outcome"] in ("win", "loss")]
    wins = [t for t in decided if t["outcome"] == "win"]
    avg_return = sum(t["return_pct"] for t in subset) / len(subset) if subset else 0
    win_rate = len(wins) / len(decided) * 100 if decided else 0
    return {
        "trades": len(subset),
        "decided": len(decided),
        "timeouts": len(subset) - len(decided),
        "win_rate_pct": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 2),
    }


def summarize(trades):
    report = {"overall": stats(trades)}

    by_category = defaultdict(list)
    for t in trades:
        by_category[t["category"]].append(t)
    report["by_category"] = {k: stats(v) for k, v in by_category.items()}

    # Same breakdown, but only trades where the tested category matched the
    # signal's own risk-derived category — the subset the OLD (pre-cross-horizon)
    # backtest measured. Useful for sanity-checking that cross-horizon testing
    # didn't change the natural-category numbers much.
    natural_only = [t for t in trades if t["category"] == t["natural_category"]]
    by_category_natural_only = defaultdict(list)
    for t in natural_only:
        by_category_natural_only[t["category"]].append(t)
    report["by_category_natural_only"] = {k: stats(v) for k, v in by_category_natural_only.items()}

    by_score = defaultdict(list)
    for t in trades:
        by_score[t["score"]].append(t)
    report["by_score"] = {str(k): stats(v) for k, v in sorted(by_score.items())}

    # The actual source for CONFIDENCE_CALIBRATION in picks.py: win rate per
    # (category, score), backtested for every category regardless of whether
    # it was the signal's natural one — this is what makes every horizon on
    # the search card "accurate" instead of an untested extrapolation.
    by_category_and_score = defaultdict(lambda: defaultdict(list))
    for t in trades:
        by_category_and_score[t["category"]][t["score"]].append(t)
    report["by_category_and_score"] = {
        cat: {str(score): stats(v) for score, v in sorted(scores.items())}
        for cat, scores in by_category_and_score.items()
    }

    # For each signal: of the trades where it fired *in agreement* with the
    # overall call, what was the win rate? Answers "does this signal help?"
    by_component = {}
    for name in ["macd", "ema", "pattern"]:
        agreed = [
            t for t in trades
            if t["components"].get(name) == ("bull" if t["signal"] == "BUY" else "bear")
        ]
        by_component[name] = stats(agreed)
    report["by_signal_when_it_agreed_with_the_call"] = by_component

    return report


def walk_forward_report(trades):
    """Splits trades chronologically in half — calibrate-fold (older) and
    validate-fold (newer) — and compares win-rate-by-score in each.

    This is the honest check the single full-period backtest can't do on its
    own: CONFIDENCE_CALIBRATION in picks.py was fit on the *entire* 2-year
    period and then evaluated against that same period, which is in-sample
    and can look better than it should. If the by-score win-rate pattern
    from the calibrate-fold doesn't hold up in the validate-fold, the
    calibration was likely fitting noise, not a real relationship.

    This is a single chronological train/test split, not a many-fold rolling
    walk-forward — with only 2 years of history, more folds would each be too
    small to read much into.
    """
    dated = sorted(trades, key=lambda t: t["date"])
    if not dated:
        return {"calibrate_fold": {}, "validate_fold": {}}

    midpoint = len(dated) // 2
    calibrate_fold, validate_fold = dated[:midpoint], dated[midpoint:]

    def by_score(subset):
        buckets = defaultdict(list)
        for t in subset:
            buckets[t["score"]].append(t)
        return {str(k): stats(v) for k, v in sorted(buckets.items())}

    def by_category_and_score(subset):
        buckets = defaultdict(lambda: defaultdict(list))
        for t in subset:
            buckets[t["category"]][t["score"]].append(t)
        return {
            cat: {str(score): stats(v) for score, v in sorted(scores.items())}
            for cat, scores in buckets.items()
        }

    return {
        "calibrate_fold": {
            "date_range": [calibrate_fold[0]["date"], calibrate_fold[-1]["date"]],
            "by_score": by_score(calibrate_fold),
            "by_category_and_score": by_category_and_score(calibrate_fold),
        },
        "validate_fold": {
            "date_range": [validate_fold[0]["date"], validate_fold[-1]["date"]],
            "by_score": by_score(validate_fold),
            "by_category_and_score": by_category_and_score(validate_fold),
        },
    }


if __name__ == "__main__":
    tickers = UNIVERSE + CRYPTO_UNIVERSE

    # `python3 backtest.py intraday` calibrates the one_hour category on
    # 5-minute candles — Yahoo caps intraday retention at ~60 days, so this
    # is a much smaller/noisier sample than the 2-year daily default below.
    if len(sys.argv) > 1 and sys.argv[1] == "intraday":
        period, interval, results_path = "60d", "5m", "backtest_results_intraday.json"
    else:
        period = sys.argv[1] if len(sys.argv) > 1 else "2y"
        interval, results_path = "1d", "backtest_results.json"

    trades = run_backtest(tickers, period=period, interval=interval)
    report = summarize(trades)
    walk_forward = walk_forward_report(trades)

    print(json.dumps(report, indent=2))
    print(json.dumps({"walk_forward": walk_forward}, indent=2))
    with open(results_path, "w") as f:
        json.dump({"report": report, "walk_forward": walk_forward, "trades": trades}, f, indent=2)
    print(f"\n{len(trades)} total trades simulated across {len(tickers)} tickers. Full results in {results_path}", file=sys.stderr)
