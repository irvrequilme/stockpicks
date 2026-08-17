"""
Simulates an actual capital-constrained account trading the backtest.py
signals, instead of scoring every trade as an independent, unlimited-capital
position. backtest.py answers "does each signal individually beat a coin
flip"; this answers "if you actually traded this with one account and a
position-size limit, what would your equity curve have looked like."

Sizing is risk-based (R-multiples): each position risks a fixed % of
*current* equity (the distance from entry to stop), not a fixed share count.
A trade's dollar P&L = equity_at_entry * risk_per_trade * (return_pct/100 /
risk_pct) — i.e. its outcome expressed as a multiple of what was risked.

Known limitations:
- Trades are opened/closed only at the daily granularity backtest.py
  simulates at — no intra-day fills, no partial fills.
- If more signals fire than there are open slots, the earliest-dated ones
  win; skipped trades are recorded but don't affect equity.
- No transaction costs, spread, or slippage (same caveat as backtest.py).
- Position sizing assumes stop-losses fill exactly at the stop price,
  which real markets don't guarantee (gaps, slippage).
"""
import json
import sys

from backtest import run_backtest
from universe import UNIVERSE
from crypto_universe import CRYPTO_UNIVERSE


def simulate_portfolio(trades, starting_capital=10000, risk_per_trade=0.01, max_positions=20):
    decided = [t for t in trades if t["outcome"] in ("win", "loss", "timeout") and t.get("risk_pct")]
    decided.sort(key=lambda t: t["date"])

    equity = starting_capital
    open_positions = []  # list of {exit_date, pnl} scheduled to close
    equity_curve = [{"date": decided[0]["date"] if decided else None, "equity": equity}]
    skipped = 0
    taken = 0

    def settle_positions_up_to(date):
        nonlocal equity
        still_open = []
        to_close = []
        for pos in open_positions:
            if pos["exit_date"] <= date:
                to_close.append(pos)
            else:
                still_open.append(pos)
        to_close.sort(key=lambda p: p["exit_date"])
        for pos in to_close:
            equity += pos["pnl"]
            equity_curve.append({"date": pos["exit_date"], "equity": round(equity, 2)})
        return still_open

    for t in decided:
        open_positions = settle_positions_up_to(t["date"])

        if len(open_positions) >= max_positions:
            skipped += 1
            continue

        r_multiple = (t["return_pct"] / 100) / t["risk_pct"] if t["risk_pct"] else 0
        pnl = equity * risk_per_trade * r_multiple
        open_positions.append({"exit_date": t["exit_date"], "pnl": pnl})
        taken += 1

    # settle anything still open at the end
    for pos in sorted(open_positions, key=lambda p: p["exit_date"]):
        equity += pos["pnl"]
        equity_curve.append({"date": pos["exit_date"], "equity": round(equity, 2)})

    peak = starting_capital
    max_drawdown_pct = 0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        drawdown = (peak - point["equity"]) / peak * 100 if peak else 0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)

    total_return_pct = (equity - starting_capital) / starting_capital * 100

    return {
        "starting_capital": starting_capital,
        "ending_capital": round(equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "trades_taken": taken,
        "trades_skipped_capacity": skipped,
        "risk_per_trade_pct": risk_per_trade * 100,
        "max_concurrent_positions": max_positions,
        "equity_curve_points": len(equity_curve),
    }


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "2y"
    tickers = UNIVERSE + CRYPTO_UNIVERSE

    trades = run_backtest(tickers, period=period)
    result = simulate_portfolio(trades)

    print(json.dumps(result, indent=2))
    print(
        f"\n{result['trades_taken']} trades taken, {result['trades_skipped_capacity']} skipped "
        f"(no open slot) out of {len(trades)} total signals.",
        file=sys.stderr,
    )
