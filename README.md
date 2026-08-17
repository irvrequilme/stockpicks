# Stock Predictor Web

A personal dashboard that scans US stocks and crypto for technical-analysis
signals (RSI-filtered MACD, EMA crossovers, chart patterns), and — unlike
most tools like this — actually backtests its own rules and shows you
whether they work, instead of just asserting confidence.

**Not financial advice.** The honest headline result: a 2-year/99-ticker
backtest of this exact rule set found no clear trading edge (see the
dashboard's footer for the full breakdown, or `backend/backtest.py`).

## Scope

This is a **single-user personal tool**, not a multi-tenant service:

- No authentication — anyone who can reach the server can use it.
- Runs as a Uvicorn dev server (`--reload`). Fine for personal/local use;
  not a production ASGI setup (no process manager, no TLS termination).
- Data comes entirely from `yfinance`, an unofficial wrapper around Yahoo
  Finance's internal API. It has no SLA and can break or rate-limit without
  warning (this has happened during development — see error handling in
  `main.py`).

If you want to expose this beyond your own machine, you'd need to add
authentication and run it behind a proper production ASGI stack (e.g.
gunicorn + uvicorn workers behind a reverse proxy with TLS).

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `http://localhost:8000`.

## Running the tests

```bash
cd backend
source venv/bin/activate
python3 -m pytest tests/ -v
```

Tests cover the pure indicator math (RSI/MACD/EMA/Bollinger/ATR) and the
`evaluate()` scoring rule on synthetic price series — all without hitting
the network. There's no test coverage for the live Yahoo Finance
integration itself; that's exercised by actually running the app.

## Architecture

**Backend** (`backend/`, FastAPI):
- `picks.py` — the scoring engine. `evaluate()` is the single source of
  truth for the rules; it's indexed at an arbitrary bar so the exact same
  logic runs live and inside the backtester, so they can never silently
  drift apart. Results are cached in-memory for 45s to avoid hammering
  Yahoo Finance on repeated refreshes. `horizon_variants()` sizes
  entry/target/stop for the three daily-candle horizons (Day Trade/1
  Week/Swing Trade) from a signal's current volatility, sqrt(time)-scaling
  the move for whichever ones aren't the ticker's risk-derived natural
  category — the search card lets you switch between all three. A fourth
  horizon, 1 Hour, is a genuinely separate analysis on 5-minute candles
  (fetched on demand, not scaled from the daily signal), sized from
  `atr_pct()` — actual recent 5-minute-bar price range — rather than the
  daily categories' Bollinger-based move. That distinction mattered: the
  first version of 1 Hour reused the daily move formula and backtested at
  an 85% timeout rate / 18.8% win rate (most setups never resolved within
  the hour); switching to ATR-based sizing cut the timeout rate to 40% and
  raised the win rate to 26.3%, with the walk-forward split now stable to
  within ~1 point per score (see `CONFIDENCE_CALIBRATION`'s comment) — a
  real, verified improvement, not just re-badged numbers. Still the least
  reliable of the four horizons, and the dashboard says so directly on the
  1 Hour tab rather than hiding it. The same ATR-scaled-by-sqrt(bars)
  approach was then tried on the three daily categories too, on the theory
  that it should generalize — and re-backtested, honestly, rather than
  assumed. It made all three worse (28.4/29.1/30.2% -> 25.0/25.3/26.1% win
  rate), so it was reverted for those and kept only where it measurably
  helped (`move_for_category()`'s docstring has the full numbers).
- `patterns.py` — chart pattern detection (double top/bottom, head &
  shoulders, triangles) via peak/trough analysis, scaled per candle interval.
- `backtest.py` / `portfolio_sim.py` — offline analysis scripts (not called
  by the running app) used to validate the rules in `picks.py` and produce
  the numbers shown in the dashboard's footer. `python3 backtest.py 2y`
  tests every historical signal against **all three daily** horizons via
  the same `horizon_variants()` sizing the live app uses — not just each
  signal's natural risk-derived category — so `CONFIDENCE_CALIBRATION`
  (per category, per score) reflects a real backtested Win Rate for
  whichever horizon you pick on the search card, not an untested
  extrapolation. `python3 backtest.py intraday` runs the separate 5-minute-
  candle backtest behind the 1 Hour horizon, limited to Yahoo's ~60-day
  intraday retention window (vs. 2 years for the daily categories) — a much
  smaller, noisier sample, and results are correspondingly less reliable.
- `news_signal.py` / `fundamentals_signal.py` — live-only informational
  context (news sentiment, analyst rating). Explicitly NOT backtested or
  folded into the score — Yahoo only exposes current snapshots, not
  point-in-time history, so using them retroactively would be look-ahead
  bias.

**Frontend** (`frontend/index.html`): a single self-contained HTML file, no
build step — the FastAPI backend serves it directly.

## Known limitations

See the dashboard's own footer for the full backtest disclosure. In short:
this system has not been shown to beat a coin flip after accounting for its
2:1 reward:risk sizing, and a portfolio simulation showed wildly unstable
results depending on unrelated position-sizing parameters — itself evidence
of no real edge. Treat every signal as "what the rules currently say," not
investment advice.
