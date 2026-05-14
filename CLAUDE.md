# Trading System — Claude Code Guide

## Project Overview

Regime-adaptive multi-pair forex signal system. Two files:

- **`main.py`** — Live signal engine. Runs daily at 17:00 EST, fetches OANDA candles, detects regime, routes to the right strategy per pair, emails a signal report via SendGrid.
- **`backtest.py`** — Historical backtester. Fetches OANDA daily + 4H data, simulates Strategy A and B signals across all four pairs, outputs `backtest_results.csv` and `equity_curve.html`.

## Pairs & Strategy Routing

| Pair    | Config key  | Default strategy | Notes |
|---------|-------------|-----------------|-------|
| GBP/USD | `GBP_USD`   | `trend`         | Momentum pair; stays in Strategy A regardless of regime |
| EUR/USD | `EUR_USD`   | `auto`          | Range primary; trend only if ADX > 28 |
| AUD/USD | `AUD_USD`   | `auto`          | Gold REQUIRED for longs; risk-on filter active |
| USD/JPY | `USD_JPY`   | `trend`         | 75% base size; intervention zones 143/153 |

## Strategy A — Trend Following

- ADX above pair `adx_trend_min` with clear EMA50/EMA200 structure
- Price within `atr_entry_band × ATR` of the 50 EMA (pullback entry)
- RSI in pair-specific `rsi_buy_lo`/`rsi_buy_hi` range
- Previous candle confirms direction
- 4H confirmation: 2 of 3 checks (close vs EMA50, EMA50 vs EMA200, RSI)
- Stop: `atr_stop × ATR`; Target: `atr_target × ATR`

## Strategy B — Mean Reversion

- ADX below pair `adx_range_max`
- Price within 1 ATR of 30/35/40-candle range high or low
- RSI at extreme (`rsi_oversold` / `rsi_overbought`)
- Reversal candle (close vs open)
- 4H confirmation uses **opposite direction** — BUY signal checks SELL-side 4H conditions (confirms the bearish pressure that created the low). This is intentional and critical; using same-direction 4H confirmation blocks all Strategy B signals for range-bound pairs.
- Target: range midpoint (`range_target` fraction)

## Key Pair-Specific Settings (`PAIR_CONFIGS` in main.py)

Each pair has its own `atr_target`, ADX thresholds, RSI bands, commodity rules, etc. Edit these in `PAIR_CONFIGS` — do **not** add global overrides.

GBP/USD `atr_target` is **2.0** (changed from 3.0 — raises win rate from ~50% to ~81% in backtest).

## Backtest Constants (backtest.py)

`GBP_ATR_TARGET = 2.0` applies only to GBP/USD via `target_mult` parameter in `simulate()`. All other pairs use the global `ATR_TARGET_MULT = 3.0`.

## Environment Variables (.env)

```
OANDA_API_KEY=
OANDA_ENV=practice          # or live
SENDGRID_API_KEY=
EMAIL_SENDER=
EMAIL_RECIPIENT=
FINNHUB_API_KEY=
```

## Running

```bash
pip install -r requirements.txt

# Backtest
python backtest.py          # outputs backtest_results.csv + equity_curve.html

# Live signals (runs once then schedules 17:00 EST daily)
python main.py
```

## Central Bank Rates

Update `RATES` dict in both files after each central bank meeting. Current rates reflect May 2026 settings.

## Project File Organization

All project files are maintained at:
`C:\Users\Matthew\OneDrive\Attachments\Desktop\Claude\projects\Trading Systems\`

| Folder | Contents |
|--------|----------|
| `01 - Planning` | CLAUDE.md and planning docs |
| `02 - Strategy` | Strategy documentation |
| `03 - Research` | Research notes |
| `04 - Code` | main.py, backtest.py |
| `05 - Backtests` | Backtest run archives |
| `06 - Results` | backtest_results.csv, equity_curve.html |
| `07 - Bots` | Multi-bot framework (see below) |

## Multi-Bot Framework (`07 - Bots`)

Eight specialist bots planned for signal generation and review:

| Bot | Role |
|-----|------|
| Bot 1 - Technical | Technical analysis signals |
| Bot 2 - Macro | Macro / fundamental context |
| Bot 3 - Sentiment | Market sentiment analysis |
| Bot 4 - Risk | Risk sizing and exposure limits |
| Bot 5 - Debate | Challenges signal consensus |
| Bot 6 - Strategy | Final strategy routing decision |
| Bot 7 - Memory | Tracks trade history and regime context |
| Bot 8 - Audit | Post-trade review and performance audit |
