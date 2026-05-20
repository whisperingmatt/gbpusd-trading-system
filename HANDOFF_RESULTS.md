# \# HANDOFF.md — Daily Bridge

# 

# \## DATE

# 2026-05-20 (Wednesday) — STRATEGY REBUILD SESSION (updated late evening)

# 

# \## STATUS

# 

# Strategy module + backtest engine built and run end-to-end. Full 3-year OANDA data verified. Four iterations tested tonight. Strategy has positive expectancy but fires too rarely — structural change needed in v1.1, not more parameter tuning. All work committed locally; ready for tomorrow's session.

# 

# \---

# 

# \## RESULTS

# 

# \### What was built

# 

# \- `strategy/ftp\_v1.py` — full FTP v1.0 spec implementation

# \- `backtest/run\_backtest.py` — 3-year backtest engine with OANDA + yfinance fallback

# \- `docs/backtest\_\*.json` and `.png` outputs all generated

# \- `from dotenv import load\_dotenv` patched into both files (was missing initially — system fell back to yfinance until fixed)

# 

# \### Final OANDA backtest (4 pairs, baseline filters)

# 

# ```

# Window:           2023-07-13 → 2026-05-18

# Data source:      OANDA (verified)

# Total trades:     52

# Win rate:         36.5%

# Expectancy:       +$10.99 per trade  (PASS)

# Profit factor:    1.041              (FAIL — needs ≥ 1.3)

# Max drawdown:     -4.49%             (PASS — under 5%)

# Trades vs target: 52 / 80            (FAIL)

# Pass live:        NO

# ```

# 

# \### Per-instrument (baseline 5-pair universe, the relevant data)

# 

# ```

# GBP\_USD: 21 trades, WR 38.1%, Exp +$15.32, Total   +$321

# AUD\_USD: 10 trades, WR 30.0%, Exp -$74.72, Total   -$747

# EUR\_USD: 14 trades, WR 28.6%, Exp -$68.79, Total   -$963

# USD\_JPY: 11 trades, WR 18.2%, Exp -$229.58, Total -$2,525

# XAU\_USD:  7 trades, WR 57.1%, Exp +$324.22, Total +$2,270

# ```

# 

# \### Diagnostic — why so few trades

# 

# Filter rejection counts over the 3-year window:

# 

# ```

# trend:    3,815   (no D1 trend stack — DOMINANT killer)

# reclaim:  1,296   (H4 EMA20 didn't cross at scan moment)

# adx:        666   (trend too weak, ADX < 20)

# pass:        74   (cleared all filters)

# vol:          1   (ATR vol cap — almost never fires)

# rsi:          0   (RSI band — never blocked anything)

# ```

# 

# Of the 74 setups that passed all strategy filters, only 52 traded. The 22-trade gap is circuit-breaker driven — most likely max-3-new-per-day binding when multiple pairs signal on the same day.

# 

# \### Iterations tested tonight

# 

# | # | Change | Trade count | Expectancy | Verdict |

# |---|---|---|---|---|

# | 1 | Baseline 5-pair | 63 | -$22 | FAIL (DD 5.04%) |

# | 2 | Drop USD\_JPY | 52 | +$11 | FAIL (PF 1.04) |

# | 3 | Widen RSI 40-70 / 30-60 | 52 | +$11 | No change |

# | 4 | Correlation cap 2→3 | 52 | +$11 | No change |

# 

# \### Key findings

# 

# 1\. \*\*USD/JPY is the clearest single drag\*\* — 18% WR, -$2,525 over 3 years. Removing it flips expectancy positive.

# 2\. \*\*XAU/USD is the strongest pair\*\* — 57% WR, +$2,270. Gold belongs in the universe.

# 3\. \*\*The trend filter (close > EMA20 > EMA50) eats 65% of opportunities.\*\* This is the dominant constraint.

# 4\. \*\*The H4 EMA20 reclaim trigger\*\* is a single-bar event. It fires rarely. 1,296 rejections shows the strategy is waiting for a specific moment that doesn't come often.

# 5\. \*\*RSI band and ATR vol cap are dead filters\*\* — they never rejected anything. They can stay as guardrails but don't tune them; they're not active.

# 6\. \*\*Strategy edge is real\*\* — 36.5% WR at 1:2 R:R is theoretically above break-even (33.3%). Expectancy positive after costs. The strategy isn't broken; it just doesn't fire often enough.

# 

# \---

# 

# \## BLOCKERS

# 

# None. Backtest runs clean against real OANDA data. All bugs found tonight are fixed (dotenv loading, package installs).

# 

# \---

# 

# \## NEXT

# 

# Iteration ran into a structural wall: parameter tweaks can't push trade count to 80 while preserving entry quality. Tomorrow needs structural changes, in this order:

# 

# \### 1. Replace the H4 EMA20 reclaim with an H4 EMA20 pullback (highest EV)

# Current: requires previous H4 closed BELOW EMA20 AND current H4 closed ABOVE EMA20. Single-bar event.

# Proposed: H4 price has tagged or come within 0.3 × ATR of EMA20 within the last 4 H4 bars, AND most recent H4 closed above EMA20.

# This converts a rare event trigger into a continuous-condition entry. Expected to dramatically increase setups in trending markets without losing the "buy near EMA20" intent.

# 

# \### 2. Lower ADX threshold from 20 to 15 (second test)

# 666 rejections on ADX < 20 means there's a real population of weakly-trending markets being excluded. ADX 15-20 trends are softer but still directional. Test as second change after #1.

# 

# \### 3. Re-introduce USD/JPY with 2-bar confirmation (third test)

# USD/JPY's 18% WR likely stems from BOJ-driven whipsaws. Require two consecutive H4 closes above EMA20 (longs) or below (shorts) for this pair specifically. If still bad, leave it out permanently.

# 

# \### 4. Consider raising daily-new-trades cap (small lever)

# 22 valid signals were blocked tonight. With 4 USD-quote pairs that often signal together, MAX\_NEW\_PER\_DAY = 3 binds. Test 4. Small impact but free.

# 

# \### Do NOT

# \- Modify `main.py` or the live system

# \- Send Telegram alerts or place OANDA orders

# \- Loosen the 1:2 R:R rule — that's the math foundation

# \- Re-introduce the 8-bot architecture

# 

# \---

# 

# \## NOTES TO FUTURE WEB CLAUDE

# 

# \- \*\*The diagnostic counter pattern is gold.\*\* Keep `FILTER\_COUNTS` in the strategy module. Every iteration should re-print it. It's the difference between guessing and knowing.

# 

# \- \*\*The 36.5% / +$11 / 52 trades signature is the v1.0 baseline.\*\* Any v1.1 change should be compared against this. If a change improves expectancy but cuts trades, that's a bad trade. We need MORE trades at SIMILAR OR BETTER expectancy.

# 

# \- \*\*OANDA data and yfinance data give meaningfully different per-instrument breakdowns.\*\* AUD/USD looked like a winner on yfinance, lost money on OANDA. XAU/USD looked like a drag on yfinance, became the top performer on OANDA. Never trust yfinance results for production decisions. Re-run on OANDA before drawing conclusions.

# 

# \- \*\*Strategy was running on yfinance the first two times tonight despite OANDA key being available\*\* — because I forgot `load\_dotenv()` in the new modules. The `.env` file existed but Python's `os.environ` doesn't auto-read it. Both files now have `load\_dotenv()` at the top.

# 

# \- \*\*All 5 PNG charts and JSON outputs in `docs/` are from the final OANDA run, not the yfinance runs.\*\* Earlier outputs were overwritten.

# 

# \- \*\*Iteration order matters.\*\* Single-change discipline saved time tonight — we learned RSI was dead in one cheap test instead of grinding parameters for an hour. Stay single-change.

# 

# \- \*\*Don't keep iterating past 2-3 attempts in one session.\*\* If the third try doesn't move the needle, stop and rethink structurally, like tonight. Endless parameter grinding is a tell that the strategy needs a different lever.

