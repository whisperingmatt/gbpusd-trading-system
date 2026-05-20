"""
backtest/run_backtest.py — 3-year backtest of FTP v1.0 strategy

Runs daily scan, places pending limit orders, tracks fills + stops + targets,
honours circuit breakers, applies realistic spread + slippage, compounds equity.

Data source:
    - If OANDA_API_KEY env var is set: pulls real OANDA candles (full 3 years).
    - Else: falls back to yfinance (D1 = 3yr, H4 resampled from H1 = ~2yr).

Outputs:
    docs/backtest_trades.json
    docs/backtest_summary.json
    docs/backtest_equity.png
    docs/backtest_drawdown.png
    docs/backtest_distribution.png

Run: python -m backtest.run_backtest
"""

from __future__ import annotations

import os
import sys
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy.ftp_v1 import (
    INSTRUMENTS, PIP_SIZE, TYPICAL_SPREAD,
    RISK_PER_TRADE_PCT, RISK_HALVED_PCT, STOP_ATR_MULT, REWARD_MULT,
    ADX_THRESHOLD, ATR_VOL_CAP, RSI_LONG_BAND, RSI_SHORT_BAND, TIME_EXIT_DAYS,
    FILTER_COUNTS,
    compute_indicators, check_long_entry, check_short_entry,
    size_position, usd_pnl,
)


SLIPPAGE_PIPS = 0.5
STARTING_EQUITY = 100_000.0
WARMUP_BARS = 60
DAILY_LOSS_HALT_PCT = 0.02     # halt new entries if today's PnL <= -2%
DD_HALVE_PCT        = 0.04     # halve risk when DD >= 4% from HWM
DD_RESTORE_PCT      = 0.02     # restore risk when within 2% of HWM
MAX_OPEN_POSITIONS  = 5
MAX_NEW_PER_DAY     = 3
MAX_SAME_USD_DIR    = 3


# ----------------------------------------------------------------------------
# Data layer — OANDA primary, yfinance fallback
# ----------------------------------------------------------------------------

YF_SYMBOLS = {
    "GBP_USD": "GBPUSD=X",
    "EUR_USD": "EURUSD=X",
    "AUD_USD": "AUDUSD=X",
    "USD_JPY": "JPY=X",
    "XAU_USD": "GC=F",          # gold futures — closest free proxy
}


def _have_oanda() -> bool:
    return bool(os.environ.get("OANDA_API_KEY"))


def _fetch_oanda(pair: str, granularity: str, years: int) -> pd.DataFrame:
    """OANDA paginated fetch (max 5000 candles per call)."""
    import requests
    env = os.environ.get("OANDA_ENV", "practice").lower()
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    headers = {"Authorization": f"Bearer {os.environ['OANDA_API_KEY']}"}

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 30)

    all_rows = []
    cursor = start_dt
    while cursor < end_dt:
        url = f"https://{host}/v3/instruments/{pair}/candles"
        params = {
            "granularity": granularity,
            "from":        cursor.isoformat().replace("+00:00", "Z"),
            "count":       5000,
            "price":       "M",
            "dailyAlignment": 17,
            "alignmentTimezone": "America/New_York",
        }
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        candles = r.json().get("candles", [])
        if not candles:
            break
        for c in candles:
            if not c.get("complete"):
                continue
            m = c["mid"]
            all_rows.append({
                "time":   pd.Timestamp(c["time"]).tz_convert("UTC"),
                "open":   float(m["o"]),
                "high":   float(m["h"]),
                "low":    float(m["l"]),
                "close":  float(m["c"]),
                "volume": int(c["volume"]),
            })
        last_t = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")
        if last_t <= cursor:
            break
        cursor = last_t + timedelta(seconds=1)

    df = pd.DataFrame(all_rows).drop_duplicates("time").set_index("time").sort_index()
    return df


def _fetch_yf(pair: str, granularity: str, years: int) -> pd.DataFrame:
    """yfinance fallback. granularity in {'D','H4'}."""
    import yfinance as yf
    sym = YF_SYMBOLS[pair]
    end_dt = datetime.now()

    if granularity == "D":
        start_dt = end_dt - timedelta(days=int(years * 365.25) + 30)
        df = yf.download(sym, start=start_dt, end=end_dt, interval="1d",
                         progress=False, auto_adjust=False)
    elif granularity == "H4":
        # yfinance caps 1h at ~730 days; pull H1 and resample
        start_dt = end_dt - timedelta(days=720)
        df = yf.download(sym, start=start_dt, end=end_dt, interval="1h",
                         progress=False, auto_adjust=False)
    else:
        raise ValueError(granularity)

    if df.empty:
        return df

    # yfinance sometimes returns MultiIndex columns; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })[["open", "high", "low", "close", "volume"]]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "time"

    if granularity == "H4":
        # Resample to 4H bars
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        df = df.resample("4h", label="right", closed="right").agg(agg).dropna()

    return df


def fetch_history(pair: str, granularity: str, years: int = 3) -> pd.DataFrame:
    if _have_oanda():
        return _fetch_oanda(pair, granularity, years)
    return _fetch_yf(pair, granularity, years)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def usd_direction(pair: str, direction: str) -> str:
    """Anti-dollar (USD_SHORT) vs pro-dollar (USD_LONG)."""
    if pair == "USD_JPY":
        return "USD_LONG" if direction == "BUY" else "USD_SHORT"
    return "USD_SHORT" if direction == "BUY" else "USD_LONG"


def half_spread(pair: str) -> float:
    return TYPICAL_SPREAD[pair] / 2.0


def slip(pair: str) -> float:
    return SLIPPAGE_PIPS * PIP_SIZE[pair]


def is_friday_after_noon_est(ts: pd.Timestamp) -> bool:
    """Spec excludes new entries Friday after 12:00 EST (~17:00 UTC standard, 16:00 UTC summer)."""
    est_offset = -5  # treat conservatively as standard; entries blocked from 17:00 UTC Fri
    if ts.weekday() != 4:
        return False
    return ts.hour >= 17


# ----------------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------------

def run_backtest(verbose: bool = True):
    print(f"Data source: {'OANDA' if _have_oanda() else 'yfinance fallback'}")
    print("Fetching candles for 5 instruments...")

    data = {}
    for pair in INSTRUMENTS:
        d1 = fetch_history(pair, "D",  years=3)
        h4 = fetch_history(pair, "H4", years=3)
        if len(d1) == 0 or len(h4) == 0:
            print(f"  WARN {pair}: empty data (D1={len(d1)} H4={len(h4)})")
            continue
        d1 = compute_indicators(d1)
        h4 = compute_indicators(h4)
        data[pair] = {"d1": d1, "h4": h4}
        print(f"  {pair}: D1 {len(d1)} bars [{d1.index[0].date()}..{d1.index[-1].date()}]"
              f"  H4 {len(h4)} bars [{h4.index[0].date()}..{h4.index[-1].date()}]")

    if not data:
        print("FATAL: no data fetched")
        return

    # Find common date range across instruments (start where every instrument
    # has both D1 + H4 warmup complete)
    starts = []
    for pair, v in data.items():
        d1_start = v["d1"].iloc[WARMUP_BARS]["close"] if len(v["d1"]) > WARMUP_BARS else None
        h4_start = v["h4"].iloc[WARMUP_BARS]["close"] if len(v["h4"]) > WARMUP_BARS else None
        if d1_start is None or h4_start is None:
            continue
        starts.append(max(v["d1"].index[WARMUP_BARS], v["h4"].index[WARMUP_BARS]))
    if not starts:
        print("FATAL: insufficient warmup data")
        return
    bt_start = max(starts)
    bt_end = min(v["d1"].index[-1] for v in data.values())
    print(f"\nBacktest window: {bt_start.date()} -> {bt_end.date()}"
          f"  ({(bt_end - bt_start).days} calendar days)\n")

    # Build daily timeline (use first instrument's D1 index, filtered)
    timeline = sorted(set().union(*[set(v["d1"].index) for v in data.values()]))
    timeline = [t for t in timeline if bt_start <= t <= bt_end]

    # State
    equity = STARTING_EQUITY
    hwm    = STARTING_EQUITY
    risk_pct = RISK_PER_TRADE_PCT
    open_positions = []     # list of dicts
    _store_closed.clear()   # _process_exits appends here; we read from it after
    equity_curve   = []     # (timestamp, equity)
    daily_pnl      = {}     # date -> realized USD

    # ------------------------------------------------------------------------
    # Main loop — bar by bar at D1 close
    # ------------------------------------------------------------------------
    for today in timeline:
        # 1. Process exits for open positions on bars up to today's close
        open_positions, equity, day_realized = _process_exits(
            open_positions, equity, data, today
        )
        date_key = today.date()
        daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + day_realized

        # 2. Drawdown circuit
        if equity > hwm:
            hwm = equity
        dd_pct = (hwm - equity) / hwm if hwm > 0 else 0
        if dd_pct >= DD_HALVE_PCT:
            risk_pct = RISK_HALVED_PCT
        elif dd_pct <= DD_RESTORE_PCT:
            risk_pct = RISK_PER_TRADE_PCT

        equity_curve.append((today, equity))

        # 3. Daily loss circuit
        if daily_pnl[date_key] <= -DAILY_LOSS_HALT_PCT * equity:
            continue

        # 4. Friday-after-noon-EST block (applies if today's close is Friday)
        if is_friday_after_noon_est(today):
            continue

        # 5. Scan for new setups
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        candidates = []
        for pair in INSTRUMENTS:
            if pair not in data:
                continue
            d1 = data[pair]["d1"].loc[:today]
            h4 = data[pair]["h4"].loc[:today]
            if len(d1) < WARMUP_BARS or len(h4) < WARMUP_BARS:
                continue
            setup = check_long_entry(d1, h4) or check_short_entry(d1, h4)
            if setup:
                setup["pair"] = pair
                candidates.append(setup)

        # 6. Apply correlation cap + max new + open-position cap
        new_today = 0
        dir_counts = _count_usd_dirs(open_positions)
        for setup in candidates:
            if new_today >= MAX_NEW_PER_DAY:
                break
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            pair = setup["pair"]
            ud = usd_direction(pair, setup["direction"])
            if dir_counts.get(ud, 0) >= MAX_SAME_USD_DIR:
                continue
            # Skip if we already have a position open in this pair
            if any(p["pair"] == pair for p in open_positions):
                continue

            # Place pending limit — try to fill in next 24h of H4 bars
            fill = _try_fill_pending(setup, data[pair]["h4"], today)
            if fill is None:
                continue

            units = size_position(
                equity * (risk_pct / RISK_PER_TRADE_PCT),
                setup["stop_distance"], pair, setup["entry"]
            )
            # ^ adjusts risk_amount by risk_pct ratio without changing the sizing fn
            if units <= 0:
                continue

            pos = {
                "pair":           pair,
                "direction":      setup["direction"],
                "entry_signal":   setup["entry"],
                "entry_fill":     fill["fill_price"],
                "entry_time":     fill["fill_time"],
                "stop":           setup["stop"],
                "target":         setup["target"],
                "stop_distance":  setup["stop_distance"],
                "units":          units,
                "risk_usd":       equity * risk_pct,
                "reason":         setup["reason"],
            }
            open_positions.append(pos)
            dir_counts[ud] = dir_counts.get(ud, 0) + 1
            new_today += 1

    # End: force-close anything still open at market on final bar
    for pos in list(open_positions):
        last_h4 = data[pos["pair"]]["h4"]
        last_bar = last_h4.iloc[-1]
        exit_price = float(last_bar["close"])
        pnl = usd_pnl(pos["pair"], pos["direction"], pos["entry_fill"],
                      exit_price, pos["units"])
        # subtract exit costs
        pnl -= _exit_cost(pos["pair"], pos["units"], exit_price)
        equity += pnl
        _store_closed.append({**pos,
                              "exit_time": last_h4.index[-1],
                              "exit_price": exit_price,
                              "exit_reason": "BACKTEST_END",
                              "pnl_usd": pnl})
    open_positions.clear()
    equity_curve.append((timeline[-1], equity))

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------
    summary = _compute_summary(_store_closed, equity_curve, daily_pnl,
                               bt_start, bt_end)
    _write_outputs(_store_closed, summary, equity_curve)
    _print_summary(summary)
    return summary


def _process_exits(open_positions, equity, data, today):
    """
    Walk H4 bars from each position's last-checked time up to today's D1 close.
    Close positions on stop, target, or time exit. Returns updated list + equity.
    """
    still_open = []
    day_realized = 0.0
    for pos in open_positions:
        h4 = data[pos["pair"]]["h4"]
        # Bars strictly after entry, up to and including today's close
        entry_time = pos["entry_time"]
        bars = h4.loc[(h4.index > entry_time) & (h4.index <= today)]
        if len(bars) == 0:
            still_open.append(pos)
            continue

        closed = False
        for ts, bar in bars.iterrows():
            # Time exit
            age_days = (ts - entry_time).total_seconds() / 86400
            if age_days >= TIME_EXIT_DAYS:
                exit_price = float(bar["open"])
                pnl = usd_pnl(pos["pair"], pos["direction"], pos["entry_fill"],
                              exit_price, pos["units"])
                pnl -= _exit_cost(pos["pair"], pos["units"], exit_price)
                equity += pnl
                day_realized += pnl
                closed_trade = {**pos, "exit_time": ts, "exit_price": exit_price,
                                "exit_reason": "TIME", "pnl_usd": pnl}
                _store_closed.append(closed_trade)
                closed = True
                break

            hi, lo = float(bar["high"]), float(bar["low"])

            if pos["direction"] == "BUY":
                stop_hit   = lo <= pos["stop"]
                target_hit = hi >= pos["target"]
                # Conservative: if both inside same bar, assume stop first
                if stop_hit and target_hit:
                    exit_price = pos["stop"]
                    reason = "STOP"
                elif stop_hit:
                    exit_price = pos["stop"]
                    reason = "STOP"
                elif target_hit:
                    exit_price = pos["target"]
                    reason = "TARGET"
                else:
                    continue
            else:  # SELL
                stop_hit   = hi >= pos["stop"]
                target_hit = lo <= pos["target"]
                if stop_hit and target_hit:
                    exit_price = pos["stop"]
                    reason = "STOP"
                elif stop_hit:
                    exit_price = pos["stop"]
                    reason = "STOP"
                elif target_hit:
                    exit_price = pos["target"]
                    reason = "TARGET"
                else:
                    continue

            pnl = usd_pnl(pos["pair"], pos["direction"], pos["entry_fill"],
                          exit_price, pos["units"])
            pnl -= _exit_cost(pos["pair"], pos["units"], exit_price)
            equity += pnl
            day_realized += pnl
            closed_trade = {**pos, "exit_time": ts, "exit_price": exit_price,
                            "exit_reason": reason, "pnl_usd": pnl}
            _store_closed.append(closed_trade)
            closed = True
            break

        if not closed:
            still_open.append(pos)

    return still_open, equity, day_realized


def _exit_cost(pair: str, units: int, exit_price: float) -> float:
    """Half-spread + slippage on exit, converted to USD."""
    cost_quote = (half_spread(pair) + slip(pair)) * units
    if pair == "USD_JPY":
        return cost_quote / exit_price
    return cost_quote


def _try_fill_pending(setup, h4, signal_time):
    """
    Limit order placed at H4 EMA20. Check next 6 H4 bars (~24h) for fill.
    Apply 0.5 pip slippage in adverse direction + half-spread.
    """
    pair = setup["pair"]
    bars = h4.loc[h4.index > signal_time].head(6)
    if len(bars) == 0:
        return None

    limit = setup["entry"]
    for ts, bar in bars.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if setup["direction"] == "BUY":
            if lo <= limit <= hi:
                # Fill at limit + slippage + half-spread (worse than mid)
                fill = limit + slip(pair) + half_spread(pair)
                return {"fill_price": fill, "fill_time": ts}
        else:
            if lo <= limit <= hi:
                fill = limit - slip(pair) - half_spread(pair)
                return {"fill_price": fill, "fill_time": ts}
    return None


def _count_usd_dirs(open_positions):
    out = {}
    for p in open_positions:
        ud = usd_direction(p["pair"], p["direction"])
        out[ud] = out.get(ud, 0) + 1
    return out


# Module-level store filled by _process_exits (avoids passing around)
_store_closed = []


# ----------------------------------------------------------------------------
# Summary stats + outputs
# ----------------------------------------------------------------------------

def _compute_summary(closed_trades, equity_curve, daily_pnl, bt_start, bt_end):
    if not closed_trades:
        return {"error": "no trades"}

    df = pd.DataFrame(closed_trades)
    wins   = df[df["pnl_usd"] > 0]
    losses = df[df["pnl_usd"] <= 0]
    total  = len(df)

    avg_win  = wins["pnl_usd"].mean()  if len(wins)   else 0.0
    avg_loss = losses["pnl_usd"].mean() if len(losses) else 0.0
    win_rate = len(wins) / total if total else 0.0
    expectancy = df["pnl_usd"].mean() if total else 0.0
    gross_profit = wins["pnl_usd"].sum()  if len(wins)   else 0.0
    gross_loss   = -losses["pnl_usd"].sum() if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity-curve based DD
    eq = pd.Series([e for _, e in equity_curve],
                   index=pd.DatetimeIndex([t for t, _ in equity_curve]))
    eq = eq.sort_index()
    running_max = eq.cummax()
    dd_usd = eq - running_max
    dd_pct = dd_usd / running_max
    max_dd_usd = float(dd_usd.min())
    max_dd_pct = float(dd_pct.min())

    # Per-instrument
    per_inst = {}
    for pair, sub in df.groupby("pair"):
        sub_wins = sub[sub["pnl_usd"] > 0]
        per_inst[pair] = {
            "trades":     int(len(sub)),
            "win_rate":   float(len(sub_wins) / len(sub)) if len(sub) else 0.0,
            "expectancy": float(sub["pnl_usd"].mean()),
            "total_pnl":  float(sub["pnl_usd"].sum()),
        }

    # Monthly P&L
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["exit_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_usd"].sum().to_dict()
    monthly_str = {str(k): float(v) for k, v in monthly.items()}
    months_positive = sum(1 for v in monthly_str.values() if v > 0)

    # Sharpe-like — daily returns
    daily_series = pd.Series(daily_pnl).sort_index()
    daily_returns = daily_series / STARTING_EQUITY
    sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252)) \
             if daily_returns.std() > 0 else 0.0

    # Consistency rule check — single-day max as % of total profit
    total_profit = float(df["pnl_usd"].sum())
    if total_profit > 0:
        single_day = pd.Series(daily_pnl).max()
        single_day_pct = float(single_day / total_profit)
    else:
        single_day_pct = 0.0

    final_equity = float(eq.iloc[-1])

    summary = {
        "data_source":          "OANDA" if _have_oanda() else "yfinance",
        "backtest_start":       str(bt_start.date()),
        "backtest_end":         str(bt_end.date()),
        "starting_equity":      STARTING_EQUITY,
        "final_equity":         round(final_equity, 2),
        "total_return_pct":     round((final_equity / STARTING_EQUITY - 1) * 100, 2),
        "total_trades":         int(total),
        "wins":                 int(len(wins)),
        "losses":               int(len(losses)),
        "win_rate":             round(win_rate, 4),
        "avg_win_usd":          round(float(avg_win), 2),
        "avg_loss_usd":         round(float(avg_loss), 2),
        "expectancy_usd":       round(float(expectancy), 2),
        "profit_factor":        round(profit_factor, 3) if math.isfinite(profit_factor) else "inf",
        "max_drawdown_usd":     round(max_dd_usd, 2),
        "max_drawdown_pct":     round(max_dd_pct * 100, 2),
        "sharpe_annualised":    round(sharpe, 3),
        "months_in_profit":     int(months_positive),
        "total_months":         int(len(monthly_str)),
        "max_single_day_pct_of_profit": round(single_day_pct * 100, 2),
        "per_instrument":       per_inst,
        "monthly_pnl":          monthly_str,
        "passes_live_criteria": _passes_criteria(profit_factor, max_dd_pct,
                                                  total, single_day_pct, expectancy),
    }
    return summary


def _passes_criteria(profit_factor, max_dd_pct, total_trades,
                     single_day_pct, expectancy):
    fails = []
    if profit_factor < 1.3:
        fails.append(f"profit_factor {profit_factor:.2f} < 1.3")
    if abs(max_dd_pct) > 0.05:
        fails.append(f"max_dd {max_dd_pct*100:.2f}% > 5%")
    if total_trades < 80:
        fails.append(f"trades {total_trades} < 80")
    if single_day_pct > 0.35:
        fails.append(f"single_day {single_day_pct*100:.1f}% > 35%")
    if expectancy <= 0:
        fails.append(f"expectancy ${expectancy:.2f} <= 0")
    return {"pass": len(fails) == 0, "fails": fails}


def _write_outputs(closed_trades, summary, equity_curve):
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # Trade log
    serial = []
    for t in closed_trades:
        serial.append({
            **{k: (v.isoformat() if hasattr(v, "isoformat") else v)
               for k, v in t.items() if k not in ()},
        })
    with open(docs / "backtest_trades.json", "w") as f:
        json.dump(serial, f, indent=2, default=str)

    with open(docs / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Equity curve
    times  = [t for t, _ in equity_curve]
    values = [e for _, e in equity_curve]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, values, linewidth=1.2)
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("FTP v1.0 — Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(docs / "backtest_equity.png", dpi=130)
    plt.close(fig)

    # Drawdown curve
    eq = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()
    dd_pct = (eq - eq.cummax()) / eq.cummax() * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd_pct.index, dd_pct.values, 0, color="red", alpha=0.35)
    ax.set_title("FTP v1.0 — Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(docs / "backtest_drawdown.png", dpi=130)
    plt.close(fig)

    # Distribution histogram
    if closed_trades:
        pnls = [t["pnl_usd"] for t in closed_trades]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(pnls, bins=40, edgecolor="black", alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("FTP v1.0 — Trade P&L Distribution")
        ax.set_xlabel("P&L per trade (USD)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(docs / "backtest_distribution.png", dpi=130)
        plt.close(fig)


def _print_summary(s):
    if "error" in s:
        print(f"\nNO TRADES: {s['error']}")
        return
    print("\n" + "=" * 60)
    print("FTP v1.0 BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Data source:           {s['data_source']}")
    print(f"Window:                {s['backtest_start']} -> {s['backtest_end']}")
    print(f"Starting equity:       ${s['starting_equity']:,.0f}")
    print(f"Final equity:          ${s['final_equity']:,.2f}")
    print(f"Total return:          {s['total_return_pct']:+.2f}%")
    print(f"Total trades:          {s['total_trades']}  (W: {s['wins']}, L: {s['losses']})")
    print(f"Win rate:              {s['win_rate']*100:.1f}%")
    print(f"Avg win / Avg loss:    ${s['avg_win_usd']:.2f}  /  ${s['avg_loss_usd']:.2f}")
    print(f"Expectancy per trade:  ${s['expectancy_usd']:.2f}")
    print(f"Profit factor:         {s['profit_factor']}")
    print(f"Max drawdown:          ${s['max_drawdown_usd']:,.2f}  ({s['max_drawdown_pct']}%)")
    print(f"Sharpe (annualised):   {s['sharpe_annualised']}")
    print(f"Months in profit:      {s['months_in_profit']} / {s['total_months']}")
    print(f"Max single-day %:      {s['max_single_day_pct_of_profit']:.2f}% of total profit")
    print("\nPer instrument:")
    for pair, v in s["per_instrument"].items():
        print(f"  {pair}: {v['trades']:3d} trades, "
              f"WR {v['win_rate']*100:5.1f}%, "
              f"Exp ${v['expectancy']:7.2f}, "
              f"Total ${v['total_pnl']:>10.2f}")
    print("\nPass live-deployment criteria:", "YES" if s["passes_live_criteria"]["pass"] else "NO")
    for f in s["passes_live_criteria"]["fails"]:
        print(f"  - FAIL: {f}")
    print("=" * 60)

    print("\nFilter rejection counts (sorted by most rejections):")
    for name, count in sorted(FILTER_COUNTS.items(), key=lambda x: -x[1]):
        print(f"  {name:>10}: {count}")


if __name__ == "__main__":
    _store_closed.clear()
    summary = run_backtest()
    # write the actual closed trades captured during run
    if _store_closed:
        with open(ROOT / "docs" / "backtest_trades.json", "w") as f:
            json.dump([{k: (v.isoformat() if hasattr(v, "isoformat") else v)
                        for k, v in t.items()} for t in _store_closed],
                      f, indent=2, default=str)
