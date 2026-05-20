"""
strategy/ftp_v1.py — Funded Trading Plus $100k strategy v1.0

Spec: STRATEGY_SPEC.md (2026-05-20)
Pure signal generation. No live execution, no Telegram, no order placement.

Public functions:
    fetch_candles(pair, granularity, count)   -> pandas.DataFrame
    compute_indicators(df)                    -> pandas.DataFrame
    check_long_entry(d1_df, h4_df)            -> dict | None
    check_short_entry(d1_df, h4_df)           -> dict | None
    size_position(equity, stop_distance, instrument, entry_price=None) -> int
    scan_all_instruments(equity)              -> list[dict]
"""

from __future__ import annotations

import os
import math
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

INSTRUMENTS = ["GBP_USD", "EUR_USD", "AUD_USD", "XAU_USD"]

# Risk parameters from spec
RISK_PER_TRADE_PCT = 0.004        # 0.4% of equity
RISK_HALVED_PCT    = 0.002        # 0.2% when drawdown circuit fires
STOP_ATR_MULT      = 1.5
REWARD_MULT        = 2.0          # 1:2 R:R fixed
ADX_THRESHOLD      = 20.0
ATR_VOL_CAP        = 2.0          # ATR must be < 2× its 20-day avg
RSI_LONG_BAND      = (40, 70)
RSI_SHORT_BAND     = (30, 60)
TIME_EXIT_DAYS     = 10

FILTER_COUNTS = {"trend": 0, "adx": 0, "vol": 0, "reclaim": 0, "rsi": 0, "pass": 0}

# Pip definitions (for sizing + spread)
PIP_SIZE = {
    "GBP_USD": 0.0001,
    "EUR_USD": 0.0001,
    "AUD_USD": 0.0001,
    "USD_JPY": 0.01,
    "XAU_USD": 0.01,      # gold convention: 1 pip = $0.01 (sometimes $0.10 — see backtest)
}

# Average spread (price units, both sides applied at fill + exit by backtest)
TYPICAL_SPREAD = {
    "GBP_USD": 0.00015,   # 1.5 pips
    "EUR_USD": 0.00015,
    "AUD_USD": 0.00015,
    "USD_JPY": 0.015,     # 1.5 pips (JPY pair)
    "XAU_USD": 0.25,      # ~25 pips = $0.25/oz
}


# ----------------------------------------------------------------------------
# Data fetch — OANDA
# ----------------------------------------------------------------------------

def fetch_candles(pair: str, granularity: str, count: int) -> pd.DataFrame:
    """
    Pull OANDA candles. Returns DataFrame indexed by UTC timestamp with
    columns: open, high, low, close, volume.

    Requires env vars:
        OANDA_API_KEY
        OANDA_ACCOUNT_ID (not strictly required for candles, but conventional)
        OANDA_ENV  -> 'practice' (default) or 'live'

    granularity examples: 'D' (daily), 'H4' (4-hour), 'H1' (1-hour)
    """
    import requests

    env = os.environ.get("OANDA_ENV", "practice").lower()
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY env var is not set")

    url = f"https://{host}/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",          # midpoint
        "dailyAlignment": 17,  # NY close convention
        "alignmentTimezone": "America/New_York",
    }

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])

    rows = []
    for c in candles:
        if not c.get("complete"):
            continue
        m = c["mid"]
        rows.append({
            "time":   pd.Timestamp(c["time"]).tz_convert("UTC"),
            "open":   float(m["o"]),
            "high":   float(m["h"]),
            "low":    float(m["l"]),
            "close":  float(m["c"]),
            "volume": int(c["volume"]),
        })
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _atr(df, period) * period  # un-smooth to get a TR proxy
    # cleaner: recompute TR directly
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA20, EMA50, ATR14, RSI14, ADX14, ATR20avg columns."""
    out = df.copy()
    out["ema20"]    = _ema(out["close"], 20)
    out["ema50"]    = _ema(out["close"], 50)
    out["atr14"]    = _atr(out, 14)
    out["rsi14"]    = _rsi(out["close"], 14)
    out["adx14"]    = _adx(out, 14)
    out["atr20avg"] = out["atr14"].rolling(20).mean()
    return out


# ----------------------------------------------------------------------------
# Entry checks
# ----------------------------------------------------------------------------

def _has_warmup(d1: pd.DataFrame, h4: pd.DataFrame) -> bool:
    if len(d1) < 60 or len(h4) < 60:
        return False
    last_d1 = d1.iloc[-1]
    last_h4 = h4.iloc[-1]
    prev_h4 = h4.iloc[-2]
    for v in [last_d1.get("ema50"), last_d1.get("adx14"), last_d1.get("atr20avg"),
              last_h4.get("ema20"), last_h4.get("rsi14"), last_h4.get("atr14"),
              prev_h4.get("ema20")]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
    return True


def check_long_entry(d1_df: pd.DataFrame, h4_df: pd.DataFrame) -> Optional[dict]:
    """Return setup dict if all 5 long conditions met, else None."""
    if not _has_warmup(d1_df, h4_df):
        return None

    d  = d1_df.iloc[-1]
    h  = h4_df.iloc[-1]
    hp = h4_df.iloc[-2]

    # 1. Daily uptrend stack
    if not (d["close"] > d["ema20"] > d["ema50"]):
        FILTER_COUNTS["trend"] += 1
        return None
    # 2. Daily ADX
    if d["adx14"] <= ADX_THRESHOLD:
        FILTER_COUNTS["adx"] += 1
        return None
    # 3. Volatility cap
    if d["atr14"] >= ATR_VOL_CAP * d["atr20avg"]:
        FILTER_COUNTS["vol"] += 1
        return None
    # 4. H4 EMA20 reclaim
    if not (hp["close"] < hp["ema20"] and h["close"] > h["ema20"]):
        FILTER_COUNTS["reclaim"] += 1
        return None
    # 5. RSI band
    if not (RSI_LONG_BAND[0] <= h["rsi14"] <= RSI_LONG_BAND[1]):
        FILTER_COUNTS["rsi"] += 1
        return None

    entry         = float(h["ema20"])
    stop_distance = STOP_ATR_MULT * float(h["atr14"])
    stop          = entry - stop_distance
    target        = entry + REWARD_MULT * stop_distance

    FILTER_COUNTS["pass"] += 1
    return {
        "direction": "BUY",
        "entry":     entry,
        "stop":      stop,
        "target":    target,
        "stop_distance": stop_distance,
        "d1_adx":  float(d["adx14"]),
        "h4_rsi":  float(h["rsi14"]),
        "h4_atr":  float(h["atr14"]),
        "reason":  f"D1 uptrend + ADX {d['adx14']:.1f} + H4 EMA20 reclaim, RSI {h['rsi14']:.1f}",
    }


def check_short_entry(d1_df: pd.DataFrame, h4_df: pd.DataFrame) -> Optional[dict]:
    if not _has_warmup(d1_df, h4_df):
        return None

    d  = d1_df.iloc[-1]
    h  = h4_df.iloc[-1]
    hp = h4_df.iloc[-2]

    if not (d["close"] < d["ema20"] < d["ema50"]):
        FILTER_COUNTS["trend"] += 1
        return None
    if d["adx14"] <= ADX_THRESHOLD:
        FILTER_COUNTS["adx"] += 1
        return None
    if d["atr14"] >= ATR_VOL_CAP * d["atr20avg"]:
        FILTER_COUNTS["vol"] += 1
        return None
    if not (hp["close"] > hp["ema20"] and h["close"] < h["ema20"]):
        FILTER_COUNTS["reclaim"] += 1
        return None
    if not (RSI_SHORT_BAND[0] <= h["rsi14"] <= RSI_SHORT_BAND[1]):
        FILTER_COUNTS["rsi"] += 1
        return None

    entry         = float(h["ema20"])
    stop_distance = STOP_ATR_MULT * float(h["atr14"])
    stop          = entry + stop_distance
    target        = entry - REWARD_MULT * stop_distance

    FILTER_COUNTS["pass"] += 1
    return {
        "direction": "SELL",
        "entry":     entry,
        "stop":      stop,
        "target":    target,
        "stop_distance": stop_distance,
        "d1_adx":  float(d["adx14"]),
        "h4_rsi":  float(h["rsi14"]),
        "h4_atr":  float(h["atr14"]),
        "reason":  f"D1 downtrend + ADX {d['adx14']:.1f} + H4 EMA20 breakdown, RSI {h['rsi14']:.1f}",
    }


# ----------------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------------

def size_position(equity: float, stop_distance: float, instrument: str,
                  entry_price: Optional[float] = None) -> int:
    """
    Returns integer instrument units.

    USD-quoted majors (EUR_USD, GBP_USD, AUD_USD):
        PnL_USD per unit per price-unit move = 1.0
        units = risk_USD / stop_distance

    USD-base pair (USD_JPY):
        PnL_USD per unit per price-unit move = 1 / entry_price
        units = risk_USD * entry_price / stop_distance

    XAU_USD (units = ounces, quote is USD/oz):
        units = risk_USD / stop_distance
    """
    risk_usd = equity * RISK_PER_TRADE_PCT
    if stop_distance <= 0:
        return 0

    if instrument == "USD_JPY":
        if not entry_price:
            return 0
        units = risk_usd * entry_price / stop_distance
    else:
        units = risk_usd / stop_distance

    return int(math.floor(units))


def usd_pnl(instrument: str, direction: str, entry_price: float,
            exit_price: float, units: int) -> float:
    """Convert a closed trade to USD P&L."""
    move = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
    if instrument == "USD_JPY":
        return units * move / exit_price
    return units * move  # USD-quoted majors and XAU_USD


# ----------------------------------------------------------------------------
# Top-level scan
# ----------------------------------------------------------------------------

def scan_all_instruments(equity: float) -> dict:
    """
    Run a full daily scan across all 5 instruments. Live use.
    Returns dict shaped like the daily_alert.json in the spec.
    """
    trades_today = []
    filtered = []

    for pair in INSTRUMENTS:
        try:
            d1_raw = fetch_candles(pair, "D",  120)
            h4_raw = fetch_candles(pair, "H4", 200)
        except Exception as e:
            filtered.append({"instrument": pair, "skipped": f"fetch error: {e}"})
            continue

        d1 = compute_indicators(d1_raw)
        h4 = compute_indicators(h4_raw)

        setup = check_long_entry(d1, h4) or check_short_entry(d1, h4)
        if not setup:
            filtered.append({"instrument": pair, "skipped": "no setup"})
            continue

        units = size_position(equity, setup["stop_distance"], pair, setup["entry"])
        if units <= 0:
            filtered.append({"instrument": pair, "skipped": "units below minimum"})
            continue

        risk_usd   = equity * RISK_PER_TRADE_PCT
        reward_usd = risk_usd * REWARD_MULT

        trades_today.append({
            "instrument":   pair,
            "direction":    setup["direction"],
            "entry_limit":  round(setup["entry"],  5),
            "stop_loss":    round(setup["stop"],   5),
            "take_profit":  round(setup["target"], 5),
            "units":        units,
            "risk_usd":     round(risk_usd,   2),
            "reward_usd":   round(reward_usd, 2),
            "expires_utc":  None,  # filled by caller (NY-close + 24h)
            "reason":       setup["reason"],
        })

    return {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "account_equity":   round(equity, 2),
        "drawdown_status":  "OK",
        "trades_today":     trades_today,
        "filtered_setups":  filtered,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    equity = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
    out = scan_all_instruments(equity)
    print(json.dumps(out, indent=2))
