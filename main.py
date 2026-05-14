"""
Multi-Pair Forex Signal System v3.0
Regime-adaptive, pair-specific strategy routing.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import schedule
import pytz

load_dotenv()

# ─── CREDENTIALS ─────────────────────────────────────────────────────────────
OANDA_API_KEY    = os.getenv("OANDA_API_KEY")
OANDA_ENV        = os.getenv("OANDA_ENV", "practice")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_SENDER     = os.getenv("EMAIL_SENDER")
EMAIL_RECIPIENT  = os.getenv("EMAIL_RECIPIENT")
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY")

OANDA_BASE = (
    "https://api-fxpractice.oanda.com/v3"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com/v3"
)

LOT_SIZE         = 100_000
RISK_PER_TRADE   = 2_000.00   # base USD risk per trade

# ─── CENTRAL BANK RATES ──────────────────────────────────────────────────────
# Update these after each central bank meeting
RATES = {
    "GBP": 3.75,    # Bank of England
    "USD": 3.625,   # Federal Reserve (midpoint 3.50-3.75)
    "EUR": 2.50,    # European Central Bank
    "AUD": 4.10,    # Reserve Bank of Australia
    "JPY": 0.50,    # Bank of Japan
}

# ─── PAIR-SPECIFIC CONFIGURATIONS ────────────────────────────────────────────
# Each pair has its own strategy, thresholds and special rules.
# This is the heart of the pair-specific approach.
PAIR_CONFIGS = {

    # ── GBP/USD ─────────────────────────────────────────────────────────────
    # Personality: momentum/trend pair, large directional moves, responds
    # cleanly to BoE/Fed divergence. Best trend-follower in our universe.
    "GBP_USD": {
        "name":           "GBP/USD",
        "base":           "GBP",
        "quote":          "USD",
        "pip":            0.0001,

        # Strategy routing
        "strategy":       "trend",     # trend | range | auto
        "adx_trend_min":  25,          # ADX must exceed to call trending
        "adx_range_max":  20,          # ADX must be below to call ranging

        # Trend entry (Strategy A)
        "atr_entry_band": 1.5,         # price within N×ATR of 50 EMA
        "rsi_buy_lo":     35,
        "rsi_buy_hi":     65,
        "rsi_sell_lo":    35,
        "rsi_sell_hi":    65,
        "atr_stop":       1.5,
        "atr_target":     2.0,         # 1:1.33 R:R — tighter target, higher win rate
        "max_hold":       20,

        # Range entry (Strategy B) — available if regime flips
        "range_lookback": 40,
        "rsi_oversold":   33,
        "rsi_overbought": 67,
        "range_target":   0.50,        # target 50% of range midpoint
        "atr_range_stop": 1.0,
        "max_hold_range": 12,

        # Commodity filter
        "commodity":           "XAU_USD",
        "commodity_label":     "Gold",
        "commodity_required":  False,  # preferred not required — size mod only
        "commodity_contrary_mod": 0.5, # half size if contrary

        # Interest rate
        "rate_base":     "GBP",
        "rate_quote":    "USD",

        # Special rules
        "intervention_long_above":  None,   # no BOJ-style intervention
        "intervention_short_below": None,
        "base_size_mod":            1.0,    # full size

        # Risk-on confirmation (not needed for GBP/USD)
        "require_risk_on": False,
    },

    # ── EUR/USD ─────────────────────────────────────────────────────────────
    # Personality: range-bound oscillator. ECB/Fed rate convergence means
    # EUR/USD spends most of its time in defined multi-week ranges, fading
    # extremes. Trend-following fails here. Mean reversion is the edge.
    # Higher ADX threshold (28) before calling it trending — needs a
    # stronger signal to treat it as a trend pair.
    "EUR_USD": {
        "name":           "EUR/USD",
        "base":           "EUR",
        "quote":          "USD",
        "pip":            0.0001,

        # Strategy routing
        "strategy":       "auto",      # auto = range by default, trend if ADX > 28
        "adx_trend_min":  28,          # higher bar for EUR — needs real momentum
        "adx_range_max":  22,          # slightly wider range zone

        # Trend entry (only when ADX > 28)
        "atr_entry_band": 1.5,
        "rsi_buy_lo":     35,
        "rsi_buy_hi":     60,          # tighter — don't buy EUR when extended
        "rsi_sell_lo":    40,
        "rsi_sell_hi":    65,
        "atr_stop":       1.5,
        "atr_target":     2.5,         # slightly tighter target for EUR
        "max_hold":       18,

        # Range entry (primary mode for EUR)
        "range_lookback": 40,          # 40-candle range definition
        "rsi_oversold":   32,
        "rsi_overbought": 68,
        "range_target":   0.45,        # target 45% toward range midpoint
        "atr_range_stop": 1.0,
        "max_hold_range": 12,

        # Commodity filter
        "commodity":           "XAU_USD",
        "commodity_label":     "Gold",
        "commodity_required":  False,
        "commodity_contrary_mod": 0.5,

        "rate_base":     "EUR",
        "rate_quote":    "USD",

        "intervention_long_above":  None,
        "intervention_short_below": None,
        "base_size_mod":            1.0,

        "require_risk_on": False,
    },

    # ── AUD/USD ─────────────────────────────────────────────────────────────
    # Personality: commodity/risk-on pair. Driven by gold, iron ore, Chinese
    # growth expectations and global risk appetite. Trends smoothly when
    # commodity cycle and risk-on align — but reverses fast on China
    # disappointment or risk-off. Gold alignment is REQUIRED for longs.
    # Lower ADX threshold (22) because AUD trends more gently.
    "AUD_USD": {
        "name":           "AUD/USD",
        "base":           "AUD",
        "quote":          "USD",
        "pip":            0.0001,

        "strategy":       "auto",
        "adx_trend_min":  22,          # lower bar — AUD trends more gently
        "adx_range_max":  18,

        "atr_entry_band": 1.5,
        "rsi_buy_lo":     35,
        "rsi_buy_hi":     62,
        "rsi_sell_lo":    38,
        "rsi_sell_hi":    65,
        "atr_stop":       1.5,
        "atr_target":     3.0,
        "max_hold":       20,

        "range_lookback": 35,
        "rsi_oversold":   33,
        "rsi_overbought": 67,
        "range_target":   0.50,
        "atr_range_stop": 1.0,
        "max_hold_range": 12,

        # Commodity filter — REQUIRED for AUD
        # Gold must be above its 50 EMA for longs, below for shorts
        # If contrary: NO TRADE (commodity_contrary_mod = 0 blocks the trade)
        "commodity":           "XAU_USD",
        "commodity_label":     "Gold",
        "commodity_required":  True,   # REQUIRED — no trade if contrary
        "commodity_contrary_mod": 0.0, # zero = blocked

        "rate_base":     "AUD",
        "rate_quote":    "USD",

        "intervention_long_above":  None,
        "intervention_short_below": None,
        "base_size_mod":            1.0,

        # Risk-on filter: AUD needs the broader market in risk-on mode
        # We proxy this as: price has made a higher high in last 10 candles
        "require_risk_on": True,
    },

    # ── USD/JPY ─────────────────────────────────────────────────────────────
    # Personality: carry trade and safe-haven pair with BOJ intervention
    # risk. Trends strongly when carry is in play (US rates >> Japan rates)
    # but subject to sudden government intervention near extreme levels.
    # Oil rising = JPY weakens = USD/JPY bullish.
    # NEVER trade within 2% of historical intervention zones.
    # Always 75% base position size due to gap risk from intervention.
    "USD_JPY": {
        "name":           "USD/JPY",
        "base":           "USD",
        "quote":          "JPY",
        "pip":            0.01,

        "strategy":       "trend",
        "adx_trend_min":  25,
        "adx_range_max":  20,

        "atr_entry_band": 1.5,
        "rsi_buy_lo":     38,          # tighter RSI range for JPY
        "rsi_buy_hi":     62,
        "rsi_sell_lo":    38,
        "rsi_sell_hi":    62,
        "atr_stop":       1.5,
        "atr_target":     2.5,         # tighter target — intervention caps moves
        "max_hold":       15,          # shorter hold — intervention can end trend

        "range_lookback": 30,
        "rsi_oversold":   35,
        "rsi_overbought": 65,
        "range_target":   0.45,
        "atr_range_stop": 1.0,
        "max_hold_range": 10,

        # Oil: rising oil = JPY weakens = USD/JPY longs supported
        "commodity":           "WTICO_USD",
        "commodity_label":     "Oil",
        "commodity_required":  False,
        "commodity_contrary_mod": 0.5,

        "rate_base":     "USD",
        "rate_quote":    "JPY",

        # BOJ intervention zones — historically triggered near 150-160
        # No longs above 153 (risk of sudden yen-buying intervention)
        # No shorts below 143 (risk of sudden yen-selling intervention)
        "intervention_long_above":  153.0,
        "intervention_short_below": 143.0,

        # Always trade at 75% size due to overnight gap/intervention risk
        "base_size_mod": 0.75,

        "require_risk_on": False,
    },
}


# ─── DATA FETCHING ────────────────────────────────────────────────────────────
def fetch_candles(instrument, granularity="D", count=250):
    url     = f"{OANDA_BASE}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params  = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    rows = []
    for c in r.json()["candles"]:
        if c["complete"]:
            rows.append({
                "time":  c["time"][:19],
                "open":  float(c["mid"]["o"]),
                "high":  float(c["mid"]["h"]),
                "low":   float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


# ─── INDICATORS ──────────────────────────────────────────────────────────────
def calculate_indicators(df):
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta     = df["close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() /
                                   loss.ewm(com=13, adjust=False).mean()))

    prev      = df["close"].shift(1)
    df["tr"]  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()

    # ADX (Wilder's directional movement)
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = df["tr"].ewm(com=13, adjust=False).mean()
    df["pdi"] = 100 * pd.Series(pdm, index=df.index).ewm(com=13, adjust=False).mean() / tr_s
    df["ndi"] = 100 * pd.Series(ndm, index=df.index).ewm(com=13, adjust=False).mean() / tr_s
    dx        = 100 * (df["pdi"] - df["ndi"]).abs() / (df["pdi"] + df["ndi"])
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    return df


# ─── REGIME DETECTION ─────────────────────────────────────────────────────────
def get_regime(df, cfg):
    """
    Returns (regime, direction_hint, adx_value) using pair-specific thresholds.
    TRENDING: ADX above pair threshold with clear EMA structure
    RANGING:  ADX below pair threshold
    AMBIGUOUS: in between — no trade
    """
    adx  = df["adx"].iloc[-1]
    pdi  = df["pdi"].iloc[-1]
    ndi  = df["ndi"].iloc[-1]
    e50  = df["ema50"].iloc[-1]
    e200 = df["ema200"].iloc[-1]

    if adx >= cfg["adx_trend_min"]:
        if pdi > ndi and e50 > e200: return "TRENDING", "BULLISH", round(adx, 1)
        if ndi > pdi and e50 < e200: return "TRENDING", "BEARISH", round(adx, 1)
        return "TRENDING", "MIXED", round(adx, 1)

    if adx <= cfg["adx_range_max"]:
        return "RANGING", "NEUTRAL", round(adx, 1)

    return "AMBIGUOUS", "MIXED", round(adx, 1)


# ─── 4H CONFIRMATION ──────────────────────────────────────────────────────────
def get_4h_confirmation(df_4h, direction):
    """
    Hard blocker. 2 of 3 checks must pass.
    Neutral if no 4H data — do not block.
    """
    if df_4h is None or len(df_4h) < 3:
        return True, 0

    l      = df_4h.iloc[-1]
    checks = 0

    if direction == "BUY":
        if l["close"] > l["ema50"]:  checks += 1
        if l["ema50"] > l["ema200"]: checks += 1
        if l["rsi"] < 65:            checks += 1
    else:
        if l["close"] < l["ema50"]:  checks += 1
        if l["ema50"] < l["ema200"]: checks += 1
        if l["rsi"] > 35:            checks += 1

    return checks >= 2, checks


# ─── COMMODITY ALIGNMENT ──────────────────────────────────────────────────────
def get_commodity_alignment(comm_df, direction, cfg):
    """
    Checks commodity trend direction against trade direction.
    Returns (aligned, label, mod_factor).
    mod_factor = 1.0 if aligned or neutral, cfg value if contrary.
    """
    if comm_df is None or len(comm_df) < 52:
        return True, "No data — neutral", 1.0

    price        = comm_df["close"].iloc[-1]
    ema50        = comm_df["ema50"].iloc[-1]
    ema50_10ago  = comm_df["ema50"].iloc[-11] if len(comm_df) >= 11 else ema50
    slope        = (ema50 - ema50_10ago) / ema50_10ago * 100
    comm_bullish = price > ema50 and slope > 0

    label = cfg["commodity_label"]

    if direction == "BUY":
        if comm_bullish:
            return True,  f"{label} ▲ above 50 EMA — aligned with long", 1.0
        else:
            mod = cfg["commodity_contrary_mod"]
            if mod == 0.0:
                return False, f"{label} ▼ below 50 EMA — BLOCKS trade (required)", 0.0
            return False, f"{label} ▼ below 50 EMA — headwind, size {int(mod*100)}%", mod
    else:
        if not comm_bullish:
            return True,  f"{label} ▼ below 50 EMA — aligned with short", 1.0
        else:
            mod = cfg["commodity_contrary_mod"]
            if mod == 0.0:
                return False, f"{label} ▲ above 50 EMA — BLOCKS trade (required)", 0.0
            return False, f"{label} ▲ above 50 EMA — headwind, size {int(mod*100)}%", mod


# ─── RISK-ON FILTER (AUD/USD) ─────────────────────────────────────────────────
def get_risk_on(df, direction):
    """
    AUD/USD proxy for risk-on: price has made a higher high in last 10 candles.
    Risk-on for longs, risk-off for shorts.
    """
    if direction == "BUY":
        recent_high   = df["high"].iloc[-10:].max()
        previous_high = df["high"].iloc[-20:-10].max()
        return recent_high > previous_high
    else:
        recent_low   = df["low"].iloc[-10:].min()
        previous_low = df["low"].iloc[-20:-10].min()
        return recent_low < previous_low


# ─── RATE DIFFERENTIAL ────────────────────────────────────────────────────────
def get_rate_modifier(cfg, direction):
    """
    Returns size modifier based on rate differential direction.
    Trading against the rate differential = 0.5x size.
    """
    diff = RATES.get(cfg["rate_base"], 0) - RATES.get(cfg["rate_quote"], 0)
    if direction == "BUY"  and diff < -0.5: return 0.5, f"Rate differential headwind ({cfg['rate_base']} {RATES.get(cfg['rate_base'],0)}% vs {cfg['rate_quote']} {RATES.get(cfg['rate_quote'],0)}%)"
    if direction == "SELL" and diff >  0.5: return 0.5, f"Rate differential headwind ({cfg['rate_base']} {RATES.get(cfg['rate_base'],0)}% vs {cfg['rate_quote']} {RATES.get(cfg['rate_quote'],0)}%)"
    sign   = "+" if diff >= 0 else ""
    return 1.0, f"{cfg['rate_base']} {RATES.get(cfg['rate_base'],0)}% vs {cfg['rate_quote']} {RATES.get(cfg['rate_quote'],0)}% ({sign}{round(diff,2)}%)"


# ─── TIME FILTER ──────────────────────────────────────────────────────────────
def get_time_filter():
    """Returns (can_trade, size_mod, reason)."""
    now   = datetime.now(pytz.timezone("America/New_York"))
    month = now.month
    day   = now.day
    wday  = now.weekday()

    if wday == 4:                  return False, 0.0, "Friday — no new trades"
    if month == 12 and day >= 18:  return False, 0.0, "Year-end thin markets"
    if month == 1  and day <= 5:   return False, 0.0, "New year thin markets"
    if month == 8:                 return True,  0.5, "August — 50% size (thin markets)"
    return True, 1.0, ""


# ─── INTERVENTION CHECK (USD/JPY) ─────────────────────────────────────────────
def check_intervention_zone(price, direction, cfg):
    """Returns True if price is in a safe zone to trade."""
    if direction == "BUY" and cfg["intervention_long_above"] and price > cfg["intervention_long_above"]:
        return False, f"BLOCKED: price {price} above intervention zone {cfg['intervention_long_above']}"
    if direction == "SELL" and cfg["intervention_short_below"] and price < cfg["intervention_short_below"]:
        return False, f"BLOCKED: price {price} below intervention zone {cfg['intervention_short_below']}"
    return True, "Clear of intervention zones"


# ─── STRATEGY A: TREND FOLLOWING ──────────────────────────────────────────────
def strategy_trend(df, df_4h, comm_df, cfg):
    """
    Entry when:
    1. ADX > pair threshold (trending)
    2. Price has pulled back to 50 EMA (within atr_entry_band × ATR)
    3. 50/200 EMA structure aligned
    4. RSI in pair-specific range
    5. Previous candle confirms direction
    6. 4H agrees
    7. Commodity aligned (or within size rules)
    8. Risk-on confirmed (if required by pair)
    9. Not in intervention zone (USD/JPY)
    """
    l    = df.iloc[-1]
    prev = df.iloc[-2]

    price = l["close"]
    e50   = l["ema50"]
    e200  = l["ema200"]
    rsi   = l["rsi"]
    atr   = l["atr"]
    adx   = l["adx"]

    near_ema50 = abs(price - e50) <= cfg["atr_entry_band"] * atr

    reasons = []

    # BUY conditions
    if (adx >= cfg["adx_trend_min"]
            and e50 > e200 and near_ema50
            and cfg["rsi_buy_lo"] <= rsi <= cfg["rsi_buy_hi"]
            and prev["close"] > prev["open"]):

        h4_ok, h4_checks = get_4h_confirmation(df_4h, "BUY")
        if not h4_ok:
            return "BUY_NO4H", None, f"4H contradicts ({h4_checks}/3 checks)"

        comm_ok, comm_detail, comm_mod = get_commodity_alignment(comm_df, "BUY", cfg)
        if not comm_ok and comm_mod == 0.0:
            return "NO_SIGNAL", None, f"Commodity blocks: {comm_detail}"

        if cfg["require_risk_on"] and not get_risk_on(df, "BUY"):
            return "BUY_BIAS", None, "Risk-on not confirmed (no higher high)"

        safe, iv_msg = check_intervention_zone(price, "BUY", cfg)
        if not safe:
            return "NO_SIGNAL", None, iv_msg

        return "BUY", comm_mod, f"ADX {round(adx,1)} | near 50 EMA | RSI {round(rsi,1)} | 4H {h4_checks}/3 | {comm_detail}"

    # SELL conditions
    if (adx >= cfg["adx_trend_min"]
            and e50 < e200 and near_ema50
            and cfg["rsi_sell_lo"] <= rsi <= cfg["rsi_sell_hi"]
            and prev["close"] < prev["open"]):

        h4_ok, h4_checks = get_4h_confirmation(df_4h, "SELL")
        if not h4_ok:
            return "SELL_NO4H", None, f"4H contradicts ({h4_checks}/3 checks)"

        comm_ok, comm_detail, comm_mod = get_commodity_alignment(comm_df, "SELL", cfg)
        if not comm_ok and comm_mod == 0.0:
            return "NO_SIGNAL", None, f"Commodity blocks: {comm_detail}"

        if cfg["require_risk_on"] and not get_risk_on(df, "SELL"):
            return "SELL_BIAS", None, "Risk-off not confirmed (no lower low)"

        safe, iv_msg = check_intervention_zone(price, "SELL", cfg)
        if not safe:
            return "NO_SIGNAL", None, iv_msg

        return "SELL", comm_mod, f"ADX {round(adx,1)} | near 50 EMA | RSI {round(rsi,1)} | 4H {h4_checks}/3 | {comm_detail}"

    # Bias — structure is there but not at entry yet
    if adx >= cfg["adx_trend_min"] and e50 > e200:
        dist = round((price - e50) / atr, 1)
        return "BUY_BIAS", None, f"Wait for pullback to 50 EMA (currently {dist}×ATR away)"
    if adx >= cfg["adx_trend_min"] and e50 < e200:
        dist = round((e50 - price) / atr, 1)
        return "SELL_BIAS", None, f"Wait for rally to 50 EMA (currently {dist}×ATR away)"

    return "NO_SIGNAL", None, "No setup"


# ─── STRATEGY B: MEAN REVERSION ───────────────────────────────────────────────
def strategy_range(df, df_4h, comm_df, cfg):
    """
    Entry when:
    1. ADX < pair threshold (ranging)
    2. Price at range boundary (within 1 ATR of high or low)
    3. RSI extreme (oversold at low, overbought at high)
    4. Reversal candle
    5. 4H confirms
    6. Commodity aligned (or within size rules)
    """
    l      = df.iloc[-1]
    recent = df.iloc[-cfg["range_lookback"]:]

    price = l["close"]
    rsi   = l["rsi"]
    atr   = l["atr"]
    rh    = recent["high"].max()
    rl    = recent["low"].min()
    rmid  = (rh + rl) / 2

    near_low  = price <= rl + atr
    near_high = price >= rh - atr
    bull_candle = l["close"] > l["open"]
    bear_candle = l["close"] < l["open"]

    # BUY at range low — check SELL-side 4H (fading bearish pressure that created the low)
    if near_low and rsi <= cfg["rsi_oversold"] and bull_candle:
        h4_ok, h4_checks = get_4h_confirmation(df_4h, "SELL")
        if not h4_ok:
            return "BUY_NO4H", None, f"4H contradicts ({h4_checks}/3)", rh, rl, rmid

        comm_ok, comm_detail, comm_mod = get_commodity_alignment(comm_df, "BUY", cfg)
        if not comm_ok and comm_mod == 0.0:
            return "NO_SIGNAL", None, f"Commodity blocks: {comm_detail}", rh, rl, rmid

        return "BUY", comm_mod, f"At range low {round(rl,5)} | RSI {round(rsi,1)} oversold | reversal candle | 4H {h4_checks}/3", rh, rl, rmid

    # SELL at range high — check BUY-side 4H (fading bullish pressure that created the high)
    if near_high and rsi >= cfg["rsi_overbought"] and bear_candle:
        h4_ok, h4_checks = get_4h_confirmation(df_4h, "BUY")
        if not h4_ok:
            return "SELL_NO4H", None, f"4H contradicts ({h4_checks}/3)", rh, rl, rmid

        comm_ok, comm_detail, comm_mod = get_commodity_alignment(comm_df, "SELL", cfg)
        if not comm_ok and comm_mod == 0.0:
            return "NO_SIGNAL", None, f"Commodity blocks: {comm_detail}", rh, rl, rmid

        return "SELL", comm_mod, f"At range high {round(rh,5)} | RSI {round(rsi,1)} overbought | reversal candle | 4H {h4_checks}/3", rh, rl, rmid

    # Approaching boundaries — give advance warning
    if near_low:
        return "WATCH_BUY", None, f"Approaching range low {round(rl,5)} — watch for reversal candle + RSI {cfg['rsi_oversold']}", rh, rl, rmid
    if near_high:
        return "WATCH_SELL", None, f"Approaching range high {round(rh,5)} — watch for reversal candle + RSI {cfg['rsi_overbought']}", rh, rl, rmid

    pct = round((price - rl) / (rh - rl) * 100, 0)
    return "NO_SIGNAL", None, f"In range {round(rl,5)}-{round(rh,5)} at {int(pct)}%", rh, rl, rmid


# ─── POSITION SIZING ──────────────────────────────────────────────────────────
def calculate_position_size(atr, price, cfg, direction,
                             time_mod, comm_mod, rate_mod):
    """
    Volatility-adjusted position size with stacked modifiers.
    All modifiers multiply together — they do NOT replace each other.
    """
    stop_dist = cfg["atr_stop"] * atr

    # Dollar value of stop distance per standard lot
    if cfg["quote"] == "USD":
        stop_val = stop_dist * LOT_SIZE
    else:
        # USD is the base (USD/JPY) — convert stop to USD
        stop_val = (stop_dist / price) * LOT_SIZE

    # Stacked modifiers
    total_mod = cfg["base_size_mod"] * time_mod * comm_mod * rate_mod
    total_mod = max(total_mod, 0.0)  # no negative size

    effective_risk = RISK_PER_TRADE * total_mod
    lots           = round(effective_risk / max(stop_val, 1), 2)

    return max(lots, 0.01), round(effective_risk, 0), round(total_mod, 2)


# ─── ECONOMIC CALENDAR ────────────────────────────────────────────────────────
def fetch_economic_events():
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"https://finnhub.io/api/v1/calendar/economic"
           f"?from={today}&to={tomorrow}&token={FINNHUB_API_KEY}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return [e for e in r.json().get("economicCalendar", [])
                if e.get("impact") == "high"
                and e.get("country") in ("US", "GB", "EU", "AU", "JP")]
    except Exception as e:
        print(f"Calendar error: {e}")
        return []


# ─── ANALYSE SINGLE PAIR ──────────────────────────────────────────────────────
def analyse_pair(pair, df, df_4h, comm_df, time_mod):
    cfg   = PAIR_CONFIGS[pair]
    l     = df.iloc[-1]
    prev  = df.iloc[-2]

    price = round(l["close"], 5)
    e50   = round(l["ema50"], 5)
    e200  = round(l["ema200"], 5)
    rsi   = round(l["rsi"], 1)
    atr   = round(l["atr"], 5)
    adx   = round(l["adx"], 1)
    pdi   = round(l["pdi"], 1)
    ndi   = round(l["ndi"], 1)

    price_chg = round(price - prev["close"], 5)
    chg_str   = f"+{price_chg}" if price_chg > 0 else str(price_chg)
    price_dir = "▲" if price_chg > 0 else "▼"
    e50_dir   = "▲" if e50 > prev["ema50"]  else "▼"
    e200_dir  = "▲" if e200 > prev["ema200"] else "▼"
    rsi_dir   = "▲" if rsi  > prev["rsi"]    else "▼"
    atr_dir   = "▲" if atr  > prev["atr"]    else "▼"

    regime, direction_hint, adx_val = get_regime(df, cfg)
    rate_mod, rate_detail           = get_rate_modifier(cfg, "BUY")  # neutral check

    out = {
        "pair": pair, "cfg": cfg,
        "price": price, "price_dir": price_dir, "price_chg": chg_str,
        "e50": e50, "e50_dir": e50_dir,
        "e200": e200, "e200_dir": e200_dir,
        "rsi": rsi, "rsi_dir": rsi_dir,
        "atr": atr, "atr_dir": atr_dir,
        "adx": adx_val, "pdi": pdi, "ndi": ndi,
        "regime": regime, "direction_hint": direction_hint,
        "rate_detail": rate_detail,
        "signal": "NO_SIGNAL", "direction": None, "strategy_used": None,
        "signal_detail": "", "entry": None, "stop": None, "target": None,
        "lots": None, "eff_risk": None, "total_mod": None,
        "modifiers": [],
        "range_high": None, "range_low": None, "range_mid": None,
    }

    if regime == "AMBIGUOUS":
        out["signal"] = "STAND DOWN — ADX ambiguous"
        out["signal_detail"] = f"ADX {adx_val} between thresholds {cfg['adx_range_max']}-{cfg['adx_trend_min']}"
        return out

    # Route to correct strategy based on pair config and regime
    strategy_type = cfg["strategy"]
    if strategy_type == "trend" or (strategy_type == "auto" and regime == "TRENDING"):
        sig, comm_mod, detail = strategy_trend(df, df_4h, comm_df, cfg)
        out["strategy_used"] = "A — Trend Following"
        rh = rl = rm = None
    elif strategy_type == "range" or (strategy_type == "auto" and regime == "RANGING"):
        sig, comm_mod, detail, rh, rl, rm = strategy_range(df, df_4h, comm_df, cfg)
        out["strategy_used"] = "B — Mean Reversion"
        out["range_high"] = round(rh, 5) if rh else None
        out["range_low"]  = round(rl, 5) if rl else None
        out["range_mid"]  = round(rm, 5) if rm else None
    else:
        out["signal"] = "NO_SIGNAL — Regime/strategy mismatch"
        return out

    out["signal"]        = sig
    out["signal_detail"] = detail

    # Only calculate sizing for actionable signals
    if sig in ("BUY", "SELL"):
        direction = sig
        out["direction"] = direction

        # Get final rate modifier for this direction
        rate_mod, rate_detail = get_rate_modifier(cfg, direction)
        out["rate_detail"]    = rate_detail

        # Build modifier list for transparency
        mods = []
        if cfg["base_size_mod"] < 1.0:
            mods.append(f"×{cfg['base_size_mod']} base ({cfg['name']} intervention risk)")
        if time_mod < 1.0:
            mods.append(f"×{time_mod} time (thin market period)")
        if comm_mod is not None and comm_mod < 1.0:
            mods.append(f"×{comm_mod} commodity headwind")
        if rate_mod < 1.0:
            mods.append(f"×{rate_mod} rate differential headwind")

        lots, eff_risk, total_mod = calculate_position_size(
            l["atr"], price, cfg, direction,
            time_mod,
            comm_mod if comm_mod is not None else 1.0,
            rate_mod
        )

        # Entry, stop, target
        if out["strategy_used"] == "A — Trend Following":
            if direction == "BUY":
                stop   = round(price - cfg["atr_stop"] * l["atr"], 5)
                target = round(price + cfg["atr_target"] * l["atr"], 5)
            else:
                stop   = round(price + cfg["atr_stop"] * l["atr"], 5)
                target = round(price - cfg["atr_target"] * l["atr"], 5)
            rr = round(cfg["atr_target"] / cfg["atr_stop"], 1)
        else:
            if direction == "BUY":
                stop   = round(rl - l["atr"] * cfg["atr_range_stop"], 5) if rl else None
                target = round(rm, 5) if rm else None
            else:
                stop   = round(rh + l["atr"] * cfg["atr_range_stop"], 5) if rh else None
                target = round(rm, 5) if rm else None
            rr = round(abs(target - price) / abs(stop - price), 1) if stop and target else "—"

        out.update({
            "entry": price, "stop": stop, "target": target,
            "lots": lots, "eff_risk": eff_risk, "total_mod": total_mod,
            "modifiers": mods,
            "rr": rr,
        })

    return out


# ─── EMAIL BUILDER & SENDER ───────────────────────────────────────────────────
def build_and_send_email(pair_results, events, tf_msg, size_mod):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    def action_emoji(sig):
        if sig == "BUY":           return "🟢"
        if sig == "SELL":          return "🔴"
        if "BIAS" in sig:          return "🟡"
        if "NO4H" in sig:          return "🟡"
        if "WATCH" in sig:         return "🟡"
        if "STAND DOWN" in sig:    return "⛔"
        return "⚪"

    confirmed = [r for r in pair_results if r["signal"] in ("BUY","SELL")]
    watching  = [r for r in pair_results if any(x in r["signal"] for x in ("BIAS","NO4H","WATCH"))]
    summary   = f"{len(confirmed)} confirmed | {len(watching)} watching | {4-len(confirmed)-len(watching)} no setup"

    # News block
    news_block = ""
    if events:
        lines = "\n".join(f"  ⚠️  {e.get('event','?')} ({e.get('country','?')})" for e in events)
        news_block = f"\n{'─'*64}\n⛔ HIGH IMPACT NEWS TODAY — VERIFY BEFORE ANY TRADE\n{lines}\n{'─'*64}\n"

    # Time filter block
    tf_block = ""
    if tf_msg:
        tf_block = f"\n⚠️  {tf_msg}"

    # Build pair sections
    pair_sections = ""
    for r in pair_results:
        cfg   = r["cfg"]
        emoji = action_emoji(r["signal"])
        reg_icon = {"TRENDING": "📈", "RANGING": "➡️", "AMBIGUOUS": "⚠️"}.get(r["regime"], "")

        # Entry box for confirmed signals
        entry_block = ""
        if r["signal"] in ("BUY", "SELL"):
            direction = r["direction"]
            mods_str  = "\n    ".join(r["modifiers"]) if r["modifiers"] else "None — full size"
            entry_block = f"""
  ╔══ {direction} SIGNAL — EXECUTE ══════════════════════════╗
  ║  Strategy : {r.get('strategy_used','—')}
  ║  Entry    : {r['entry']}
  ║  Stop     : {r['stop']}   ({cfg['atr_stop']}× ATR)
  ║  Target   : {r['target']}   ({cfg['atr_target']}× ATR)
  ║  R:R      : 1:{r.get('rr','—')}
  ║  Size     : {r['lots']} lots  (${r['eff_risk']:,.0f} effective risk)
  ║  Modifier : ×{r['total_mod']}
  ║  Detail   : {r['signal_detail'][:55]}
  ╚═══════════════════════════════════════════════════════╝
  Size modifiers:
    {mods_str}"""
        elif any(x in r["signal"] for x in ("BIAS","NO4H","WATCH")):
            entry_block = f"\n  🟡 {r['signal']}\n  {r.get('signal_detail','')}"
        else:
            entry_block = f"\n  ⚪ {r.get('signal_detail', 'No setup today')}"

        # Range info for Strategy B
        range_block = ""
        if r["range_high"]:
            pct = round((r["price"] - r["range_low"]) / (r["range_high"] - r["range_low"]) * 100)
            range_block = f"""
  Range     : {r['range_low']} ─── {r['range_mid']} ─── {r['range_high']}  (at {pct}%)"""

        pair_sections += f"""
{'═'*64}
  {emoji}  {cfg['name']}   │   {reg_icon} {r['regime']}   │   ADX {r['adx']}  (+DI {r['pdi']} / −DI {r['ndi']})
  Strategy  : {r.get('strategy_used', 'None')}
{'─'*64}{entry_block}

  ─ Market Snapshot ─────────────────────────────────────
  Price     : {r['price']} {r['price_dir']}  ({r['price_chg']} today)
  50 EMA    : {r['e50']} {r['e50_dir']}
  200 EMA   : {r['e200']} {r['e200_dir']}
  RSI (14)  : {r['rsi']} {r['rsi_dir']}
  ATR (14)  : {r['atr']} {r['atr_dir']}{range_block}

  ─ Fundamental Filters ──────────────────────────────────
  Rates     : {r['rate_detail']}
  Commodity : {cfg['commodity_label']}  ({cfg['commodity']})
"""

    body = f"""MULTI-PAIR FOREX SIGNAL REPORT
{ny_time}
{'═'*64}
{summary}{tf_block}
{news_block}
{pair_sections}
{'═'*64}
STRATEGY REFERENCE
  Strategy A (Trending)   ADX>{'{}'} | price at 50 EMA | prev candle | 4H | commodity
  Strategy B (Ranging)    ADX<{'{}'} | range boundary | RSI extreme | reversal candle | 4H
  Ambiguous zone          ADX between thresholds — STAND DOWN on affected pair

PAIR-SPECIFIC RULES
  GBP/USD   Trend (ADX>{PAIR_CONFIGS['GBP_USD']['adx_trend_min']}) | 1:{PAIR_CONFIGS['GBP_USD']['atr_target']/PAIR_CONFIGS['GBP_USD']['atr_stop']:.2f} R:R | Gold preferred
  EUR/USD   Auto (range primary, trend if ADX>{PAIR_CONFIGS['EUR_USD']['adx_trend_min']}) | Gold preferred
  AUD/USD   Trend (ADX>{PAIR_CONFIGS['AUD_USD']['adx_trend_min']}) | Gold REQUIRED | Risk-on filter
  USD/JPY   Trend (ADX>{PAIR_CONFIGS['USD_JPY']['adx_trend_min']}) | No longs >{PAIR_CONFIGS['USD_JPY']['intervention_long_above']} | No shorts <{PAIR_CONFIGS['USD_JPY']['intervention_short_below']} | 75% size

RISK RULES
  Risk per trade : $2,000 base (modified per pair and conditions)
  Max open       : 4 (one per pair)
  Max same-side  : 2 (no full USD stack)
  No trades      : Fridays | Dec 18–Jan 5 | August 50% size

{'─'*64}
Multi-Pair Forex System v3.0 | Pair-Specific Regime-Adaptive""".format(
        max(c["adx_trend_min"] for c in PAIR_CONFIGS.values()),
        min(c["adx_range_max"] for c in PAIR_CONFIGS.values())
    )

    subject = f"Forex: {summary} | {ny_date}"

    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
        json={
            "personalizations": [{"to": [{"email": EMAIL_RECIPIENT}]}],
            "from":    {"email": EMAIL_SENDER},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        }
    )
    r.raise_for_status()
    print(f"Email sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─── MAIN JOB ─────────────────────────────────────────────────────────────────
def run_signal_job():
    print(f"\n{'='*50}")
    print(f"Signal job at {datetime.now(timezone.utc)} UTC")
    print(f"{'='*50}")

    try:
        can_trade, time_mod, tf_msg = get_time_filter()
        print(f"Time filter: {'OK' if can_trade else 'BLOCKED'} {tf_msg}")

        events = fetch_economic_events()
        print(f"Events today: {len(events)}")

        # Fetch commodity data
        comm_cache = {}
        for comm in {cfg["commodity"] for cfg in PAIR_CONFIGS.values()}:
            try:
                cdf = fetch_candles(comm, "D", 250)
                comm_cache[comm] = calculate_indicators(cdf)
                print(f"  {comm}: loaded")
            except Exception as e:
                print(f"  {comm}: FAILED — {e}")
                comm_cache[comm] = None

        # Analyse each pair
        pair_results = []
        for pair, cfg in PAIR_CONFIGS.items():
            print(f"  Analysing {cfg['name']}...")
            try:
                df    = fetch_candles(pair, "D",  250)
                df    = calculate_indicators(df)
                df_4h = fetch_candles(pair, "H4", 300)
                df_4h = calculate_indicators(df_4h)
                comm  = comm_cache.get(cfg["commodity"])

                result = analyse_pair(pair, df, df_4h, comm,
                                      time_mod if can_trade else 0.0)
                pair_results.append(result)
                print(f"    {cfg['name']}: {result['regime']} | {result['signal']}")

            except Exception as e:
                print(f"    {cfg['name']}: ERROR — {e}")
                pair_results.append({
                    "pair": pair, "cfg": cfg,
                    "signal": f"ERROR: {e}", "strategy_used": "—",
                    "regime": "ERROR", "direction_hint": "",
                    "price": 0, "price_dir": "", "price_chg": "",
                    "e50": 0, "e50_dir": "", "e200": 0, "e200_dir": "",
                    "rsi": 0, "rsi_dir": "", "atr": 0, "atr_dir": "",
                    "adx": 0, "pdi": 0, "ndi": 0,
                    "rate_detail": "", "signal_detail": "",
                    "direction": None, "entry": None, "stop": None,
                    "target": None, "lots": None, "eff_risk": None,
                    "total_mod": None, "modifiers": [], "rr": None,
                    "range_high": None, "range_low": None, "range_mid": None,
                })

        build_and_send_email(pair_results, events, tf_msg, time_mod)

    except Exception as e:
        print(f"Job error: {e}")
        import traceback
        traceback.print_exc()


# ─── SCHEDULER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Pair Forex Signal System v3.0")
    print("Pairs: GBP/USD | EUR/USD | AUD/USD | USD/JPY")
    print("Strategies: Pair-specific, regime-adaptive")
    print("Schedule: 5:00 PM EST daily (NY close)")
    print()
    for pair, cfg in PAIR_CONFIGS.items():
        print(f"  {cfg['name']}: {cfg['strategy'].upper()} | "
              f"ADX>{cfg['adx_trend_min']} trend | "
              f"ADX<{cfg['adx_range_max']} range | "
              f"{cfg['commodity_label']} {'REQUIRED' if cfg['commodity_required'] else 'preferred'}")
    print()

    run_signal_job()

    schedule.every().day.at("22:00").do(run_signal_job)  # 22:00 UTC = 17:00 EST

    while True:
        schedule.run_pending()
        time.sleep(60)
