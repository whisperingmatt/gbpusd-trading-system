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

# ─── CONFIG ──────────────────────────────────────────────────────────────────
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

# Risk settings
RISK_PER_TRADE   = 2000.00   # USD per trade
LOT_SIZE         = 100_000   # standard lot

# Strategy thresholds
ADX_TREND_MIN    = 25        # ADX above this = trending
ADX_RANGE_MAX    = 20        # ADX below this = ranging
RSI_BUY_MAX      = 58        # RSI upper limit for trend buy entry
RSI_BUY_MIN      = 38        # RSI lower limit for trend buy entry
RSI_SELL_MAX     = 62        # RSI upper limit for trend sell entry
RSI_SELL_MIN     = 42        # RSI lower limit for trend sell entry
RSI_OVERSOLD     = 32        # RSI for mean reversion buy
RSI_OVERBOUGHT   = 68        # RSI for mean reversion sell
ATR_ENTRY_MULT   = 0.5       # price must be within this × ATR of 50 EMA
ATR_STOP_MULT    = 1.5       # stop distance
RANGE_LOOKBACK   = 30        # candles to define range for Strategy B

# Central bank rates (update after each meeting)
RATES = {
    "GBP": 3.75,
    "USD": 3.625,
    "EUR": 2.50,
    "AUD": 4.10,
    "JPY": 0.50,
}

# Pairs config: instrument, base currency, quote currency, commodity, commodity relationship
PAIRS = {
    "GBP_USD": {"base": "GBP", "quote": "USD", "pip": 0.0001, "commodity": "XAU_USD",  "comm_label": "Gold"},
    "EUR_USD": {"base": "EUR", "quote": "USD", "pip": 0.0001, "commodity": "XAU_USD",  "comm_label": "Gold"},
    "AUD_USD": {"base": "AUD", "quote": "USD", "pip": 0.0001, "commodity": "XAU_USD",  "comm_label": "Gold"},
    "USD_JPY": {"base": "USD", "quote": "JPY", "pip": 0.01,   "commodity": "WTICO_USD","comm_label": "Oil"},
}


# ─── DATA FETCHING ───────────────────────────────────────────────────────────
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
                "time":   c["time"][:19],
                "open":   float(c["mid"]["o"]),
                "high":   float(c["mid"]["h"]),
                "low":    float(c["mid"]["l"]),
                "close":  float(c["mid"]["c"]),
            })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


# ─── INDICATORS ──────────────────────────────────────────────────────────────
def calculate_indicators(df):
    # EMAs
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() /
                                   loss.ewm(com=13, adjust=False).mean()))

    # ATR
    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()

    # ADX (Wilder's method)
    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_s   = df["tr"].ewm(com=13, adjust=False).mean()
    pdm_s  = pd.Series(pos_dm, index=df.index).ewm(com=13, adjust=False).mean()
    ndm_s  = pd.Series(neg_dm, index=df.index).ewm(com=13, adjust=False).mean()

    df["pdi"] = 100 * pdm_s / tr_s
    df["ndi"] = 100 * ndm_s / tr_s
    dx        = 100 * (df["pdi"] - df["ndi"]).abs() / (df["pdi"] + df["ndi"])
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    return df


# ─── TIME FILTER ─────────────────────────────────────────────────────────────
def time_filter():
    """
    Returns (can_trade, size_modifier, reason).
    """
    now   = datetime.now(pytz.timezone("America/New_York"))
    month = now.month
    day   = now.day
    wday  = now.weekday()  # 0=Mon, 4=Fri

    # Never trade
    if wday == 4:
        return False, 0.0, "Friday — no trades"
    if month == 12 and day >= 18:
        return False, 0.0, "Year-end thin markets"
    if month == 1 and day <= 5:
        return False, 0.0, "New year thin markets"

    # Reduced size
    if month == 8:
        return True, 0.5, "August — reduced size (thin markets)"

    return True, 1.0, ""


# ─── REGIME DETECTION ────────────────────────────────────────────────────────
def classify_regime(df):
    """
    TRENDING / RANGING / AMBIGUOUS based on ADX and EMA structure.
    Returns (regime, adx_value, direction_hint).
    """
    adx   = df["adx"].iloc[-1]
    pdi   = df["pdi"].iloc[-1]
    ndi   = df["ndi"].iloc[-1]
    e50   = df["ema50"].iloc[-1]
    e200  = df["ema200"].iloc[-1]
    price = df["close"].iloc[-1]

    # ADX rising?
    adx_rising = adx > df["adx"].iloc[-4]

    if adx > ADX_TREND_MIN and adx_rising:
        if pdi > ndi and e50 > e200:
            return "TRENDING", adx, "BULLISH"
        elif ndi > pdi and e50 < e200:
            return "TRENDING", adx, "BEARISH"
        else:
            return "AMBIGUOUS", adx, "MIXED"
    elif adx < ADX_RANGE_MAX:
        return "RANGING", adx, "NEUTRAL"
    else:
        return "AMBIGUOUS", adx, "MIXED"


# ─── 4H CONFIRMATION ─────────────────────────────────────────────────────────
def get_4h_confirmation(df_4h, direction):
    """
    Hard blocker. Returns True if 4H agrees, False if it contradicts.
    2 of 3 checks must pass.
    """
    l = df_4h.iloc[-1]
    checks = 0

    if direction == "BUY":
        if l["close"] > l["ema50"]:  checks += 1
        if l["ema50"] > l["ema200"]: checks += 1
        if 35 <= l["rsi"] <= 68:     checks += 1
        if l["rsi"] < 35:            checks += 1  # oversold = buy confirmed
    else:
        if l["close"] < l["ema50"]:  checks += 1
        if l["ema50"] < l["ema200"]: checks += 1
        if 32 <= l["rsi"] <= 65:     checks += 1
        if l["rsi"] > 65:            checks += 1  # overbought = sell confirmed

    return checks >= 2, checks, round(l["rsi"], 1)


# ─── COMMODITY ALIGNMENT ─────────────────────────────────────────────────────
def get_commodity_alignment(comm_df, direction, pair):
    """
    Checks if commodity trend aligns with trade direction.
    Returns (alignment, detail).
    ALIGNED / CONTRARY / NEUTRAL
    """
    if comm_df is None or len(comm_df) < 52:
        return "NEUTRAL", "No commodity data"

    price  = comm_df["close"].iloc[-1]
    ema50  = comm_df["ema50"].iloc[-1]
    ema50_slope = (ema50 - comm_df["ema50"].iloc[-10]) / comm_df["ema50"].iloc[-10] * 100
    comm_bullish = price > ema50 and ema50_slope > 0

    # For all our pairs: rising commodity = bullish pair direction
    # (gold = inverse USD = good for GBP/USD, EUR/USD, AUD/USD longs)
    # (oil rising = JPY weakens = USD/JPY longs)
    if direction == "BUY" and comm_bullish:
        return "ALIGNED",  f"Above 50 EMA, slope +{round(ema50_slope,2)}%"
    elif direction == "SELL" and not comm_bullish:
        return "ALIGNED",  f"Below 50 EMA, slope {round(ema50_slope,2)}%"
    elif direction == "BUY" and not comm_bullish:
        return "CONTRARY", f"Below 50 EMA — headwind for longs"
    elif direction == "SELL" and comm_bullish:
        return "CONTRARY", f"Above 50 EMA — headwind for shorts"
    return "NEUTRAL", "Flat"


# ─── RATE DIFFERENTIAL ───────────────────────────────────────────────────────
def get_rate_differential(pair):
    """
    Returns (differential, bias, detail).
    Positive = base currency has rate advantage.
    """
    cfg   = PAIRS[pair]
    base  = cfg["base"]
    quote = cfg["quote"]
    diff  = RATES.get(base, 0) - RATES.get(quote, 0)

    if diff > 0.5:
        bias   = "BULLISH"
        detail = f"{base} rate {RATES[base]}% vs {quote} {RATES[quote]}% (+{round(diff,2)}%)"
    elif diff < -0.5:
        bias   = "BEARISH"
        detail = f"{base} rate {RATES[base]}% vs {quote} {RATES[quote]}% ({round(diff,2)}%)"
    else:
        bias   = "NEUTRAL"
        detail = f"Rate differential near zero ({round(diff,2)}%)"

    return diff, bias, detail


# ─── STRATEGY A: TREND FOLLOWING ─────────────────────────────────────────────
def strategy_a_signal(df, df_4h):
    """
    Entry only when:
    1. ADX > 25 and rising
    2. Price touches 50 EMA (within ATR_ENTRY_MULT × ATR)
    3. 50 EMA above/below 200 EMA
    4. RSI in favourable zone at touch
    5. Previous candle closed in trend direction
    6. 4H confirms
    """
    l    = df.iloc[-1]
    prev = df.iloc[-2]

    price = l["close"]
    e50   = l["ema50"]
    e200  = l["ema200"]
    rsi   = l["rsi"]
    atr   = l["atr"]
    adx   = l["adx"]

    near_ema50  = abs(price - e50) <= ATR_ENTRY_MULT * atr
    adx_rising  = adx > df["adx"].iloc[-4]
    bull_struct = e50 > e200 and price >= e50
    bear_struct = e50 < e200 and price <= e50
    prev_bull   = prev["close"] > prev["open"]
    prev_bear   = prev["close"] < prev["open"]

    if (adx > ADX_TREND_MIN and adx_rising and near_ema50
            and bull_struct and RSI_BUY_MIN <= rsi <= RSI_BUY_MAX and prev_bull):
        conf, checks, rsi_4h = get_4h_confirmation(df_4h, "BUY")
        if conf:
            return "BUY", checks, rsi_4h
        return "BUY_NO4H", checks, rsi_4h

    if (adx > ADX_TREND_MIN and adx_rising and near_ema50
            and bear_struct and RSI_SELL_MIN <= rsi <= RSI_SELL_MAX and prev_bear):
        conf, checks, rsi_4h = get_4h_confirmation(df_4h, "SELL")
        if conf:
            return "SELL", checks, rsi_4h
        return "SELL_NO4H", checks, rsi_4h

    # Bias without full entry
    if adx > ADX_TREND_MIN and adx_rising and bull_struct:
        return "BUY_BIAS", 0, round(l["rsi"], 1)
    if adx > ADX_TREND_MIN and adx_rising and bear_struct:
        return "SELL_BIAS", 0, round(l["rsi"], 1)

    return "NO_SIGNAL", 0, round(rsi, 1)


# ─── STRATEGY B: MEAN REVERSION ──────────────────────────────────────────────
def strategy_b_signal(df, df_4h):
    """
    Entry only when:
    1. ADX < 20
    2. Price within 1x ATR of 30-candle range boundary
    3. RSI extreme (< 32 for buy, > 68 for sell)
    4. Reversal candle (closes in direction of trade)
    5. 4H confirms
    """
    l    = df.iloc[-1]
    recent = df.iloc[-RANGE_LOOKBACK:]

    price        = l["close"]
    rsi          = l["rsi"]
    atr          = l["atr"]
    range_high   = recent["high"].max()
    range_low    = recent["low"].min()
    range_mid    = (range_high + range_low) / 2
    near_high    = abs(price - range_high) <= atr
    near_low     = abs(price - range_low) <= atr
    bull_candle  = l["close"] > l["open"]
    bear_candle  = l["close"] < l["open"]

    if near_low and rsi < RSI_OVERSOLD and bull_candle:
        conf, checks, rsi_4h = get_4h_confirmation(df_4h, "BUY")
        if conf:
            return "BUY", checks, rsi_4h, range_high, range_low, range_mid
        return "BUY_NO4H", checks, rsi_4h, range_high, range_low, range_mid

    if near_high and rsi > RSI_OVERBOUGHT and bear_candle:
        conf, checks, rsi_4h = get_4h_confirmation(df_4h, "SELL")
        if conf:
            return "SELL", checks, rsi_4h, range_high, range_low, range_mid
        return "SELL_NO4H", checks, rsi_4h, range_high, range_low, range_mid

    return "NO_SIGNAL", 0, round(rsi, 1), range_high, range_low, range_mid


# ─── POSITION SIZING ─────────────────────────────────────────────────────────
def calculate_position_size(atr, price, pair, size_modifier, comm_alignment, rate_bias, direction):
    """
    Volatility-adjusted position sizing.
    Returns (lots, effective_risk, modifiers_applied).
    """
    stop_distance = ATR_STOP_MULT * atr
    cfg           = PAIRS[pair]

    # Dollar value of stop distance per lot
    if cfg["quote"] == "USD":
        stop_value_per_lot = stop_distance * LOT_SIZE
    else:
        # USD is base (USD/JPY) — stop in JPY, convert to USD
        stop_value_per_lot = (stop_distance / price) * LOT_SIZE

    modifiers = []
    modifier  = size_modifier

    # Commodity contrary signal
    if comm_alignment == "CONTRARY":
        modifier *= 0.5
        modifiers.append("50% — commodity headwind")

    # Rate differential contrary
    if (rate_bias == "BULLISH" and direction == "SELL") or \
       (rate_bias == "BEARISH" and direction == "BUY"):
        modifier *= 0.5
        modifiers.append("50% — rate differential headwind")

    effective_risk = RISK_PER_TRADE * modifier
    lots           = round(effective_risk / stop_value_per_lot, 2)
    lots           = max(lots, 0.01)

    return lots, round(effective_risk, 0), modifiers


# ─── ECONOMIC CALENDAR ───────────────────────────────────────────────────────
def fetch_economic_events():
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    url      = (
        f"https://finnhub.io/api/v1/calendar/economic"
        f"?from={today}&to={tomorrow}&token={FINNHUB_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        events = r.json().get("economicCalendar", [])
        return [
            e for e in events
            if e.get("impact") == "high"
            and e.get("country") in ("US", "GB", "EU", "AU", "JP")
        ]
    except Exception as e:
        print(f"Calendar error: {e}")
        return []


# ─── ANALYSE SINGLE PAIR ─────────────────────────────────────────────────────
def analyse_pair(pair, df, df_4h, comm_df, size_modifier):
    cfg       = PAIRS[pair]
    l         = df.iloc[-1]
    prev      = df.iloc[-2]

    price     = round(l["close"], 5)
    e50       = round(l["ema50"], 5)
    e200      = round(l["ema200"], 5)
    rsi       = round(l["rsi"], 1)
    atr       = round(l["atr"], 5)
    adx       = round(l["adx"], 1)
    pdi       = round(l["pdi"], 1)
    ndi       = round(l["ndi"], 1)

    price_dir = "▲" if price > prev["close"] else "▼"
    e50_dir   = "▲" if e50  > prev["ema50"]  else "▼"
    e200_dir  = "▲" if e200 > prev["ema200"] else "▼"
    rsi_dir   = "▲" if rsi  > prev["rsi"]    else "▼"
    price_chg = round(price - prev["close"], 5)
    chg_str   = f"+{price_chg}" if price_chg > 0 else str(price_chg)

    regime, adx_val, direction_hint = classify_regime(df)
    diff, rate_bias, rate_detail    = get_rate_differential(pair)
    comm_align, comm_detail         = get_commodity_alignment(comm_df, "BUY", pair)

    result = {
        "pair":        pair,
        "price":       price,
        "price_dir":   price_dir,
        "price_chg":   chg_str,
        "e50":         e50,
        "e50_dir":     e50_dir,
        "e200":        e200,
        "e200_dir":    e200_dir,
        "rsi":         rsi,
        "rsi_dir":     rsi_dir,
        "atr":         atr,
        "adx":         adx,
        "pdi":         pdi,
        "ndi":         ndi,
        "regime":      regime,
        "direction_hint": direction_hint,
        "rate_bias":   rate_bias,
        "rate_detail": rate_detail,
        "signal":      "NO_SIGNAL",
        "strategy":    None,
        "direction":   None,
        "entry":       None,
        "stop":        None,
        "target":      None,
        "lots":        None,
        "effective_risk": None,
        "modifiers":   [],
        "rr":          None,
        "range_high":  None,
        "range_low":   None,
        "range_mid":   None,
        "h4_checks":   0,
        "h4_rsi":      None,
        "comm_align":  comm_align,
        "comm_detail": comm_detail,
        "comm_label":  cfg["comm_label"],
    }

    if regime == "TRENDING":
        sig, h4c, h4rsi = strategy_a_signal(df, df_4h)
        result["strategy"]  = "A — Trend Following"
        result["h4_checks"] = h4c
        result["h4_rsi"]    = h4rsi

        if sig in ("BUY", "BUY_NO4H", "SELL", "SELL_NO4H"):
            direction = "BUY" if "BUY" in sig else "SELL"
            result["direction"] = direction
            ca, cd = get_commodity_alignment(comm_df, direction, pair)
            result["comm_align"]  = ca
            result["comm_detail"] = cd

            lots, eff_risk, mods = calculate_position_size(
                l["atr"], price, pair, size_modifier, ca, rate_bias, direction
            )

            if direction == "BUY":
                stop   = round(price - ATR_STOP_MULT * l["atr"], 5)
                target = round(price + 2.5  * ATR_STOP_MULT * l["atr"], 5)
            else:
                stop   = round(price + ATR_STOP_MULT * l["atr"], 5)
                target = round(price - 2.5  * ATR_STOP_MULT * l["atr"], 5)

            rr     = round(abs(target - price) / abs(stop - price), 1)
            status = sig if sig in ("BUY", "SELL") else f"{sig} (4H blocked)"

            result.update({
                "signal":         status,
                "entry":          price,
                "stop":           stop,
                "target":         target,
                "lots":           lots,
                "effective_risk": eff_risk,
                "modifiers":      mods,
                "rr":             rr,
            })
        else:
            result["signal"] = sig

    elif regime == "RANGING":
        sig, h4c, h4rsi, rh, rl, rm = strategy_b_signal(df, df_4h)
        result["strategy"]   = "B — Mean Reversion"
        result["h4_checks"]  = h4c
        result["h4_rsi"]     = h4rsi
        result["range_high"] = round(rh, 5)
        result["range_low"]  = round(rl, 5)
        result["range_mid"]  = round(rm, 5)

        if sig in ("BUY", "BUY_NO4H", "SELL", "SELL_NO4H"):
            direction = "BUY" if "BUY" in sig else "SELL"
            result["direction"] = direction
            ca, cd = get_commodity_alignment(comm_df, direction, pair)
            result["comm_align"]  = ca
            result["comm_detail"] = cd

            lots, eff_risk, mods = calculate_position_size(
                l["atr"], price, pair, size_modifier, ca, rate_bias, direction
            )

            if direction == "BUY":
                stop   = round(rl - l["atr"], 5)
                target = round(rm, 5)
            else:
                stop   = round(rh + l["atr"], 5)
                target = round(rm, 5)

            rr     = round(abs(target - price) / max(abs(stop - price), 0.00001), 1)
            status = sig if sig in ("BUY", "SELL") else f"{sig} (4H blocked)"

            result.update({
                "signal":         status,
                "entry":          price,
                "stop":           stop,
                "target":         target,
                "lots":           lots,
                "effective_risk": eff_risk,
                "modifiers":      mods,
                "rr":             rr,
            })
        else:
            result["signal"] = sig

    else:
        result["strategy"] = "NONE — Ambiguous regime"
        result["signal"]   = "STAND DOWN — Ambiguous ADX"

    return result


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def build_and_send_email(pair_results, events, time_filter_msg, size_modifier):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    def sig_emoji(sig):
        if "BUY" in sig and "BIAS" not in sig and "NO4H" not in sig: return "🟢"
        if "SELL" in sig and "BIAS" not in sig and "NO4H" not in sig: return "🔴"
        if "BIAS" in sig or "NO4H" in sig: return "🟡"
        if "STAND DOWN" in sig: return "⛔"
        return "⚪"

    # Count actionable signals
    confirmed = [r for r in pair_results if r["signal"] in ("BUY","SELL")]
    watch     = [r for r in pair_results if "BIAS" in r["signal"] or "NO4H" in r["signal"]]

    # News block
    news_block = ""
    if events:
        ev_lines   = "\n".join(f"  ⚠️  {e.get('event','?')} ({e.get('country','?')})" for e in events)
        news_block = f"""
╔══════════════════════════════════════════════════════════════╗
  ⛔ HIGH IMPACT NEWS TODAY — VERIFY BEFORE TRADING
{ev_lines}
╚══════════════════════════════════════════════════════════════╝
"""

    # Time filter
    tf_block = ""
    if time_filter_msg:
        tf_block = f"\n⚠️  TIME FILTER: {time_filter_msg}"
    if size_modifier < 1.0:
        tf_block += f"\n⚠️  POSITION SIZE: Reduced to {int(size_modifier*100)}% this period"

    # Pair blocks
    pair_blocks = ""
    for r in pair_results:
        emoji  = sig_emoji(r["signal"])
        regime_icon = {"TRENDING": "📈", "RANGING": "➡️", "AMBIGUOUS": "⚠️ "}.get(r["regime"], "")
        comm_icon   = {"ALIGNED": "✅", "NEUTRAL": "—", "CONTRARY": "❌"}.get(r["comm_align"], "—")
        rate_icon   = {"BULLISH": "✅", "NEUTRAL": "—", "BEARISH": "❌"}.get(r["rate_bias"], "—")
        h4_icon     = "✅" if r["h4_checks"] >= 2 else "❌"

        entry_block = ""
        if r["signal"] in ("BUY","SELL"):
            direction  = r["direction"]
            mod_str    = "  " + "\n  ".join(r["modifiers"]) if r["modifiers"] else "  None — full size"
            entry_block = f"""
  ┌─────────────────────────────────────┐
  │  {direction} SIGNAL — EXECUTE             │
  ├─────────────────────────────────────┤
  │  Entry  : {str(r['entry']):<27}│
  │  Stop   : {str(r['stop']):<27}│
  │  Target : {str(r['target']):<27}│
  │  R:R    : 1:{str(r['rr']):<26}│
  │  Size   : {str(r['lots']):<5} lots  (${r['effective_risk']:,.0f} risk)   │
  └─────────────────────────────────────┘
  Size modifiers:
{mod_str}"""
        elif "BIAS" in r["signal"] or "NO4H" in r["signal"]:
            entry_block = f"\n  ⏳ {r['signal']} — Wait for full conditions"
        elif "STAND DOWN" in r["signal"]:
            entry_block = f"\n  ⛔ {r['signal']}"
        else:
            entry_block = f"\n  ⚪ No setup today"

        range_block = ""
        if r["range_high"]:
            range_block = f"""
  Range    : {r['range_low']} — {r['range_high']}  ({RANGE_LOOKBACK}-day)
  Mid      : {r['range_mid']}"""

        pair_blocks += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {emoji}  {r['pair'].replace('_','/')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Regime   : {regime_icon} {r['regime']}  (ADX {r['adx']} | +DI {r['pdi']} / -DI {r['ndi']})
  Strategy : {r['strategy']}
{entry_block}

  ─── Market Snapshot ────────────────────────────────────────
  Price    : {r['price']} {r['price_dir']}  ({r['price_chg']} today)
  50 EMA   : {r['e50']} {r['e50_dir']}
  200 EMA  : {r['e200']} {r['e200_dir']}
  RSI (14) : {r['rsi']} {r['rsi_dir']}
  ATR (14) : {r['atr']}
{range_block}

  ─── Confluence Filters ─────────────────────────────────────
  {h4_icon}  4H Confirm  : {r['h4_checks']}/3 checks  (4H RSI {r['h4_rsi']})
  {comm_icon}  {r['comm_label']:<10}  : {r['comm_align']}  — {r['comm_detail']}
  {rate_icon}  Rate Diff   : {r['rate_bias']}  — {r['rate_detail']}
"""

    summary = f"{len(confirmed)} confirmed | {len(watch)} watching | {4-len(confirmed)-len(watch)} no signal"

    body = f"""DAILY FOREX SIGNAL REPORT
{ny_time}
════════════════════════════════════════════════════════════════
{summary}
{tf_block}
{news_block}
{pair_blocks}
════════════════════════════════════════════════════════════════
STRATEGY GUIDE
  Strategy A (Trending)  : ADX > 25 | Price touches 50 EMA | 4H confirms
  Strategy B (Ranging)   : ADX < 20 | Price at range boundary | RSI extreme
  Ambiguous ADX 20-25    : Stand down — no edge
  Size modifiers         : Contrary commodity × 0.5 | Contrary rates × 0.5

RISK RULES
  Max open trades        : 4 (one per pair)
  Max same-direction     : 2 (no full USD stack)
  Risk per trade         : $2,000 base (modified by filters)
  No trades              : Fridays | Dec 18–Jan 5 | 3+ news events

────────────────────────────────────────────────────────────────
Multi-Pair Trading System v2.0 | Regime-Adaptive"""

    subject = f"Forex Signals: {summary} | {ny_date}"

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


# ─── MAIN JOB ────────────────────────────────────────────────────────────────
def run_signal_job():
    print(f"\n{'='*50}")
    print(f"Signal job at {datetime.now(timezone.utc)} UTC")
    print(f"{'='*50}")

    try:
        # Time filter
        can_trade, size_modifier, tf_msg = time_filter()
        print(f"Time filter: {'OK' if can_trade else 'BLOCKED'}  {tf_msg}")
        if not can_trade:
            print("Skipping — time filter blocked trading today")
            # Still send email to notify
            size_modifier = 0.0

        # Economic calendar
        print("Fetching economic calendar...")
        events = fetch_economic_events()

        # Fetch commodity data (shared across pairs)
        print("Fetching commodity data...")
        try:
            gold_df = fetch_candles("XAU_USD", "D", 250)
            gold_df = calculate_indicators(gold_df)
        except Exception as e:
            print(f"Gold fetch failed: {e}")
            gold_df = None

        try:
            oil_df = fetch_candles("WTICO_USD", "D", 250)
            oil_df = calculate_indicators(oil_df)
        except Exception as e:
            print(f"Oil fetch failed: {e}")
            oil_df = None

        # Analyse each pair
        pair_results = []
        for pair in PAIRS:
            print(f"Analysing {pair}...")
            try:
                df    = fetch_candles(pair, "D", 250)
                df    = calculate_indicators(df)
                df_4h = fetch_candles(pair, "H4", 300)
                df_4h = calculate_indicators(df_4h)

                # Assign correct commodity
                comm_df = gold_df if PAIRS[pair]["commodity"] == "XAU_USD" else oil_df

                result = analyse_pair(pair, df, df_4h, comm_df,
                                      size_modifier if can_trade else 0.0)
                pair_results.append(result)
                print(f"  {pair}: {result['regime']} | {result['signal']}")

            except Exception as e:
                print(f"  {pair} failed: {e}")
                pair_results.append({
                    "pair": pair, "signal": "ERROR", "strategy": f"Error: {e}",
                    "regime": "ERROR", "price": 0, "price_dir": "", "price_chg": "",
                    "e50": 0, "e50_dir": "", "e200": 0, "e200_dir": "",
                    "rsi": 0, "rsi_dir": "", "atr": 0, "adx": 0, "pdi": 0, "ndi": 0,
                    "direction_hint": "", "rate_bias": "", "rate_detail": "",
                    "comm_align": "", "comm_detail": "", "comm_label": "",
                    "direction": None, "entry": None, "stop": None, "target": None,
                    "lots": None, "effective_risk": None, "modifiers": [], "rr": None,
                    "range_high": None, "range_low": None, "range_mid": None,
                    "h4_checks": 0, "h4_rsi": None,
                })

        print("Sending email...")
        build_and_send_email(pair_results, events, tf_msg, size_modifier)

    except Exception as e:
        print(f"Job error: {e}")


# ─── SCHEDULER ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Pair Forex Signal System v2.0")
    print("Pairs: GBP/USD | EUR/USD | AUD/USD | USD/JPY")
    print("Scheduled: 5:00 PM EST (NY close) daily")
    print()

    run_signal_job()

    schedule.every().day.at("22:00").do(run_signal_job)  # 22:00 UTC = 17:00 EST

    while True:
        schedule.run_pending()
        time.sleep(60)
