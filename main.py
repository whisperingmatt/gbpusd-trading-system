import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import schedule
import pytz

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OANDA_API_KEY    = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
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

INSTRUMENT = "GBP_USD"


# ─── FETCH CANDLES (reusable for any granularity) ────────────────────────────
def fetch_candles(granularity="D", count=250):
    url     = f"{OANDA_BASE}/instruments/{INSTRUMENT}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params  = {"granularity": granularity, "count": count, "price": "M"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    rows = []
    for c in response.json()["candles"]:
        if c["complete"]:
            rows.append({
                "time":   c["time"][:19],
                "open":   float(c["mid"]["o"]),
                "high":   float(c["mid"]["h"]),
                "low":    float(c["mid"]["l"]),
                "close":  float(c["mid"]["c"]),
                "volume": int(c["volume"]),
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

    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()

    return df


# ─── 4H CONFIRMATION ─────────────────────────────────────────────────────────
def get_4h_confirmation(df_4h):
    """
    Checks the most recent 4H candle for directional confirmation.
    Returns: 'BULL', 'BEAR', or 'NEUTRAL' plus a detail string.

    Rules (need 2 of 3 to confirm):
      1. Price vs 4H 50 EMA
      2. 4H 50 EMA vs 4H 200 EMA
      3. 4H RSI in favourable zone
    """
    latest   = df_4h.iloc[-1]
    previous = df_4h.iloc[-2]

    price    = latest["close"]
    ema20_4h = latest["ema20"]
    ema50_4h = latest["ema50"]
    ema200_4h = latest["ema200"]
    rsi_4h   = latest["rsi"]

    ema20_dir = "▲" if ema20_4h > previous["ema20"] else "▼"
    ema50_dir = "▲" if ema50_4h > previous["ema50"] else "▼"

    bull_checks = 0
    bear_checks = 0

    # Check 1: Price vs 4H 50 EMA
    if price > ema50_4h:
        bull_checks += 1
    else:
        bear_checks += 1

    # Check 2: 4H 50 EMA vs 4H 200 EMA
    if ema50_4h > ema200_4h:
        bull_checks += 1
    else:
        bear_checks += 1

    # Check 3: 4H RSI
    if 40 <= rsi_4h <= 65:
        bull_checks += 1
    elif rsi_4h < 35:
        bull_checks += 1
    elif rsi_4h > 68:
        bear_checks += 1

    if bull_checks >= 2:
        direction = "BULL"
        detail    = f"4H EMA {ema50_dir}: {round(ema50_4h,5)} | 4H RSI: {round(rsi_4h,1)} | {bull_checks}/3 bull checks"
    elif bear_checks >= 2:
        direction = "BEAR"
        detail    = f"4H EMA {ema50_dir}: {round(ema50_4h,5)} | 4H RSI: {round(rsi_4h,1)} | {bear_checks}/3 bear checks"
    else:
        direction = "NEUTRAL"
        detail    = f"4H EMA {ema50_dir}: {round(ema50_4h,5)} | 4H RSI: {round(rsi_4h,1)} | Mixed — no clear direction"

    return direction, detail, round(ema50_4h,5), round(rsi_4h,1)


# ─── LONG-TERM TREND (200 EMA) ───────────────────────────────────────────────
def classify_long_term_trend(df):
    price        = df.iloc[-1]["close"]
    ema200_now   = df["ema200"].iloc[-1]
    ema200_20ago = df["ema200"].iloc[-20]
    ema200_50ago = df["ema200"].iloc[-50]
    slope_20     = (ema200_now - ema200_20ago) / ema200_20ago * 100
    slope_50     = (ema200_now - ema200_50ago) / ema200_50ago * 100

    high_200  = df["high"].iloc[-200:].max()
    low_200   = df["low"].iloc[-200:].min()
    range_200 = high_200 - low_200
    range_mid = low_200 + range_200 * 0.5
    range_pct = range_200 / ema200_now * 100

    is_wide   = range_pct > 8.0
    is_flat   = abs(slope_50) < 0.3
    in_upper  = price > range_mid

    if is_wide and is_flat:
        position = "upper half" if in_upper else "lower half"
        return "SIDEWAYS", f"Channel {round(low_200,4)}-{round(high_200,4)} | In {position}"
    elif price > ema200_now and slope_20 > 0.05 and slope_50 > 0.1:
        return "BULLISH", "200 EMA rising | Price above"
    elif price < ema200_now and slope_20 < -0.05 and slope_50 < -0.1:
        return "BEARISH", "200 EMA falling | Price below"
    elif is_wide and in_upper:
        return "SIDEWAYS", f"Channel {round(low_200,4)}-{round(high_200,4)} | In upper half"
    else:
        return "SIDEWAYS", f"Channel {round(low_200,4)}-{round(high_200,4)} | In lower half"


# ─── SHORT-TERM TREND (20 EMA) ───────────────────────────────────────────────
def classify_short_term_trend(df):
    price       = df.iloc[-1]["close"]
    ema20_now   = df["ema20"].iloc[-1]
    ema20_5ago  = df["ema20"].iloc[-5]
    ema20_slope = (ema20_now - ema20_5ago) / ema20_5ago * 100

    if price > ema20_now and ema20_slope > 0.02:
        return "BULLISH", f"Price above rising 20 EMA ({round(ema20_now,5)})"
    elif price < ema20_now and ema20_slope < -0.02:
        return "BEARISH", f"Price below falling 20 EMA ({round(ema20_now,5)})"
    else:
        return "NEUTRAL", f"Price near 20 EMA ({round(ema20_now,5)}) — no clear direction"


# ─── KEY LEVELS & BREAK-AND-RETEST ───────────────────────────────────────────
def detect_key_levels_and_bnr(df, lookback=200, tolerance_atr=0.75):
    price  = df.iloc[-1]["close"]
    atr    = df.iloc[-1]["atr"]
    tol    = tolerance_atr * atr
    recent = df.iloc[-lookback:].reset_index()
    window = 5

    swing_highs = []
    swing_lows  = []

    for i in range(window, len(recent) - window):
        hi = recent.iloc[i]["high"]
        lo = recent.iloc[i]["low"]
        if hi == recent.iloc[i-window:i+window+1]["high"].max():
            swing_highs.append(round(hi, 5))
        if lo == recent.iloc[i-window:i+window+1]["low"].min():
            swing_lows.append(round(lo, 5))

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = []
        group = [levels[0]]
        for lv in levels[1:]:
            if lv - group[-1] <= 0.5 * atr:
                group.append(lv)
            else:
                clustered.append(round(sum(group) / len(group), 5))
                group = [lv]
        clustered.append(round(sum(group) / len(group), 5))
        return clustered

    resistance = cluster([h for h in swing_highs if h > price])
    support    = cluster([l for l in swing_lows  if l < price])

    nearest_r = min(resistance, key=lambda x: abs(x - price)) if resistance else None
    nearest_s = max(support,    key=lambda x: abs(x - price)) if support    else None

    bnr_bull = None
    bnr_bear = None
    last_20  = df.iloc[-20:]
    all_lvls = swing_highs + swing_lows

    for level in set([round(l, 5) for l in all_lvls]):
        above = (last_20["close"] > level).sum()
        below = (last_20["close"] < level).sum()

        if above >= 3 and below >= 2 and price > level and abs(price - level) <= tol:
            if bnr_bull is None or abs(price - level) < abs(price - bnr_bull):
                bnr_bull = level
        if below >= 3 and above >= 2 and price < level and abs(price - level) <= tol:
            if bnr_bear is None or abs(price - level) < abs(price - bnr_bear):
                bnr_bear = level

    return {
        "nearest_resistance": nearest_r,
        "nearest_support":    nearest_s,
        "bnr_bull":           bnr_bull,
        "bnr_bear":           bnr_bear,
    }


# ─── FIBONACCI ───────────────────────────────────────────────────────────────
def calculate_fibonacci(df, lookback=50):
    recent     = df.iloc[-lookback:]
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    diff       = swing_high - swing_low
    price      = df.iloc[-1]["close"]
    atr        = df.iloc[-1]["atr"]

    fib50  = round(swing_high - 0.500 * diff, 5)
    fib618 = round(swing_high - 0.618 * diff, 5)

    return {
        "swing_high":  round(swing_high, 5),
        "swing_low":   round(swing_low, 5),
        "fib50":       fib50,
        "fib618":      fib618,
        "near_fib50":  abs(price - fib50)  <= 0.5 * atr,
        "near_fib618": abs(price - fib618) <= 0.5 * atr,
    }


# ─── FAIR VALUE GAP ──────────────────────────────────────────────────────────
def detect_fvg(df, lookback=30):
    recent = df.iloc[-(lookback + 2):].reset_index()
    price  = df.iloc[-1]["close"]
    atr    = df.iloc[-1]["atr"]

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(len(recent) - 2):
        c1 = recent.iloc[i]
        c3 = recent.iloc[i + 2]

        if c3["low"] > c1["high"] and price > c1["high"]:
            bullish_fvgs.append({
                "date":     str(recent.iloc[i + 1]["time"])[:10],
                "gap_high": round(float(c3["low"]), 5),
                "gap_low":  round(float(c1["high"]), 5),
            })
        if c3["high"] < c1["low"] and price < c1["low"]:
            bearish_fvgs.append({
                "date":     str(recent.iloc[i + 1]["time"])[:10],
                "gap_high": round(float(c1["low"]), 5),
                "gap_low":  round(float(c3["high"]), 5),
            })

    nearest_bull = bullish_fvgs[-1] if bullish_fvgs else None
    nearest_bear = bearish_fvgs[-1] if bearish_fvgs else None

    bull_active = (
        nearest_bull is not None and
        nearest_bull["gap_low"] - 0.5*atr <= price <= nearest_bull["gap_high"] + 0.5*atr
    )
    bear_active = (
        nearest_bear is not None and
        nearest_bear["gap_low"] - 0.5*atr <= price <= nearest_bear["gap_high"] + 0.5*atr
    )

    return {
        "bullish_fvg":    nearest_bull,
        "bearish_fvg":    nearest_bear,
        "bullish_active": bull_active,
        "bearish_active": bear_active,
    }


# ─── ECONOMIC CALENDAR ───────────────────────────────────────────────────────
def fetch_economic_events():
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    url      = (
        f"https://finnhub.io/api/v1/calendar/economic"
        f"?from={today}&to={tomorrow}&token={FINNHUB_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json().get("economicCalendar", [])
        return [
            e for e in events
            if e.get("impact") == "high"
            and e.get("country") in ("US", "GB")
        ]
    except Exception as e:
        print(f"Calendar fetch error: {e}")
        return []


# ─── SCORECARD ───────────────────────────────────────────────────────────────
def build_scorecard(df, lt_trend, st_trend, h4_conf, h4_detail,
                    h4_ema50, h4_rsi, levels, fib, fvg):
    """
    Scoring:
      Long-term trend    : 3 pts  (SIDEWAYS = 0)
      Short-term trend   : 2 pts  (NEUTRAL  = 0)
      Price vs 50 EMA    : 2 pts
      50 EMA vs 200 EMA  : 2 pts
      4H confirmation    : 2 pts  (NEUTRAL  = 0)
      Break-and-retest   : 3 pts
      FVG active         : 2 pts
      Fib 0.618          : 2 pts  (confluence bonus)
      Fib 0.500          : 1 pt   (confluence bonus)
      RSI                : 1 pt
      ATR                : info only

    CONFIRMED >= 8 | WATCH 5-7 | NO TRADE < 5
    """
    latest   = df.iloc[-1]
    previous = df.iloc[-2]

    price  = latest["close"]
    ema20  = latest["ema20"]
    ema50  = latest["ema50"]
    ema200 = latest["ema200"]
    rsi    = latest["rsi"]
    atr    = latest["atr"]

    prev_close  = previous["close"]
    prev_ema20  = previous["ema20"]
    prev_ema50  = previous["ema50"]
    prev_ema200 = previous["ema200"]
    prev_rsi    = previous["rsi"]
    prev_atr    = previous["atr"]

    price_dir  = "▲" if price  > prev_close  else "▼"
    ema20_dir  = "▲" if ema20  > prev_ema20  else "▼"
    ema50_dir  = "▲" if ema50  > prev_ema50  else "▼"
    ema200_dir = "▲" if ema200 > prev_ema200 else "▼"
    rsi_dir    = "▲" if rsi    > prev_rsi    else "▼"
    atr_dir    = "▲" if atr    > prev_atr    else "▼"

    ema50_zone_low  = round(ema50 - 0.75 * atr, 5)
    ema50_zone_high = round(ema50 + 0.75 * atr, 5)

    rows       = []
    bull_score = 0
    bear_score = 0

    # ── TREND ──────────────────────────────────────────────────────────────
    lt_icon = {"BULLISH": "📈", "BEARISH": "📉", "SIDEWAYS": "➡️"}.get(lt_trend, "➡️")
    if lt_trend == "BULLISH":
        rows.append(("✅", f"Long-Term {ema200_dir}", f"{lt_icon} BULLISH", "Price > rising 200 EMA"))
        bull_score += 3
    elif lt_trend == "BEARISH":
        rows.append(("❌", f"Long-Term {ema200_dir}", f"{lt_icon} BEARISH", "Price < falling 200 EMA"))
        bear_score += 3
    else:
        rows.append(("⚠️ ", f"Long-Term {ema200_dir}", f"{lt_icon} SIDEWAYS", "Range market — BNR entries preferred"))

    st_icon = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(st_trend, "➡️")
    if st_trend == "BULLISH":
        rows.append(("✅", f"Short-Term {ema20_dir}", f"{st_icon} BULLISH", f"20 EMA: {round(ema20,5)}  Price above & rising"))
        bull_score += 2
    elif st_trend == "BEARISH":
        rows.append(("❌", f"Short-Term {ema20_dir}", f"{st_icon} BEARISH", f"20 EMA: {round(ema20,5)}  Price below & falling"))
        bear_score += 2
    else:
        rows.append(("⚠️ ", f"Short-Term {ema20_dir}", f"{st_icon} NEUTRAL", f"20 EMA: {round(ema20,5)}  No clear direction"))

    # ── 4H CONFIRMATION ────────────────────────────────────────────────────
    rows.append(("──", "SEP", "", ""))

    h4_icon = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➡️"}.get(h4_conf, "➡️")
    if h4_conf == "BULL":
        rows.append(("✅", "4H Confirmation", f"{h4_icon} BULLISH", h4_detail))
        bull_score += 2
    elif h4_conf == "BEAR":
        rows.append(("❌", "4H Confirmation", f"{h4_icon} BEARISH", h4_detail))
        bear_score += 2
    else:
        rows.append(("⚠️ ", "4H Confirmation", f"{h4_icon} NEUTRAL", h4_detail))

    # ── PRICE & EMAS ───────────────────────────────────────────────────────
    rows.append(("──", "SEP", "", ""))

    price_change = round(price - prev_close, 5)
    change_str   = f"+{price_change}" if price_change > 0 else str(price_change)
    rows.append(("  ", f"Current Price {price_dir}", str(round(price, 5)), f"{change_str} from yesterday"))

    if price > ema50:
        rows.append(("✅", f"50 EMA {ema50_dir}", str(round(ema50, 5)), f"Entry zone {ema50_zone_low}-{ema50_zone_high}"))
        bull_score += 2
    else:
        rows.append(("❌", f"50 EMA {ema50_dir}", str(round(ema50, 5)), f"Entry zone {ema50_zone_low}-{ema50_zone_high}"))
        bear_score += 2

    if ema50 > ema200:
        rows.append(("✅", f"200 EMA {ema200_dir}", str(round(ema200, 5)), "50 EMA above = bull structure"))
        bull_score += 2
    else:
        rows.append(("❌", f"200 EMA {ema200_dir}", str(round(ema200, 5)), "50 EMA below = bear structure"))
        bear_score += 2

    rsi_val = f"{round(rsi,1)}"
    if 40 <= rsi <= 65:
        rows.append(("✅", f"RSI (14) {rsi_dir}", rsi_val, "Ideal range 40-65"))
        bull_score += 1
    elif rsi < 30:
        rows.append(("✅", f"RSI (14) {rsi_dir}", rsi_val, "Oversold — bullish reversal zone"))
        bull_score += 1
    elif rsi > 70:
        rows.append(("❌", f"RSI (14) {rsi_dir}", rsi_val, "Overbought — avoid buying here"))
        bear_score += 1
    elif 30 <= rsi < 40:
        rows.append(("⚠️ ", f"RSI (14) {rsi_dir}", rsi_val, "Approaching oversold"))
    else:
        rows.append(("⚠️ ", f"RSI (14) {rsi_dir}", rsi_val, "Neutral zone"))

    atr_note = "Expanding — wider stops" if atr > prev_atr else "Contracting — tighter stops"
    rows.append(("ℹ️ ", f"ATR (14) {atr_dir}", str(round(atr, 5)), atr_note))

    # ── KEY LEVELS & BNR ───────────────────────────────────────────────────
    rows.append(("──", "SEP", "", ""))

    nr = levels["nearest_resistance"]
    ns = levels["nearest_support"]
    rows.append(("  ", "Nearest Resistance", str(nr) if nr else "None", "Watch for rejection or break"))
    rows.append(("  ", "Nearest Support",    str(ns) if ns else "None", "Watch for bounce or break"))

    if levels["bnr_bull"]:
        rows.append(("✅", "Break & Retest", f"BULLISH at {levels['bnr_bull']}", "Broken resistance = new support — high conviction"))
        bull_score += 3
    elif levels["bnr_bear"]:
        rows.append(("❌", "Break & Retest", f"BEARISH at {levels['bnr_bear']}", "Broken support = new resistance — high conviction"))
        bear_score += 3
    else:
        rows.append(("—", "Break & Retest", "None active", "No confirmed BNR setup"))

    # ── FVG ────────────────────────────────────────────────────────────────
    rows.append(("──", "SEP", "", ""))

    if fvg["bullish_active"]:
        g = fvg["bullish_fvg"]
        rows.append(("✅", "Fair Value Gap", "BULLISH ACTIVE", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
        bull_score += 2
    elif fvg["bearish_active"]:
        g = fvg["bearish_fvg"]
        rows.append(("❌", "Fair Value Gap", "BEARISH ACTIVE", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
        bear_score += 2
    elif fvg["bullish_fvg"]:
        g = fvg["bullish_fvg"]
        rows.append(("—", "Fair Value Gap", "Bullish unfilled", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
    elif fvg["bearish_fvg"]:
        g = fvg["bearish_fvg"]
        rows.append(("—", "Fair Value Gap", "Bearish unfilled", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
    else:
        rows.append(("—", "Fair Value Gap", "None detected", ""))

    # ── FIBONACCI ──────────────────────────────────────────────────────────
    rows.append(("──", "SEP", "", ""))

    if fib["near_fib618"]:
        rows.append(("✅", "Fib 0.618", str(fib["fib618"]), "CONFLUENCE — price at key retracement"))
        bull_score += 2
        bear_score += 2
    else:
        rows.append(("—", "Fib 0.618", str(fib["fib618"]), "Key retracement level"))

    if fib["near_fib50"]:
        rows.append(("✅", "Fib 0.500", str(fib["fib50"]), "CONFLUENCE — price at mid retracement"))
        bull_score += 1
        bear_score += 1
    else:
        rows.append(("—", "Fib 0.500", str(fib["fib50"]), "Mid retracement level"))

    rows.append(("  ", "Swing High", str(fib["swing_high"]), "50-candle high"))
    rows.append(("  ", "Swing Low",  str(fib["swing_low"]),  "50-candle low"))

    return rows, bull_score, bear_score


# ─── SIGNAL ENGINE ───────────────────────────────────────────────────────────
def generate_signal(df, df_4h, events):
    latest = df.iloc[-1]
    price  = latest["close"]
    ema50  = latest["ema50"]
    atr    = latest["atr"]

    lt_trend, lt_detail         = classify_long_term_trend(df)
    st_trend, st_detail         = classify_short_term_trend(df)
    h4_conf, h4_detail, h4_ema50, h4_rsi = get_4h_confirmation(df_4h)
    levels                      = detect_key_levels_and_bnr(df)
    fib                         = calculate_fibonacci(df)
    fvg                         = detect_fvg(df)
    scorecard, bull_score, bear_score = build_scorecard(
        df, lt_trend, st_trend, h4_conf, h4_detail,
        h4_ema50, h4_rsi, levels, fib, fvg
    )

    news_today = len(events) > 0

    # 4H must not contradict the daily signal
    if bull_score >= 8:
        if h4_conf == "BEAR":
            raw        = "WATCH — BUY BIAS (4H CONTRADICTS)"
            raw_detail = f"Daily score {bull_score}/10 but 4H is bearish. Wait for 4H to align."
        else:
            raw        = "CONFIRMED BUY"
            raw_detail = f"Score {bull_score}/10 — Daily + 4H aligned. Limit at 50 EMA ({round(ema50,5)})."
    elif bear_score >= 8:
        if h4_conf == "BULL":
            raw        = "WATCH — SELL BIAS (4H CONTRADICTS)"
            raw_detail = f"Daily score {bear_score}/10 but 4H is bullish. Wait for 4H to align."
        else:
            raw        = "CONFIRMED SELL"
            raw_detail = f"Score {bear_score}/10 — Daily + 4H aligned. Limit at 50 EMA ({round(ema50,5)})."
    elif bull_score >= 5:
        raw        = "WATCH — BUY SETUP FORMING"
        raw_detail = f"Score {bull_score}/10 — Not enough confluence yet. Monitor."
    elif bear_score >= 5:
        raw        = "WATCH — SELL SETUP FORMING"
        raw_detail = f"Score {bear_score}/10 — Not enough confluence yet. Monitor."
    else:
        raw        = "NO TRADE"
        raw_detail = f"Bull {bull_score}/10  Bear {bear_score}/10 — No edge. Stay out."

    if news_today:
        action        = "STAND DOWN — HIGH IMPACT NEWS TODAY"
        action_detail = f"Underlying setup: {raw} — Do NOT trade into news. Reassess tomorrow."
    else:
        action        = raw
        action_detail = raw_detail

    return {
        "action":        action,
        "action_detail": action_detail,
        "bull_score":    bull_score,
        "bear_score":    bear_score,
        "price":         round(price, 5),
        "ema50":         round(ema50, 5),
        "ema200":        round(df.iloc[-1]["ema200"], 5),
        "rsi":           round(df.iloc[-1]["rsi"], 2),
        "atr":           round(atr, 5),
        "fib50":         fib["fib50"],
        "fib618":        fib["fib618"],
        "stop_buy":      round(price - 1.5 * atr, 5),
        "target_buy":    round(price + 3.0 * atr, 5),
        "stop_sell":     round(price + 1.5 * atr, 5),
        "target_sell":   round(price - 3.0 * atr, 5),
        "lt_trend":      lt_trend,
        "st_trend":      st_trend,
        "h4_conf":       h4_conf,
        "scorecard":     scorecard,
        "news_warning":  news_today,
        "events":        [e.get("event", "Unknown") for e in events],
    }


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def build_and_send_email(data):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    if   "CONFIRMED BUY"  in data["action"]: action_emoji = "🟢"
    elif "CONFIRMED SELL" in data["action"]: action_emoji = "🔴"
    elif "WATCH"          in data["action"]: action_emoji = "🟡"
    elif "STAND DOWN"     in data["action"]: action_emoji = "⛔"
    else:                                    action_emoji = "⚪"

    col1 = 5
    col2 = 22
    col3 = 16
    col4 = 38

    divider = "  " + "-" * (col1 + col2 + col3 + col4) + "\n"
    table   = divider
    table  += f"  {'':>{col1}}  {'INDICATOR':<{col2}}  {'CURRENT':<{col3}}  {'IDEAL RANGE / INFO':<{col4}}\n"
    table  += divider

    for status, indicator, current, info in data["scorecard"]:
        if indicator == "SEP":
            table += divider
            continue
        table += f"  {status:<{col1}}  {indicator:<{col2}}  {current:<{col3}}  {info:<{col4}}\n"

    table += divider
    table += f"  {'':>{col1}}  {'BULL SCORE':<{col2}}  {str(data['bull_score']) + '/10':<{col3}}\n"
    table += f"  {'':>{col1}}  {'BEAR SCORE':<{col2}}  {str(data['bear_score']) + '/10':<{col3}}\n"
    table += divider

    news_block = ""
    if data["news_warning"]:
        event_lines = "\n".join(f"    {e}" for e in data["events"])
        news_block  = f"""
------------------------------------------------------------
⛔ HIGH IMPACT NEWS TODAY — DO NOT TRADE
------------------------------------------------------------
{event_lines}
"""

    body = f"""GBP/USD DAILY SIGNAL REPORT
{ny_time}
============================================================

{action_emoji} {data['action']}
   {data['action_detail']}
{news_block}
------------------------------------------------------------
MARKET SNAPSHOT & SCORECARD
------------------------------------------------------------
{table}
------------------------------------------------------------
RISK LEVELS  (CONFIRMED signals only)
------------------------------------------------------------
  Entry  : Limit order at 50 EMA ({data['ema50']})
           or at Break-and-Retest level if BNR active

  IF BUYING:
    Stop   : {data['stop_buy']}
    Target : {data['target_buy']}
    R:R    : 1:2

  IF SELLING:
    Stop   : {data['stop_sell']}
    Target : {data['target_sell']}
    R:R    : 1:2

------------------------------------------------------------
SCORING GUIDE
------------------------------------------------------------
  🟢 CONFIRMED  8+/10  Execute — daily + 4H aligned
  🟡 WATCH      5-7    Setup forming — do not enter yet
  ⚪ NO TRADE   0-4    No edge — stay out
  ⛔ STAND DOWN        News day — wait regardless of score

  4H confirmation required — daily score alone is not enough
  BNR + FVG + Fib confluence = maximum conviction entry

------------------------------------------------------------
GBP/USD Trading System v1.6"""

    subject = f"GBP/USD: {action_emoji} {data['action']} | Bull {data['bull_score']}/10  Bear {data['bear_score']}/10 | {ny_date}"

    response = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "personalizations": [{"to": [{"email": EMAIL_RECIPIENT}]}],
            "from":    {"email": EMAIL_SENDER},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        }
    )
    response.raise_for_status()
    print(f"Email sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─── MAIN JOB ────────────────────────────────────────────────────────────────
def run_signal_job():
    print(f"\n{'='*40}")
    print(f"Running signal job at {datetime.now(timezone.utc)} UTC")
    print(f"{'='*40}")
    try:
        print("Fetching daily candles...")
        df = fetch_candles(granularity="D", count=250)
        df = calculate_indicators(df)

        print("Fetching 4H candles...")
        df_4h = fetch_candles(granularity="H4", count=300)
        df_4h = calculate_indicators(df_4h)

        print("Fetching economic calendar...")
        events = fetch_economic_events()

        print("Generating signal...")
        signal_data = generate_signal(df, df_4h, events)
        print(f"Action: {signal_data['action']} | Bull: {signal_data['bull_score']} Bear: {signal_data['bear_score']} | 4H: {signal_data['h4_conf']}")

        print("Sending email...")
        build_and_send_email(signal_data)

    except Exception as e:
        print(f"Error: {e}")


# ─── SCHEDULER ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GBP/USD Signal System starting...")
    print("Scheduled daily at 5:00 PM EST (New York close)")

    run_signal_job()

    schedule.every().day.at("22:00").do(run_signal_job)  # 22:00 UTC = 17:00 EST

    while True:
        schedule.run_pending()
        time.sleep(60)
