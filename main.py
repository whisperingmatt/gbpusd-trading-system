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


# ─── OANDA: FETCH DAILY CANDLES ──────────────────────────────────────────────
def fetch_candles(count=250):
    url     = f"{OANDA_BASE}/instruments/{INSTRUMENT}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params  = {"granularity": "D", "count": count, "price": "M"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    rows = []
    for c in response.json()["candles"]:
        if c["complete"]:
            rows.append({
                "time":   c["time"][:10],
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


# ─── TECHNICAL INDICATORS ────────────────────────────────────────────────────
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

    is_wide_range = range_pct > 8.0
    is_slope_flat = abs(slope_50) < 0.3
    in_upper_half = price > range_mid

    if is_wide_range and is_slope_flat:
        position = "upper half" if in_upper_half else "lower half"
        return "SIDEWAYS", f"Channel {round(low_200,4)}-{round(high_200,4)} | In {position}"
    elif price > ema200_now and slope_20 > 0.05 and slope_50 > 0.1:
        return "BULLISH", f"200 EMA rising | Price above"
    elif price < ema200_now and slope_20 < -0.05 and slope_50 < -0.1:
        return "BEARISH", f"200 EMA falling | Price below"
    elif is_wide_range and in_upper_half:
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

        if c3["low"] > c1["high"]:
            if price > c1["high"]:
                bullish_fvgs.append({
                    "date":     str(recent.iloc[i + 1]["time"])[:10],
                    "gap_high": round(float(c3["low"]), 5),
                    "gap_low":  round(float(c1["high"]), 5),
                })

        if c3["high"] < c1["low"]:
            if price < c1["low"]:
                bearish_fvgs.append({
                    "date":     str(recent.iloc[i + 1]["time"])[:10],
                    "gap_high": round(float(c1["low"]), 5),
                    "gap_low":  round(float(c3["high"]), 5),
                })

    nearest_bullish = bullish_fvgs[-1] if bullish_fvgs else None
    nearest_bearish = bearish_fvgs[-1] if bearish_fvgs else None

    bullish_active = (
        nearest_bullish is not None and
        nearest_bullish["gap_low"] - 0.5 * atr <= price <= nearest_bullish["gap_high"] + 0.5 * atr
    )
    bearish_active = (
        nearest_bearish is not None and
        nearest_bearish["gap_low"] - 0.5 * atr <= price <= nearest_bearish["gap_high"] + 0.5 * atr
    )

    return {
        "bullish_fvg":    nearest_bullish,
        "bearish_fvg":    nearest_bearish,
        "bullish_active": bullish_active,
        "bearish_active": bearish_active,
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
def build_scorecard(df, lt_trend, st_trend, fib, fvg):
    """
    Scoring weights:
      Long-term trend  : 3 pts   (SIDEWAYS = 0)
      Short-term trend : 2 pts   (NEUTRAL  = 0)
      Price vs 50 EMA  : 2 pts
      50 EMA vs 200 EMA: 2 pts
      FVG active       : 2 pts
      Fib 0.618        : 2 pts   (confluence bonus)
      Fib 0.500        : 1 pt    (confluence bonus)
      RSI              : 1 pt
      ATR              : info only

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

    prev_close = previous["close"]
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

    # Entry zone: price within 0.75x ATR of 50 EMA
    ema50_zone_low  = round(ema50 - 0.75 * atr, 5)
    ema50_zone_high = round(ema50 + 0.75 * atr, 5)

    rows       = []
    bull_score = 0
    bear_score = 0

    # ── TREND SECTION ──────────────────────────────────────────────────────

    # 1. Long-term trend (3 pts)
    lt_icons = {"BULLISH": "📈", "BEARISH": "📉", "SIDEWAYS": "➡️"}
    lt_icon  = lt_icons.get(lt_trend, "➡️")
    if lt_trend == "BULLISH":
        rows.append(("✅", f"Long-Term {ema200_dir}", f"{lt_icon} BULLISH", "Ideal: price > 200 EMA"))
        bull_score += 3
    elif lt_trend == "BEARISH":
        rows.append(("❌", f"Long-Term {ema200_dir}", f"{lt_icon} BEARISH", "Ideal: price < 200 EMA"))
        bear_score += 3
    else:
        rows.append(("⚠️ ", f"Long-Term {ema200_dir}", f"{lt_icon} SIDEWAYS", "No trend — range rules apply"))

    # 2. Short-term trend (2 pts)
    st_icons = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}
    st_icon  = st_icons.get(st_trend, "➡️")
    if st_trend == "BULLISH":
        rows.append(("✅", f"Short-Term {ema20_dir}", f"{st_icon} BULLISH", f"20 EMA: {round(ema20,5)}  Ideal: price above & rising"))
        bull_score += 2
    elif st_trend == "BEARISH":
        rows.append(("❌", f"Short-Term {ema20_dir}", f"{st_icon} BEARISH", f"20 EMA: {round(ema20,5)}  Ideal: price below & falling"))
        bear_score += 2
    else:
        rows.append(("⚠️ ", f"Short-Term {ema20_dir}", f"{st_icon} NEUTRAL", f"20 EMA: {round(ema20,5)}  No clear direction"))

    # ── PRICE SECTION ──────────────────────────────────────────────────────

    # 3. Current price (info — not scored)
    price_change = round(price - prev_close, 5)
    change_str   = f"+{price_change}" if price_change > 0 else str(price_change)
    rows.append(("──", "PRICE SEPARATOR", "", ""))
    rows.append(("  ", f"Current Price {price_dir}", str(round(price, 5)), f"{change_str} from yesterday"))

    # 4. 50 EMA — entry level (2 pts)
    if price > ema50:
        rows.append(("✅", f"50 EMA {ema50_dir}", str(round(ema50, 5)), f"Entry zone {ema50_zone_low}-{ema50_zone_high}"))
        bull_score += 2
    else:
        rows.append(("❌", f"50 EMA {ema50_dir}", str(round(ema50, 5)), f"Entry zone {ema50_zone_low}-{ema50_zone_high}"))
        bear_score += 2

    # 5. 200 EMA — trend filter (2 pts)
    if ema50 > ema200:
        rows.append(("✅", f"200 EMA {ema200_dir}", str(round(ema200, 5)), "Bull: 50 EMA above | Bear: 50 EMA below"))
        bull_score += 2
    else:
        rows.append(("❌", f"200 EMA {ema200_dir}", str(round(ema200, 5)), "Bull: 50 EMA above | Bear: 50 EMA below"))
        bear_score += 2

    # 6. RSI (1 pt)
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

    # 7. FVG (2 pts)
    rows.append(("──", "FVG SEPARATOR", "", ""))
    if fvg["bullish_active"]:
        g = fvg["bullish_fvg"]
        rows.append(("✅", "Fair Value Gap", f"BULLISH ACTIVE", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
        bull_score += 2
    elif fvg["bearish_active"]:
        g = fvg["bearish_fvg"]
        rows.append(("❌", "Fair Value Gap", f"BEARISH ACTIVE", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
        bear_score += 2
    elif fvg["bullish_fvg"]:
        g = fvg["bullish_fvg"]
        rows.append(("—", "Fair Value Gap", "Bullish unfilled", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
    elif fvg["bearish_fvg"]:
        g = fvg["bearish_fvg"]
        rows.append(("—", "Fair Value Gap", "Bearish unfilled", f"Zone {g['gap_low']}-{g['gap_high']} ({g['date']})"))
    else:
        rows.append(("—", "Fair Value Gap", "None detected", ""))

    # 8. ATR (info only)
    atr_note = "Expanding — wider stops needed" if atr > prev_atr else "Contracting — tighter stops possible"
    rows.append(("ℹ️ ", f"ATR (14) {atr_dir}", str(round(atr, 5)), atr_note))

    # ── FIB SECTION ────────────────────────────────────────────────────────

    rows.append(("──", "FIB SEPARATOR", "", ""))

    # 9. Fib 0.618 (2 pts — confluence bonus)
    if fib["near_fib618"]:
        rows.append(("✅", "Fib 0.618", str(fib["fib618"]), "CONFLUENCE with entry — high conviction"))
        bull_score += 2
        bear_score += 2
    else:
        rows.append(("—", "Fib 0.618", str(fib["fib618"]), f"Support/resistance level"))

    # 10. Fib 0.500 (1 pt — confluence bonus)
    if fib["near_fib50"]:
        rows.append(("✅", "Fib 0.500", str(fib["fib50"]), "CONFLUENCE with entry — added weight"))
        bull_score += 1
        bear_score += 1
    else:
        rows.append(("—", "Fib 0.500", str(fib["fib50"]), f"Support/resistance level"))

    rows.append(("  ", "Swing High", str(fib["swing_high"]), "50-candle reference"))
    rows.append(("  ", "Swing Low",  str(fib["swing_low"]),  "50-candle reference"))

    return rows, bull_score, bear_score


# ─── SIGNAL ENGINE ───────────────────────────────────────────────────────────
def generate_signal(df, events):
    latest = df.iloc[-1]
    price  = latest["close"]
    ema50  = latest["ema50"]
    atr    = latest["atr"]

    lt_trend, lt_detail = classify_long_term_trend(df)
    st_trend, st_detail = classify_short_term_trend(df)
    fib                 = calculate_fibonacci(df)
    fvg                 = detect_fvg(df)
    scorecard, bull_score, bear_score = build_scorecard(df, lt_trend, st_trend, fib, fvg)

    news_today = len(events) > 0

    if bull_score >= 8:
        raw        = "CONFIRMED BUY"
        raw_detail = f"Score {bull_score}/10 — Strong confluence. Limit order at 50 EMA ({round(ema50,5)})."
    elif bear_score >= 8:
        raw        = "CONFIRMED SELL"
        raw_detail = f"Score {bear_score}/10 — Strong confluence. Limit order at 50 EMA ({round(ema50,5)})."
    elif bull_score >= 5:
        raw        = "WATCH — BUY SETUP FORMING"
        raw_detail = f"Score {bull_score}/10 — Not enough confluence yet. Wait for score to reach 8."
    elif bear_score >= 5:
        raw        = "WATCH — SELL SETUP FORMING"
        raw_detail = f"Score {bear_score}/10 — Not enough confluence yet. Wait for score to reach 8."
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
        "scorecard":     scorecard,
        "news_warning":  news_today,
        "events":        [e.get("event", "Unknown") for e in events],
    }


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def build_and_send_email(data):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    if "CONFIRMED BUY"  in data["action"]: action_emoji = "🟢"
    elif "CONFIRMED SELL" in data["action"]: action_emoji = "🔴"
    elif "WATCH"          in data["action"]: action_emoji = "🟡"
    elif "STAND DOWN"     in data["action"]: action_emoji = "⛔"
    else:                                    action_emoji = "⚪"

    # Build table — skip separator rows
    col1 = 5   # status
    col2 = 22  # indicator
    col3 = 14  # current value
    col4 = 35  # ideal range / info

    divider   = "  " + "-" * (col1 + col2 + col3 + col4) + "\n"
    table     = divider
    table    += f"  {'':>{col1}}  {'INDICATOR':<{col2}}  {'CURRENT':<{col3}}  {'IDEAL RANGE / INFO':<{col4}}\n"
    table    += divider

    for status, indicator, current, info in data["scorecard"]:
        if indicator in ("PRICE SEPARATOR", "FVG SEPARATOR", "FIB SEPARATOR"):
            table += divider
            continue
        table += f"  {status:<{col1}}  {indicator:<{col2}}  {current:<{col3}}  {info:<{col4}}\n"

    table += divider
    table += f"  {'':>{col1}}  {'BULL SCORE':<{col2}}  {str(data['bull_score']) + '/10':<{col3}}\n"
    table += f"  {'':>{col1}}  {'BEAR SCORE':<{col2}}  {str(data['bear_score']) + '/10':<{col3}}\n"
    table += divider

    # News block
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
  🟢 CONFIRMED  8+/10  Execute — strong confluence
  🟡 WATCH      5-7    Setup forming — do not enter yet
  ⚪ NO TRADE   0-4    No edge — stay out
  ⛔ STAND DOWN        News day — wait regardless of score

------------------------------------------------------------
GBP/USD Trading System v1.4"""

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
        print("Fetching candles...")
        df = fetch_candles()
        print("Calculating indicators...")
        df = calculate_indicators(df)
        print("Fetching economic calendar...")
        events = fetch_economic_events()
        print("Generating signal...")
        signal_data = generate_signal(df, events)
        print(f"Action: {signal_data['action']} | Bull: {signal_data['bull_score']} Bear: {signal_data['bear_score']}")
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
