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


# ─── LONG-TERM TREND ─────────────────────────────────────────────────────────
def classify_long_term_trend(df):
    price        = df.iloc[-1]["close"]
    ema200_now   = df["ema200"].iloc[-1]
    ema200_20ago = df["ema200"].iloc[-20]
    ema200_slope = (ema200_now - ema200_20ago) / ema200_20ago * 100

    if price > ema200_now and ema200_slope > 0.05:
        return "BULLISH", ema200_slope
    elif price < ema200_now and ema200_slope < -0.05:
        return "BEARISH", ema200_slope
    else:
        return "SIDEWAYS", ema200_slope


# ─── FIBONACCI LEVELS ────────────────────────────────────────────────────────
def calculate_fibonacci(df, lookback=50):
    recent     = df.iloc[-lookback:]
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    price      = df.iloc[-1]["close"]
    atr        = df.iloc[-1]["atr"]
    diff       = swing_high - swing_low

    levels = {
        "0.236": round(swing_high - 0.236 * diff, 5),
        "0.382": round(swing_high - 0.382 * diff, 5),
        "0.500": round(swing_high - 0.500 * diff, 5),
        "0.618": round(swing_high - 0.618 * diff, 5),
        "0.786": round(swing_high - 0.786 * diff, 5),
    }

    nearby_fib = None
    for label, level in levels.items():
        if abs(price - level) <= 0.5 * atr:
            nearby_fib = (label, level)
            break

    return {
        "swing_high": round(swing_high, 5),
        "swing_low":  round(swing_low, 5),
        "levels":     levels,
        "nearby_fib": nearby_fib,
    }


# ─── FAIR VALUE GAP (FVG) ────────────────────────────────────────────────────
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
            gap_high = c3["low"]
            gap_low  = c1["high"]
            if price > gap_low:
                bullish_fvgs.append({
                    "date":     str(recent.iloc[i + 1]["time"])[:10],
                    "gap_high": round(gap_high, 5),
                    "gap_low":  round(gap_low, 5),
                })

        if c3["high"] < c1["low"]:
            gap_high = c1["low"]
            gap_low  = c3["high"]
            if price < gap_high:
                bearish_fvgs.append({
                    "date":     str(recent.iloc[i + 1]["time"])[:10],
                    "gap_high": round(gap_high, 5),
                    "gap_low":  round(gap_low, 5),
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


# ─── SIGNAL ENGINE ───────────────────────────────────────────────────────────
def generate_signal(df, events):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]

    price  = latest["close"]
    ema50  = latest["ema50"]
    ema200 = latest["ema200"]
    rsi    = latest["rsi"]
    atr    = latest["atr"]

    bullish_trend  = price > ema50 > ema200
    bearish_trend  = price < ema50 < ema200
    rsi_overbought = rsi > 70
    rsi_oversold   = rsi < 30
    near_ema50     = abs(price - ema50) / atr < 0.5

    lt_trend, lt_slope = classify_long_term_trend(df)
    fib                = calculate_fibonacci(df)
    fvg                = detect_fvg(df)

    if bullish_trend and not rsi_overbought:
        if near_ema50 or previous["close"] < previous["ema50"]:
            signal = "BUY"
        else:
            signal = "BUY BIAS — Wait for pullback to 50 EMA"
    elif bearish_trend and not rsi_oversold:
        if near_ema50 or previous["close"] > previous["ema50"]:
            signal = "SELL"
        else:
            signal = "SELL BIAS — Wait for rally to 50 EMA"
    else:
        signal = "NO SIGNAL — Mixed conditions"

    confirmations = []
    if fib["nearby_fib"]:
        confirmations.append(f"Price near Fib {fib['nearby_fib'][0]} ({fib['nearby_fib'][1]})")
    if fvg["bullish_active"] and "BUY" in signal:
        confirmations.append(f"Inside Bullish FVG ({fvg['bullish_fvg']['gap_low']} - {fvg['bullish_fvg']['gap_high']})")
    if fvg["bearish_active"] and "SELL" in signal:
        confirmations.append(f"Inside Bearish FVG ({fvg['bearish_fvg']['gap_low']} - {fvg['bearish_fvg']['gap_high']})")
    if lt_trend == "BULLISH" and "BUY" in signal:
        confirmations.append("Long-term trend aligned (Bullish)")
    if lt_trend == "BEARISH" and "SELL" in signal:
        confirmations.append("Long-term trend aligned (Bearish)")

    return {
        "signal":        signal,
        "price":         round(price, 5),
        "ema50":         round(ema50, 5),
        "ema200":        round(ema200, 5),
        "rsi":           round(rsi, 2),
        "atr":           round(atr, 5),
        "stop_buy":      round(price - 1.5 * atr, 5),
        "target_buy":    round(price + 3.0 * atr, 5),
        "stop_sell":     round(price + 1.5 * atr, 5),
        "target_sell":   round(price - 3.0 * atr, 5),
        "lt_trend":      lt_trend,
        "fib":           fib,
        "fvg":           fvg,
        "confirmations": confirmations,
        "news_warning":  len(events) > 0,
        "events":        [e.get("event", "Unknown") for e in events],
    }


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def build_and_send_email(data):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    emoji   = "BUY" in data["signal"] and "🟢" or "SELL" in data["signal"] and "🔴" or "🟡"
    lt_icon = {"BULLISH": "📈", "BEARISH": "📉", "SIDEWAYS": "➡️"}.get(data["lt_trend"], "")

    fib    = data["fib"]
    levels = fib["levels"]
    nearby = f"  NEAR: Fib {fib['nearby_fib'][0]} at {fib['nearby_fib'][1]}" if fib["nearby_fib"] else "  No key Fib level nearby"

    fvg    = data["fvg"]
    bfvg   = fvg["bullish_fvg"]
    befvg  = fvg["bearish_fvg"]
    b_line = (f"  Bullish FVG : {bfvg['gap_low']} - {bfvg['gap_high']} ({bfvg['date']})" +
              (" << ACTIVE" if fvg["bullish_active"] else "")) if bfvg else "  Bullish FVG : None"
    s_line = (f"  Bearish FVG : {befvg['gap_low']} - {befvg['gap_high']} ({befvg['date']})" +
              (" << ACTIVE" if fvg["bearish_active"] else "")) if befvg else "  Bearish FVG : None"

    conf_lines = "\n".join(f"  CONFIRMED: {c}" for c in data["confirmations"]) if data["confirmations"] else "  No additional confirmations — trade with caution"

    news_block = ""
    if data["news_warning"]:
        event_lines = "\n".join(f"  WARNING: {e}" for e in data["events"])
        news_block  = f"""
------------------------------------------------------------
HIGH IMPACT NEWS TODAY - USE CAUTION
------------------------------------------------------------
{event_lines}
  Wait for post-news price to settle before entering.
"""

    body = f"""GBP/USD DAILY SIGNAL REPORT
{ny_time}
============================================================

SIGNAL     : {emoji} {data['signal']}
LONG-TERM  : {lt_icon} {data['lt_trend']}

------------------------------------------------------------
PRICE SNAPSHOT
------------------------------------------------------------
  Current Price : {data['price']}
  50 EMA        : {data['ema50']}
  200 EMA       : {data['ema200']}
  RSI (14)      : {data['rsi']}
  ATR (14)      : {data['atr']}

------------------------------------------------------------
RISK LEVELS
------------------------------------------------------------
  IF BUYING:
    Entry  : Limit at 50 EMA ({data['ema50']})
    Stop   : {data['stop_buy']}
    Target : {data['target_buy']}
    R:R    : 1:2

  IF SELLING:
    Entry  : Limit at 50 EMA ({data['ema50']})
    Stop   : {data['stop_sell']}
    Target : {data['target_sell']}
    R:R    : 1:2

------------------------------------------------------------
FIBONACCI LEVELS (last 50 candles)
------------------------------------------------------------
  Swing High : {fib['swing_high']}
  Swing Low  : {fib['swing_low']}
  0.236      : {levels['0.236']}
  0.382      : {levels['0.382']}
  0.500      : {levels['0.500']}
  0.618      : {levels['0.618']}
  0.786      : {levels['0.786']}
{nearby}

------------------------------------------------------------
FAIR VALUE GAPS
------------------------------------------------------------
{b_line}
{s_line}

------------------------------------------------------------
SIGNAL CONFIRMATIONS
------------------------------------------------------------
{conf_lines}
{news_block}
------------------------------------------------------------
REMINDERS
------------------------------------------------------------
  Max 1-2% risk per trade
  Confirm entry on TradingView before executing
  Check Forex Factory for any missed events
  No trade without a confirmed signal

------------------------------------------------------------
GBP/USD Trading System v1.1"""

    subject = f"GBP/USD: {emoji} {data['signal']} | {lt_icon} {data['lt_trend']} | {ny_date}"

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
        print(f"Signal: {signal_data['signal']}")
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
        
