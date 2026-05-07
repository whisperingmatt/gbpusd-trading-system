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
    """
    Uses 200 candles to determine macro trend.
    Measures the range over 200 candles — if price is oscillating
    in a wide band without a clear directional slope, it is SIDEWAYS.
    BULLISH requires price above 200 EMA, 200 EMA rising over both
    20 and 50 candles, and price in upper half of 200-candle range.
    BEARISH is the mirror image.
    """
    price        = df.iloc[-1]["close"]
    ema200_now   = df["ema200"].iloc[-1]
    ema200_20ago = df["ema200"].iloc[-20]
    ema200_50ago = df["ema200"].iloc[-50]

    slope_20 = (ema200_now - ema200_20ago) / ema200_20ago * 100
    slope_50 = (ema200_now - ema200_50ago) / ema200_50ago * 100

    # 200-candle range to detect sideways macro channel
    high_200 = df["high"].iloc[-200:].max()
    low_200  = df["low"].iloc[-200:].min()
    range_200 = high_200 - low_200
    range_mid = low_200 + range_200 * 0.5
    range_pct = range_200 / ema200_now * 100  # range as % of price

    # If 200-candle range is wide (>8%) and slope is flat, it's sideways
    is_wide_range   = range_pct > 8.0
    is_slope_flat   = abs(slope_50) < 0.3
    in_upper_half   = price > range_mid
    in_lower_half   = price < range_mid

    if is_wide_range and is_slope_flat:
        direction = "SIDEWAYS"
        detail    = f"Range {round(low_200,4)}-{round(high_200,4)} ({round(range_pct,1)}% wide)"
    elif price > ema200_now and slope_20 > 0.05 and slope_50 > 0.1:
        direction = "BULLISH"
        detail    = f"200 EMA rising, price above"
    elif price < ema200_now and slope_20 < -0.05 and slope_50 < -0.1:
        direction = "BEARISH"
        detail    = f"200 EMA falling, price below"
    elif is_wide_range and in_upper_half:
        direction = "SIDEWAYS (upper range)"
        detail    = f"Range {round(low_200,4)}-{round(high_200,4)}, price in upper half"
    elif is_wide_range and in_lower_half:
        direction = "SIDEWAYS (lower range)"
        detail    = f"Range {round(low_200,4)}-{round(high_200,4)}, price in lower half"
    else:
        direction = "SIDEWAYS"
        detail    = f"No clear trend"

    return direction, detail


# ─── FIBONACCI (0.5 and 0.618 only) ─────────────────────────────────────────
def calculate_fibonacci(df, lookback=50):
    recent     = df.iloc[-lookback:]
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    diff       = swing_high - swing_low
    price      = df.iloc[-1]["close"]
    atr        = df.iloc[-1]["atr"]

    fib50  = round(swing_high - 0.500 * diff, 5)
    fib618 = round(swing_high - 0.618 * diff, 5)

    near_fib50  = abs(price - fib50)  <= 0.5 * atr
    near_fib618 = abs(price - fib618) <= 0.5 * atr

    return {
        "fib50":      fib50,
        "fib618":     fib618,
        "near_fib50":  near_fib50,
        "near_fib618": near_fib618,
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
def build_scorecard(df, lt_trend, fib, fvg):
    """
    Each indicator is scored for bullish/bearish alignment.
    Points are weighted by reliability.

    Max bullish score: 10
    Max bearish score: 10

    CONFIRMED signal:  >= 8 points
    WATCH (setup forming): 5-7 points
    NO TRADE: < 5 points
    """
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    price    = latest["close"]
    ema50    = latest["ema50"]
    ema200   = latest["ema200"]
    rsi      = latest["rsi"]
    atr      = latest["atr"]

    prev_ema50  = previous["ema50"]
    prev_ema200 = previous["ema200"]
    prev_rsi    = previous["rsi"]
    prev_atr    = previous["atr"]

    # Direction arrows
    ema50_dir  = "▲" if ema50  > prev_ema50  else "▼"
    ema200_dir = "▲" if ema200 > prev_ema200 else "▼"
    rsi_dir    = "▲" if rsi    > prev_rsi    else "▼"
    atr_dir    = "▲" if atr    > prev_atr    else "▼"  # expanding = more volatility

    items = []
    bull_score = 0
    bear_score = 0

    # 1. Long-term trend (3 pts)
    if "BULLISH" in lt_trend:
        items.append(("Long-Term Trend", "BULLISH", "✅", 3, 0))
        bull_score += 3
    elif "BEARISH" in lt_trend:
        items.append(("Long-Term Trend", "BEARISH", "❌", 0, 3))
        bear_score += 3
    else:
        items.append(("Long-Term Trend", "SIDEWAYS", "⚠️ ", 0, 0))

    # 2. Price vs 50 EMA (2 pts)
    if price > ema50:
        items.append((f"Price vs 50 EMA {ema50_dir}", f"{price} > {round(ema50,5)}", "✅", 2, 0))
        bull_score += 2
    else:
        items.append((f"Price vs 50 EMA {ema50_dir}", f"{price} < {round(ema50,5)}", "❌", 0, 2))
        bear_score += 2

    # 3. 50 EMA vs 200 EMA (2 pts)
    if ema50 > ema200:
        items.append((f"50 EMA vs 200 EMA {ema200_dir}", f"{round(ema50,5)} > {round(ema200,5)}", "✅", 2, 0))
        bull_score += 2
    else:
        items.append((f"50 EMA vs 200 EMA {ema200_dir}", f"{round(ema50,5)} < {round(ema200,5)}", "❌", 0, 2))
        bear_score += 2

    # 4. RSI (1 pt) — bullish if 40-70, bearish if 30-60, neutral at extremes
    rsi_state = f"{round(rsi,1)} {rsi_dir}"
    if 40 <= rsi <= 65:
        items.append((f"RSI (14) {rsi_dir}", rsi_state, "✅", 1, 0))
        bull_score += 1
    elif 35 <= rsi < 40:
        items.append((f"RSI (14) {rsi_dir}", rsi_state + " (approaching oversold)", "⚠️ ", 0, 0))
    elif rsi > 70:
        items.append((f"RSI (14) {rsi_dir}", rsi_state + " (overbought — caution)", "⚠️ ", 0, 0))
    elif rsi < 30:
        items.append((f"RSI (14) {rsi_dir}", rsi_state + " (oversold)", "✅", 1, 0))
        bull_score += 1
    else:
        items.append((f"RSI (14) {rsi_dir}", rsi_state, "⚠️ ", 0, 1))
        bear_score += 1

    # 5. Fib 0.618 confluence (2 pts)
    if fib["near_fib618"]:
        items.append(("Fib 0.618 Confluence", f"Price near {fib['fib618']}", "✅", 2, 2))
        bull_score += 2
        bear_score += 2
    else:
        items.append(("Fib 0.618", f"Level at {fib['fib618']}", "—", 0, 0))

    # 6. Fib 0.5 confluence (1 pt)
    if fib["near_fib50"]:
        items.append(("Fib 0.500 Confluence", f"Price near {fib['fib50']}", "✅", 1, 1))
        bull_score += 1
        bear_score += 1
    else:
        items.append(("Fib 0.500", f"Level at {fib['fib50']}", "—", 0, 0))

    # 7. FVG active (2 pts)
    if fvg["bullish_active"]:
        g = fvg["bullish_fvg"]
        items.append(("Bullish FVG", f"ACTIVE {g['gap_low']}-{g['gap_high']}", "✅", 2, 0))
        bull_score += 2
    elif fvg["bearish_active"]:
        g = fvg["bearish_fvg"]
        items.append(("Bearish FVG", f"ACTIVE {g['gap_low']}-{g['gap_high']}", "❌", 0, 2))
        bear_score += 2
    elif fvg["bullish_fvg"]:
        g = fvg["bullish_fvg"]
        items.append(("Bullish FVG", f"Unfilled {g['gap_low']}-{g['gap_high']} ({g['date']})", "—", 0, 0))
    elif fvg["bearish_fvg"]:
        g = fvg["bearish_fvg"]
        items.append(("Bearish FVG", f"Unfilled {g['gap_low']}-{g['gap_high']} ({g['date']})", "—", 0, 0))
    else:
        items.append(("FVG", "None detected", "—", 0, 0))

    # ATR info (informational only — not scored)
    atr_state = "Expanding (volatility rising)" if atr > prev_atr else "Contracting (volatility falling)"
    items.append((f"ATR (14) {atr_dir}", f"{round(atr,5)} — {atr_state}", "ℹ️ ", 0, 0))

    return items, bull_score, bear_score


# ─── SIGNAL ENGINE ───────────────────────────────────────────────────────────
def generate_signal(df, events):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]

    price  = latest["close"]
    ema50  = latest["ema50"]
    ema200 = latest["ema200"]
    rsi    = latest["rsi"]
    atr    = latest["atr"]

    lt_trend, lt_detail = classify_long_term_trend(df)
    fib                 = calculate_fibonacci(df)
    fvg                 = detect_fvg(df)
    scorecard, bull_score, bear_score = build_scorecard(df, lt_trend, fib, fvg)

    news_today = len(events) > 0

    # Determine action status
    if news_today:
        if bull_score >= 8:
            action = "STAND DOWN — HIGH IMPACT NEWS TODAY"
            action_detail = f"Setup is CONFIRMED BUY ({bull_score}/10) but do NOT trade into news. Wait for tomorrow's signal."
        elif bear_score >= 8:
            action = "STAND DOWN — HIGH IMPACT NEWS TODAY"
            action_detail = f"Setup is CONFIRMED SELL ({bear_score}/10) but do NOT trade into news. Wait for tomorrow's signal."
        else:
            action = "STAND DOWN — HIGH IMPACT NEWS TODAY"
            action_detail = "No confirmed setup and news risk is high. Sit this one out."
    elif bull_score >= 8:
        action = "CONFIRMED BUY"
        action_detail = f"Score {bull_score}/10 — All key conditions aligned. Execute at 50 EMA entry."
    elif bear_score >= 8:
        action = "CONFIRMED SELL"
        action_detail = f"Score {bear_score}/10 — All key conditions aligned. Execute at 50 EMA entry."
    elif bull_score >= 5:
        action = "WATCH — BUY SETUP FORMING"
        action_detail = f"Score {bull_score}/10 — Setup incomplete. Monitor for confluence to improve."
    elif bear_score >= 5:
        action = "WATCH — SELL SETUP FORMING"
        action_detail = f"Score {bear_score}/10 — Setup incomplete. Monitor for confluence to improve."
    else:
        action = "NO TRADE"
        action_detail = f"Bull: {bull_score}/10  Bear: {bear_score}/10 — No edge. Stay out."

    return {
        "action":       action,
        "action_detail": action_detail,
        "bull_score":   bull_score,
        "bear_score":   bear_score,
        "price":        round(price, 5),
        "ema50":        round(ema50, 5),
        "ema200":       round(ema200, 5),
        "rsi":          round(rsi, 2),
        "atr":          round(atr, 5),
        "fib50":        fib["fib50"],
        "fib618":       fib["fib618"],
        "stop_buy":     round(price - 1.5 * atr, 5),
        "target_buy":   round(price + 3.0 * atr, 5),
        "stop_sell":    round(price + 1.5 * atr, 5),
        "target_sell":  round(price - 3.0 * atr, 5),
        "lt_trend":     lt_trend,
        "lt_detail":    lt_detail,
        "scorecard":    scorecard,
        "news_warning": news_today,
        "events":       [e.get("event", "Unknown") for e in events],
    }


# ─── EMAIL ───────────────────────────────────────────────────────────────────
def build_and_send_email(data):
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz).strftime("%A %d %B %Y - %I:%M %p EST")
    ny_date = datetime.now(ny_tz).strftime("%d %b %Y")

    # Action header emoji
    if "CONFIRMED BUY" in data["action"]:
        action_emoji = "🟢"
    elif "CONFIRMED SELL" in data["action"]:
        action_emoji = "🔴"
    elif "WATCH" in data["action"]:
        action_emoji = "🟡"
    elif "STAND DOWN" in data["action"]:
        action_emoji = "⛔"
    else:
        action_emoji = "⚪"

    lt_icon = {"BULLISH": "📈", "BEARISH": "📉"}.get(
        data["lt_trend"].split()[0], "➡️"
    )

    # Scorecard rows
    sc_lines = ""
    for name, value, icon, bp, sp in data["scorecard"]:
        sc_lines += f"  {icon}  {name:<30} {value}\n"

    # News block
    news_block = ""
    if data["news_warning"]:
        event_lines = "\n".join(f"       {e}" for e in data["events"])
        news_block  = f"""
------------------------------------------------------------
⛔ HIGH IMPACT NEWS TODAY
------------------------------------------------------------
{event_lines}
"""

    body = f"""GBP/USD DAILY SIGNAL REPORT
{ny_time}
============================================================

{action_emoji} {data['action']}
   {data['action_detail']}

   Bull Score : {data['bull_score']}/10
   Bear Score : {data['bear_score']}/10

------------------------------------------------------------
PRICE SNAPSHOT
------------------------------------------------------------
  Current Price : {data['price']}
  50 EMA        : {data['ema50']}
  200 EMA       : {data['ema200']}
  Fib 0.500     : {data['fib50']}
  Fib 0.618     : {data['fib618']}
  RSI (14)      : {data['rsi']}
  ATR (14)      : {data['atr']}

  Long-Term     : {lt_icon} {data['lt_trend']}
                  {data['lt_detail']}

------------------------------------------------------------
RISK LEVELS  (only act on CONFIRMED signals)
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
SCORECARD
------------------------------------------------------------
{sc_lines}
{news_block}
------------------------------------------------------------
REMINDERS
------------------------------------------------------------
  Only trade CONFIRMED signals (8+/10)
  Max 1-2% risk per trade
  Confirm entry on TradingView before executing
  No trade on high impact news days

------------------------------------------------------------
GBP/USD Trading System v1.2"""

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
