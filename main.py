import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import schedule
import pytz

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OANDA_API_KEY    = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV        = os.getenv("OANDA_ENV", "practice")
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY")
EMAIL_SENDER     = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD   = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT  = os.getenv("EMAIL_RECIPIENT")

OANDA_BASE = (
    "https://api-fxpractice.oanda.com/v3"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com/v3"
)

INSTRUMENT = "GBP_USD"

# ─── OANDA: FETCH DAILY CANDLES ───────────────────────────────────────────────
def fetch_candles(count=250):
    url = f"{OANDA_BASE}/instruments/{INSTRUMENT}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params  = {"granularity": "D", "count": count, "price": "M"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    candles = response.json()["candles"]

    rows = []
    for c in candles:
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

# ─── TECHNICAL INDICATORS ─────────────────────────────────────────────────────
def calculate_indicators(df):
    # EMAs
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (14)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()

    return df

# ─── ECONOMIC CALENDAR ────────────────────────────────────────────────────────
def fetch_economic_events():
    today     = datetime.utcnow().strftime("%Y-%m-%d")
    tomorrow  = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/calendar/economic"
        f"?from={today}&to={tomorrow}&token={FINNHUB_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json().get("economicCalendar", [])
        high_impact = [
            e for e in events
            if e.get("impact") == "high"
            and e.get("country") in ("US", "GB")
        ]
        return high_impact
    except Exception as e:
        print(f"Calendar fetch error: {e}")
        return []

# ─── SIGNAL ENGINE ────────────────────────────────────────────────────────────
def generate_signal(df, events):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]

    price   = latest["close"]
    ema50   = latest["ema50"]
    ema200  = latest["ema200"]
    rsi     = latest["rsi"]
    atr     = latest["atr"]

    # News warning
    news_warning = len(events) > 0
    event_names  = [e.get("event", "Unknown event") for e in events]

    # Trend structure
    bullish_trend = price > ema50 > ema200
    bearish_trend = price < ema50 < ema200

    # RSI conditions
    rsi_neutral     = 40 <= rsi <= 60
    rsi_overbought  = rsi > 70
    rsi_oversold    = rsi < 30

    # Pullback to EMA50 (entry trigger)
    near_ema50 = abs(price - ema50) / atr < 0.5

    # Determine signal
    if bullish_trend and (rsi_neutral or rsi_oversold) and not rsi_overbought:
        if near_ema50 or previous["close"] < previous["ema50"]:
            signal = "BUY"
        else:
            signal = "BUY BIAS — Wait for pullback to 50 EMA"
    elif bearish_trend and (rsi_neutral or rsi_overbought) and not rsi_oversold:
        if near_ema50 or previous["close"] > previous["ema50"]:
            signal = "SELL"
        else:
            signal = "SELL BIAS — Wait for rally to 50 EMA"
    else:
        signal = "NO SIGNAL — Mixed conditions"

    # Risk levels
    stop_buy    = round(price - (1.5 * atr), 5)
    target_buy  = round(price + (3.0 * atr), 5)
    stop_sell   = round(price + (1.5 * atr), 5)
    target_sell = round(price - (3.0 * atr), 5)

    return {
        "signal":       signal,
        "price":        round(price, 5),
        "ema50":        round(ema50, 5),
        "ema200":       round(ema200, 5),
        "rsi":          round(rsi, 2),
        "atr":          round(atr, 5),
        "stop_buy":     stop_buy,
        "target_buy":   target_buy,
        "stop_sell":    stop_sell,
        "target_sell":  target_sell,
        "news_warning": news_warning,
        "events":       event_names,
    }

# ─── EMAIL BUILDER ────────────────────────────────────────────────────────────
def build_email(data):
    ny_time = datetime.now(pytz.timezone("America/New_York")).strftime("%A %d %B %Y – %I:%M %p EST")

    signal_emoji = {
        "BUY":   "🟢",
        "SELL":  "🔴",
    }
    emoji = next((v for k, v in signal_emoji.items() if k in data["signal"]), "🟡")

    news_block = ""
    if data["news_warning"]:
        events_list = "\n".join(f"  ⚠️  {e}" for e in data["events"])
        news_block = f"""
────────────────────────────
HIGH IMPACT NEWS – CAUTION
────────────────────────────
{events_list}

⚠️  Consider waiting for post-news price to settle before entering.
"""

    body = f"""
GBP/USD DAILY SIGNAL REPORT
{ny_time}
════════════════════════════

SIGNAL:  {emoji} {data['signal']}

────────────────────────────
PRICE SNAPSHOT
────────────────────────────
Current Price : {data['price']}
50 EMA        : {data['ema50']}
200 EMA       : {data['ema200']}
RSI (14)      : {data['rsi']}
ATR (14)      : {data['atr']}

────────────────────────────
RISK LEVELS
────────────────────────────
IF BUYING:
  Entry  : Market / limit at 50 EMA
  Stop   : {data['stop_buy']}
  Target : {data['target_buy']}
  R:R    : 1:2

IF SELLING:
  Entry  : Market / limit at 50 EMA
  Stop   : {data['stop_sell']}
  Target : {data['target_sell']}
  R:R    : 1:2
{news_block}
────────────────────────────
REMINDERS
────────────────────────────
✅  Max 1-2% risk per trade
✅  Confirm entry on TradingView before executing
✅  Check Forex Factory for any missed events
✅  No trade without confirmed signal

────────────────────────────
GBP/USD Trading System v1.0
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"GBP/USD Signal: {emoji} {data['signal']} | {datetime.now(pytz.timezone('America/New_York')).strftime('%d %b %Y')}"
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT
    msg.attach(MIMEText(body, "plain"))
    return msg

# ─── EMAIL SENDER ─────────────────────────────────────────────────────────────
def send_email(msg):
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {os.getenv('SENDGRID_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "personalizations": [{"to": [{"email": EMAIL_RECIPIENT}]}],
        "from": {"email": EMAIL_SENDER},
        "subject": msg["Subject"],
        "content": [{"type": "text/plain", "value": msg.get_payload()}]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print(f"✅ Email sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── MAIN JOB ─────────────────────────────────────────────────────────────────
def run_signal_job():
    print(f"\n{'='*40}")
    print(f"Running signal job at {datetime.utcnow()} UTC")
    print(f"{'='*40}")

    try:
        print("Fetching candles from OANDA...")
        df = fetch_candles()

        print("Calculating indicators...")
        df = calculate_indicators(df)

        print("Fetching economic calendar...")
        events = fetch_economic_events()

        print("Generating signal...")
        signal_data = generate_signal(df, events)
        print(f"Signal: {signal_data['signal']}")

        print("Sending email...")
        msg = build_email(signal_data)
        send_email(msg)

    except Exception as e:
        print(f"❌ Error: {e}")

# ─── SCHEDULER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GBP/USD Signal System starting...")
    print("Scheduled daily at 5:00 PM EST (New York close)")

    # Run immediately on startup so you can test it
    run_signal_job()

    # Then schedule daily at 17:00 EST
    schedule.every().day.at("22:00").do(run_signal_job)  # 22:00 UTC = 17:00 EST

    while True:
        schedule.run_pending()
        time.sleep(60)
