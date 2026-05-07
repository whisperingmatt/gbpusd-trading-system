import os
import requests
import pandas as pd
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV     = os.getenv("OANDA_ENV", "practice")
OANDA_BASE    = (
    "https://api-fxpractice.oanda.com/v3"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com/v3"
)
INSTRUMENT = "GBP_USD"

# ─── ACCOUNT SETTINGS ────────────────────────────────────────────────────────
STARTING_BALANCE = 100_000.00
RISK_PER_TRADE   = 2_000.00   # 2% of starting balance
REWARD_RATIO     = 2.0        # 1:2 risk/reward
WIN_AMOUNT       = RISK_PER_TRADE * REWARD_RATIO   # +$4,000
LOSS_AMOUNT      = RISK_PER_TRADE                  # -$2,000
EXPIRED_AMOUNT   = -400.00                         # small cost for no-hit trades


# ─── FETCH ───────────────────────────────────────────────────────────────────
def fetch_candles(granularity="D", count=800):
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
    df["tr"]  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()
    return df


# ─── 4H LOOKUP ───────────────────────────────────────────────────────────────
def build_4h_lookup(df_4h):
    lookup = {}
    df_4h  = df_4h.copy()
    df_4h["date"] = df_4h.index.date
    for date, group in df_4h.groupby("date"):
        eod  = group[group.index.hour <= 20]
        last = (eod if not eod.empty else group).iloc[-1]
        lookup[str(date)] = {
            "close": last["close"], "ema20": last["ema20"],
            "ema50": last["ema50"], "ema200": last["ema200"], "rsi": last["rsi"],
        }
    return lookup


def get_4h_conf(lookup, date_str):
    d = lookup.get(date_str)
    if not d:
        return "NEUTRAL"
    bull = bear = 0
    if d["close"] > d["ema50"]:   bull += 1
    else:                          bear += 1
    if d["ema50"] > d["ema200"]:  bull += 1
    else:                          bear += 1
    if 40 <= d["rsi"] <= 65:      bull += 1
    elif d["rsi"] < 35:            bull += 1
    elif d["rsi"] > 68:            bear += 1
    if bull >= 2:   return "BULL"
    elif bear >= 2: return "BEAR"
    return "NEUTRAL"


# ─── TREND CLASSIFIERS ───────────────────────────────────────────────────────
def lt_at(df, i):
    if i < 200: return "INSUFFICIENT"
    price = df["close"].iloc[i]
    e200  = df["ema200"].iloc[i]
    s20   = (e200 - df["ema200"].iloc[i-20]) / df["ema200"].iloc[i-20] * 100
    s50   = (e200 - df["ema200"].iloc[i-50]) / df["ema200"].iloc[i-50] * 100
    rng   = df["high"].iloc[i-199:i+1].max() - df["low"].iloc[i-199:i+1].min()
    rpct  = rng / e200 * 100
    if rpct > 8.0 and abs(s50) < 0.3:    return "SIDEWAYS"
    elif price > e200 and s20 > 0.05 and s50 > 0.1:  return "BULLISH"
    elif price < e200 and s20 < -0.05 and s50 < -0.1: return "BEARISH"
    return "SIDEWAYS"

def st_at(df, i):
    if i < 5: return "NEUTRAL"
    price = df["close"].iloc[i]
    e20   = df["ema20"].iloc[i]
    slope = (e20 - df["ema20"].iloc[i-5]) / df["ema20"].iloc[i-5] * 100
    if price > e20 and slope > 0.02:   return "BULLISH"
    elif price < e20 and slope < -0.02: return "BEARISH"
    return "NEUTRAL"


# ─── BNR ─────────────────────────────────────────────────────────────────────
def bnr_at(df, i):
    if i < 110: return None, None
    atr   = df["atr"].iloc[i]
    price = df["close"].iloc[i]
    tol   = 0.75 * atr
    sub   = df.iloc[max(0,i-100):i].reset_index()
    swings = []
    w = 5
    for j in range(w, len(sub)-w):
        if sub.iloc[j]["high"] == sub.iloc[j-w:j+w+1]["high"].max():
            swings.append(round(sub.iloc[j]["high"], 5))
        if sub.iloc[j]["low"] == sub.iloc[j-w:j+w+1]["low"].min():
            swings.append(round(sub.iloc[j]["low"], 5))
    bb = be = None
    last20 = df.iloc[max(0,i-20):i+1]
    for lv in set(swings):
        ab = (last20["close"] > lv).sum()
        bl = (last20["close"] < lv).sum()
        if ab >= 3 and bl >= 2 and price > lv and abs(price-lv) <= tol:
            if bb is None or abs(price-lv) < abs(price-bb): bb = lv
        if bl >= 3 and ab >= 2 and price < lv and abs(price-lv) <= tol:
            if be is None or abs(price-lv) < abs(price-be): be = lv
    return bb, be


# ─── FVG ─────────────────────────────────────────────────────────────────────
def fvg_at(df, i):
    if i < 32: return False, False
    price = df["close"].iloc[i]
    atr   = df["atr"].iloc[i]
    sub   = df.iloc[max(0,i-32):i].reset_index()
    bul = ber = False
    for j in range(len(sub)-2):
        c1, c3 = sub.iloc[j], sub.iloc[j+2]
        if c3["low"] > c1["high"] and price > c1["high"]:
            if c1["high"]-0.5*atr <= price <= c3["low"]+0.5*atr: bul = True
        if c3["high"] < c1["low"] and price < c1["low"]:
            if c3["high"]-0.5*atr <= price <= c1["low"]+0.5*atr: ber = True
    return bul, ber


# ─── FIB ─────────────────────────────────────────────────────────────────────
def fib_at(df, i):
    if i < 50: return False, False
    sub  = df.iloc[i-50:i]
    sh   = sub["high"].max()
    sl   = sub["low"].min()
    diff = sh - sl
    p    = df["close"].iloc[i]
    a    = df["atr"].iloc[i]
    return abs(p-(sh-0.5*diff)) <= 0.5*a, abs(p-(sh-0.618*diff)) <= 0.5*a


# ─── SCORE ───────────────────────────────────────────────────────────────────
def score_at(df, i, h4_lookup):
    if i < 201: return 0, 0, "INSUFFICIENT"
    price = df["close"].iloc[i]
    e50   = df["ema50"].iloc[i]
    e200  = df["ema200"].iloc[i]
    rsi   = df["rsi"].iloc[i]

    lt  = lt_at(df, i)
    st  = st_at(df, i)
    h4  = get_4h_conf(h4_lookup, df.index[i].strftime("%Y-%m-%d"))
    bb, be = bnr_at(df, i)
    fb, fer = fvg_at(df, i)
    n50, n618 = fib_at(df, i)

    bull = bear = 0
    if lt == "BULLISH":  bull += 3
    elif lt == "BEARISH": bear += 3
    if st == "BULLISH":  bull += 2
    elif st == "BEARISH": bear += 2
    if h4 == "BULL":     bull += 2
    elif h4 == "BEAR":   bear += 2
    if price > e50:      bull += 2
    else:                bear += 2
    if e50 > e200:       bull += 2
    else:                bear += 2
    if 40 <= rsi <= 65:  bull += 1
    elif rsi < 30:       bull += 1
    elif rsi > 70:       bear += 1
    if bb:               bull += 3
    elif be:             bear += 3
    if fb:               bull += 2
    elif fer:            bear += 2
    if n618:             bull += 2; bear += 2
    if n50:              bull += 1; bear += 1

    if bull >= 8 and h4 != "BEAR":  sig = "CONFIRMED_BUY"
    elif bear >= 8 and h4 != "BULL": sig = "CONFIRMED_SELL"
    elif bull >= 5: sig = "WATCH_BUY"
    elif bear >= 5: sig = "WATCH_SELL"
    else:           sig = "NO_TRADE"

    return bull, bear, sig


# ─── SIMULATE TRADE ──────────────────────────────────────────────────────────
def simulate_trade(df, i, direction, atr):
    if i+1 >= len(df): return "EXPIRED", 0
    entry = df["open"].iloc[i+1]
    stop   = entry - 1.5*atr if direction=="BUY" else entry + 1.5*atr
    target = entry + 3.0*atr if direction=="BUY" else entry - 3.0*atr
    for j in range(i+1, min(i+21, len(df))):
        hi, lo = df["high"].iloc[j], df["low"].iloc[j]
        if direction == "BUY":
            if lo  <= stop:   return "LOSS", j-i
            if hi  >= target: return "WIN",  j-i
        else:
            if hi  >= stop:   return "LOSS", j-i
            if lo  <= target: return "WIN",  j-i
    return "EXPIRED", 20


# ─── RUN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest(df, h4_lookup):
    results       = []
    last_signal_i = -10
    equity        = STARTING_BALANCE
    peak_equity   = STARTING_BALANCE
    max_dd        = 0.0
    equity_curve  = [{"date": df.index[210].strftime("%Y-%m-%d"), "equity": equity}]

    print(f"Scanning {len(df)-210} candles...\n")

    for i in range(210, len(df)-1):
        bull, bear, sig = score_at(df, i, h4_lookup)
        if sig not in ("CONFIRMED_BUY", "CONFIRMED_SELL"): continue
        if i - last_signal_i < 5: continue

        direction     = "BUY" if sig == "CONFIRMED_BUY" else "SELL"
        atr           = df["atr"].iloc[i]
        outcome, hold = simulate_trade(df, i, direction, atr)
        last_signal_i = i

        # P&L
        if outcome == "WIN":      pnl = WIN_AMOUNT
        elif outcome == "LOSS":   pnl = -LOSS_AMOUNT
        else:                     pnl = EXPIRED_AMOUNT

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        if drawdown > max_dd:
            max_dd = drawdown

        entry = df["open"].iloc[i+1] if i+1 < len(df) else df["close"].iloc[i]
        lt    = lt_at(df, i)
        h4    = get_4h_conf(h4_lookup, df.index[i].strftime("%Y-%m-%d"))
        bb, be = bnr_at(df, i)
        fb, fe = fvg_at(df, i)
        date_str = df.index[i].strftime("%Y-%m-%d")

        results.append({
            "date":         date_str,
            "direction":    direction,
            "bull_score":   bull,
            "bear_score":   bear,
            "lt_trend":     lt,
            "h4_aligned":   h4 == ("BULL" if direction=="BUY" else "BEAR"),
            "bnr_active":   bb is not None or be is not None,
            "fvg_active":   fb or fe,
            "entry":        round(entry, 5),
            "atr":          round(atr, 5),
            "outcome":      outcome,
            "pnl":          pnl,
            "equity":       round(equity, 2),
            "drawdown_pct": round(drawdown, 2),
            "hold_candles": hold,
        })

        equity_curve.append({"date": date_str, "equity": round(equity, 2)})

    return pd.DataFrame(results), equity_curve, max_dd, peak_equity


# ─── ANALYSE ─────────────────────────────────────────────────────────────────
def analyse(results, equity_curve, max_dd, peak_equity):
    if results.empty:
        print("No signals fired.")
        return

    total    = len(results)
    wins     = (results["outcome"] == "WIN").sum()
    losses   = (results["outcome"] == "LOSS").sum()
    expired  = (results["outcome"] == "EXPIRED").sum()
    win_rate = wins / total * 100
    final_eq = results["equity"].iloc[-1]
    total_return = (final_eq - STARTING_BALANCE) / STARTING_BALANCE * 100
    r_total  = (wins * 2) - losses
    avg_r    = r_total / total
    years    = 3

    sep = "=" * 58
    print(sep)
    print("  GBP/USD BACKTEST — EQUITY SIMULATION")
    print(f"  Period  : {results['date'].iloc[0]}  to  {results['date'].iloc[-1]}")
    print(f"  Account : ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade (2%)")
    print(sep)
    print(f"\n  Total Signals    : {total}  (~{round(total/years,0):.0f}/year)")
    print(f"  Win Rate         : {round(win_rate,1)}%  ({wins}W / {losses}L / {expired}E)")
    print(f"  Total R          : {r_total:+.0f}R  |  Avg: {avg_r:+.2f}R/trade")
    print(f"\n  Starting Balance : ${STARTING_BALANCE:>12,.2f}")
    print(f"  Final Balance    : ${final_eq:>12,.2f}")
    print(f"  Total Return     : {total_return:>+.1f}%")
    print(f"  Peak Balance     : ${peak_equity:>12,.2f}")
    print(f"  Max Drawdown     : {max_dd:.1f}%  (${peak_equity - results['equity'].min():,.2f})")
    print(f"\n  WIN  : +${WIN_AMOUNT:,.0f} per trade")
    print(f"  LOSS : -${LOSS_AMOUNT:,.0f} per trade")
    print(f"  EXP  : -${abs(EXPIRED_AMOUNT):,.0f} per trade (no hit — time cost)")

    print(f"\n  --- By Market Condition ---")
    for lt in ["BULLISH","BEARISH","SIDEWAYS"]:
        sub = results[results["lt_trend"] == lt]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {lt:<10}: {len(sub):3d} trades | {round(wr,1)}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- BNR vs No-BNR ---")
    for flag, label in [(True,"With BNR   "), (False,"Without BNR")]:
        sub = results[results["bnr_active"]==flag]
        if len(sub):
            wr = (sub["outcome"]=="WIN").sum()/len(sub)*100
            print(f"  {label}: {len(sub):3d} trades | {round(wr,1)}% win")

    print(f"\n  --- Score Threshold ---")
    print(f"  {'Min Score':<12} {'Trades':>6} {'Win%':>7} {'P&L':>12} {'Return':>8}")
    print(f"  {'-'*50}")
    for t in [6,7,8,9,10]:
        sub = results[results[["bull_score","bear_score"]].max(axis=1) >= t]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            ret = pnl / STARTING_BALANCE * 100
            print(f"  {str(t)+'/10':<12} {len(sub):>6} {round(wr,1):>6}% ${pnl:>+10,.0f} {ret:>+7.1f}%")

    print(f"\n{sep}\n")


# ─── EQUITY CURVE HTML ───────────────────────────────────────────────────────
def generate_equity_curve(equity_curve, results, max_dd, peak_equity):
    final_eq     = results["equity"].iloc[-1]
    total_return = (final_eq - STARTING_BALANCE) / STARTING_BALANCE * 100
    wins         = (results["outcome"] == "WIN").sum()
    total        = len(results)
    win_rate     = wins / total * 100

    dates   = [p["date"]   for p in equity_curve]
    equities = [p["equity"] for p in equity_curve]

    # Trade markers for wins and losses
    win_points  = results[results["outcome"] == "WIN"][["date","equity"]].to_dict("records")
    loss_points = results[results["outcome"] == "LOSS"][["date","equity"]].to_dict("records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GBP/USD Backtest — Equity Curve</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ font-size: 20px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
  .subtitle {{ color: #7d8590; font-size: 13px; margin-bottom: 24px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .stat-label {{ font-size: 11px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .stat-value {{ font-size: 22px; font-weight: 700; }}
  .stat-value.green {{ color: #3fb950; }}
  .stat-value.red   {{ color: #f85149; }}
  .stat-value.blue  {{ color: #58a6ff; }}
  .stat-value.amber {{ color: #d29922; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .chart-title {{ font-size: 13px; font-weight: 600; color: #7d8590; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px; }}
  canvas {{ max-height: 380px; }}
  .breakdown {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .table {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
  .table h3 {{ font-size: 13px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #7d8590; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 8px; border-bottom: 1px solid #21262d; }}
  tr:last-child td {{ border-bottom: none; }}
  .win  {{ color: #3fb950; }}
  .loss {{ color: #f85149; }}
  .exp  {{ color: #d29922; }}
  .footer {{ text-align: center; color: #7d8590; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>

<h1>GBP/USD — Backtest Results</h1>
<p class="subtitle">
  {results['date'].iloc[0]} to {results['date'].iloc[-1]} &nbsp;·&nbsp;
  ${STARTING_BALANCE:,.0f} starting balance &nbsp;·&nbsp;
  ${RISK_PER_TRADE:,.0f} risk per trade (2%) &nbsp;·&nbsp;
  1:2 R:R
</p>

<div class="stats">
  <div class="stat">
    <div class="stat-label">Final Balance</div>
    <div class="stat-value {'green' if final_eq >= STARTING_BALANCE else 'red'}">${final_eq:,.0f}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Total Return</div>
    <div class="stat-value {'green' if total_return >= 0 else 'red'}">{total_return:+.1f}%</div>
  </div>
  <div class="stat">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value blue">{win_rate:.1f}%</div>
  </div>
  <div class="stat">
    <div class="stat-label">Total Trades</div>
    <div class="stat-value blue">{total}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Max Drawdown</div>
    <div class="stat-value {'amber' if max_dd < 15 else 'red'}">{max_dd:.1f}%</div>
  </div>
  <div class="stat">
    <div class="stat-label">Peak Balance</div>
    <div class="stat-value green">${peak_equity:,.0f}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Wins / Losses</div>
    <div class="stat-value">{wins}W / {(results['outcome']=='LOSS').sum()}L</div>
  </div>
  <div class="stat">
    <div class="stat-label">Signals / Year</div>
    <div class="stat-value blue">~{round(total/3)}</div>
  </div>
</div>

<div class="chart-container">
  <div class="chart-title">Equity Curve</div>
  <canvas id="equityChart"></canvas>
</div>

<div class="chart-container">
  <div class="chart-title">Drawdown (%)</div>
  <canvas id="drawdownChart"></canvas>
</div>

<div class="breakdown">
  <div class="table">
    <h3>By Market Condition</h3>
    <table>
      <tr><th>Condition</th><th>Trades</th><th>Win%</th><th>P&L</th></tr>
      {''.join(
        f"<tr><td>{lt}</td><td>{len(results[results['lt_trend']==lt])}</td>"
        f"<td>{round((results[results['lt_trend']==lt]['outcome']=='WIN').sum()/max(len(results[results['lt_trend']==lt]),1)*100,1)}%</td>"
        f"<td class=\"{'win' if results[results['lt_trend']==lt]['pnl'].sum()>=0 else 'loss'}\">"
        f"${results[results['lt_trend']==lt]['pnl'].sum():+,.0f}</td></tr>"
        for lt in ['BULLISH','BEARISH','SIDEWAYS'] if len(results[results['lt_trend']==lt])
      )}
    </table>
  </div>
  <div class="table">
    <h3>Score Threshold</h3>
    <table>
      <tr><th>Min Score</th><th>Trades</th><th>Win%</th><th>Return</th></tr>
      {''.join(
        f"<tr><td>{t}/10</td>"
        f"<td>{len(results[results[['bull_score','bear_score']].max(axis=1)>=t])}</td>"
        f"<td>{round((results[results[['bull_score','bear_score']].max(axis=1)>=t]['outcome']=='WIN').sum()/max(len(results[results[['bull_score','bear_score']].max(axis=1)>=t]),1)*100,1)}%</td>"
        f"<td class=\"{'win' if results[results[['bull_score','bear_score']].max(axis=1)>=t]['pnl'].sum()>=0 else 'loss'}\">"
        f"{results[results[['bull_score','bear_score']].max(axis=1)>=t]['pnl'].sum()/STARTING_BALANCE*100:+.1f}%</td></tr>"
        for t in [6,7,8,9,10] if len(results[results[['bull_score','bear_score']].max(axis=1)>=t])
      )}
    </table>
  </div>
</div>

<div class="table">
  <h3>Trade Log (last 20 trades)</h3>
  <table>
    <tr><th>Date</th><th>Direction</th><th>Score</th><th>LT Trend</th><th>BNR</th><th>FVG</th><th>Outcome</th><th>P&L</th><th>Balance</th></tr>
    {''.join(
      f"<tr>"
      f"<td>{r['date']}</td>"
      f"<td>{'🟢 BUY' if r['direction']=='BUY' else '🔴 SELL'}</td>"
      f"<td>{max(r['bull_score'],r['bear_score'])}/10</td>"
      f"<td>{r['lt_trend']}</td>"
      f"<td>{'✅' if r['bnr_active'] else '—'}</td>"
      f"<td>{'✅' if r['fvg_active'] else '—'}</td>"
      f"<td class=\"{'win' if r['outcome']=='WIN' else 'loss' if r['outcome']=='LOSS' else 'exp'}\">{r['outcome']}</td>"
      f"<td class=\"{'win' if r['pnl']>0 else 'loss' if r['pnl']<0 else ''}\">${r['pnl']:+,.0f}</td>"
      f"<td>${r['equity']:,.0f}</td>"
      f"</tr>"
      for _, r in results.tail(20).iterrows()
    )}
  </table>
</div>

<p class="footer">GBP/USD Trading System — Backtester v2.0 &nbsp;·&nbsp; Generated {datetime.now().strftime('%d %b %Y %H:%M')}</p>

<script>
const dates    = {json.dumps(dates)};
const equities = {json.dumps(equities)};

// Drawdown series
const drawdowns = [];
let peak = equities[0];
for (const e of equities) {{
  if (e > peak) peak = e;
  drawdowns.push(-((peak - e) / peak * 100));
}}

// Equity chart
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [
      {{
        label: 'Equity',
        data: equities,
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.08)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
      }},
      {{
        label: 'Starting Balance',
        data: Array(dates.length).fill({STARTING_BALANCE}),
        borderColor: '#444c56',
        borderWidth: 1,
        borderDash: [6,4],
        pointRadius: 0,
        fill: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#7d8590', font: {{ size: 12 }} }} }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#7d8590',
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: $` + ctx.parsed.y.toLocaleString('en-US', {{minimumFractionDigits:0}})
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#7d8590', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
      y: {{
        ticks: {{
          color: '#7d8590',
          callback: v => '$' + (v/1000).toFixed(0) + 'k'
        }},
        grid: {{ color: '#21262d' }}
      }}
    }}
  }}
}});

// Drawdown chart
new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [{{
      label: 'Drawdown %',
      data: drawdowns,
      borderColor: '#f85149',
      backgroundColor: 'rgba(248,81,73,0.1)',
      borderWidth: 1.5,
      fill: true,
      tension: 0.3,
      pointRadius: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#7d8590', font: {{ size: 12 }} }} }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#7d8590',
        callbacks: {{ label: ctx => ` Drawdown: ${{ctx.parsed.y.toFixed(1)}}%` }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#7d8590', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
      y: {{
        ticks: {{ color: '#7d8590', callback: v => v.toFixed(0) + '%' }},
        grid: {{ color: '#21262d' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open("equity_curve.html", "w") as f:
        f.write(html)
    print("Equity curve saved to equity_curve.html — open in any browser")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GBP/USD Backtester v2.0 — Equity Simulation")
    print(f"Account: ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade\n")
    print("Fetching data from OANDA...")

    df    = fetch_candles("D",  850)
    df    = calculate_indicators(df)
    df_4h = fetch_candles("H4", 4250)
    df_4h = calculate_indicators(df_4h)
    h4    = build_4h_lookup(df_4h)

    print(f"Daily : {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"4H    : {df_4h.index[0].strftime('%Y-%m-%d')} to {df_4h.index[-1].strftime('%Y-%m-%d')}\n")

    results, equity_curve, max_dd, peak_equity = run_backtest(df, h4)

    if not results.empty:
        analyse(results, equity_curve, max_dd, peak_equity)
        results.to_csv("backtest_results.csv", index=False)
        print("Trade log saved to backtest_results.csv")
        generate_equity_curve(equity_curve, results, max_dd, peak_equity)
