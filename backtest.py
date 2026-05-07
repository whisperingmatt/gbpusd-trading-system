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

# ---- ACCOUNT SETTINGS -------------------------------------------------------
STARTING_BALANCE = 100_000.00
RISK_PER_TRADE   = 2_000.00
REWARD_RATIO     = 2.0
WIN_AMOUNT       = RISK_PER_TRADE * REWARD_RATIO   # +$4,000
LOSS_AMOUNT      = RISK_PER_TRADE                  # -$2,000
EXPIRED_AMOUNT   = -400.00

# ---- STRATEGY RULES ---------------------------------------------------------
MIN_SCORE        = 9     # raised from 8
MAX_HOLD_CANDLES = 8     # reduced from 20
ATR_ENTRY_BAND   = 1.0   # price must be within 1x ATR of 50 EMA


# ---- FETCH ------------------------------------------------------------------
def fetch_candles(granularity="D", count=800):
    url     = f"{OANDA_BASE}/instruments/{INSTRUMENT}/candles"
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
                "volume": int(c["volume"]),
            })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


# ---- INDICATORS -------------------------------------------------------------
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


# ---- 4H LOOKUP --------------------------------------------------------------
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
    if not d: return "NEUTRAL"
    bull = bear = 0
    if d["close"] > d["ema50"]:  bull += 1
    else:                         bear += 1
    if d["ema50"] > d["ema200"]: bull += 1
    else:                         bear += 1
    if 40 <= d["rsi"] <= 65:     bull += 1
    elif d["rsi"] < 35:           bull += 1
    elif d["rsi"] > 68:           bear += 1
    if bull >= 2:   return "BULL"
    elif bear >= 2: return "BEAR"
    return "NEUTRAL"


# ---- TREND ------------------------------------------------------------------
def lt_at(df, i):
    if i < 200: return "INSUFFICIENT"
    price = df["close"].iloc[i]
    e200  = df["ema200"].iloc[i]
    s20   = (e200 - df["ema200"].iloc[i-20]) / df["ema200"].iloc[i-20] * 100
    s50   = (e200 - df["ema200"].iloc[i-50]) / df["ema200"].iloc[i-50] * 100
    rng   = df["high"].iloc[i-199:i+1].max() - df["low"].iloc[i-199:i+1].min()
    rpct  = rng / e200 * 100
    if rpct > 8.0 and abs(s50) < 0.3:              return "SIDEWAYS"
    elif price > e200 and s20 > 0.05 and s50 > 0.1: return "BULLISH"
    elif price < e200 and s20 < -0.05 and s50 < -0.1: return "BEARISH"
    return "SIDEWAYS"

def st_at(df, i):
    if i < 5: return "NEUTRAL"
    price = df["close"].iloc[i]
    e20   = df["ema20"].iloc[i]
    slope = (e20 - df["ema20"].iloc[i-5]) / df["ema20"].iloc[i-5] * 100
    if price > e20 and slope > 0.02:    return "BULLISH"
    elif price < e20 and slope < -0.02: return "BEARISH"
    return "NEUTRAL"


# ---- BNR --------------------------------------------------------------------
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


# ---- FVG --------------------------------------------------------------------
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


# ---- FIB --------------------------------------------------------------------
def fib_at(df, i):
    if i < 50: return False, False
    sub  = df.iloc[i-50:i]
    sh   = sub["high"].max()
    sl   = sub["low"].min()
    diff = sh - sl
    p    = df["close"].iloc[i]
    a    = df["atr"].iloc[i]
    return abs(p-(sh-0.5*diff)) <= 0.5*a, abs(p-(sh-0.618*diff)) <= 0.5*a


# ---- SCORE ------------------------------------------------------------------
def score_at(df, i, h4_lookup):
    if i < 201: return 0, 0, "INSUFFICIENT"

    price = df["close"].iloc[i]
    e50   = df["ema50"].iloc[i]
    e200  = df["ema200"].iloc[i]
    rsi   = df["rsi"].iloc[i]
    atr   = df["atr"].iloc[i]

    lt  = lt_at(df, i)
    st  = st_at(df, i)
    h4  = get_4h_conf(h4_lookup, df.index[i].strftime("%Y-%m-%d"))
    bb, be    = bnr_at(df, i)
    fb, fer   = fvg_at(df, i)
    n50, n618 = fib_at(df, i)

    # ---- RULE 1: Price must be within ATR_ENTRY_BAND of 50 EMA ----
    near_ema50 = abs(price - e50) <= ATR_ENTRY_BAND * atr

    # ---- RULE 3: SIDEWAYS market requires BNR ----
    bnr_active = bb is not None or be is not None
    if lt == "SIDEWAYS" and not bnr_active:
        return 0, 0, "NO_TRADE"

    bull = bear = 0
    if lt == "BULLISH":  bull += 3
    elif lt == "BEARISH": bear += 3
    if st == "BULLISH":  bull += 2
    elif st == "BEARISH": bear += 2
    if h4 == "BULL":     bull += 2
    elif h4 == "BEAR":   bear += 2
    if near_ema50:
        if price >= e50: bull += 2
        else:            bear += 2
    else:
        # Price not near EMA50 — half points only
        if price > e50: bull += 1
        else:           bear += 1
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

    # ---- RULE 2: Momentum filter — prev candle must close in direction ----
    prev_bullish = df["close"].iloc[i] > df["open"].iloc[i]
    prev_bearish = df["close"].iloc[i] < df["open"].iloc[i]

    # ---- RULE 4 applied in simulate_trade (max hold = 8) ----

    if bull >= MIN_SCORE and h4 != "BEAR" and near_ema50:
        sig = "CONFIRMED_BUY"
    elif bear >= MIN_SCORE and h4 != "BULL" and near_ema50:
        sig = "CONFIRMED_SELL"
    elif bull >= 5: sig = "WATCH_BUY"
    elif bear >= 5: sig = "WATCH_SELL"
    else:           sig = "NO_TRADE"

    return bull, bear, sig


# ---- SIMULATE TRADE ---------------------------------------------------------
def simulate_trade(df, i, direction, atr):
    if i+1 >= len(df): return "EXPIRED", 0
    entry  = df["open"].iloc[i+1]
    stop   = entry - 1.5*atr if direction=="BUY" else entry + 1.5*atr
    target = entry + 3.0*atr if direction=="BUY" else entry - 3.0*atr

    for j in range(i+1, min(i+MAX_HOLD_CANDLES+1, len(df))):
        hi, lo = df["high"].iloc[j], df["low"].iloc[j]
        if direction == "BUY":
            if lo  <= stop:   return "LOSS", j-i
            if hi  >= target: return "WIN",  j-i
        else:
            if hi  >= stop:   return "LOSS", j-i
            if lo  <= target: return "WIN",  j-i
    return "EXPIRED", MAX_HOLD_CANDLES


# ---- RUN BACKTEST -----------------------------------------------------------
def run_backtest(df, h4_lookup):
    results       = []
    last_signal_i = -10
    equity        = STARTING_BALANCE
    peak_equity   = STARTING_BALANCE
    max_dd        = 0.0
    equity_curve  = [{"date": df.index[210].strftime("%Y-%m-%d"), "equity": equity}]

    print(f"Scanning {len(df)-210} candles with tightened rules...\n")
    print(f"  Min score       : {MIN_SCORE}/10")
    print(f"  Max hold        : {MAX_HOLD_CANDLES} candles")
    print(f"  Entry band      : within {ATR_ENTRY_BAND}x ATR of 50 EMA")
    print(f"  SIDEWAYS filter : BNR required\n")

    for i in range(210, len(df)-1):
        bull, bear, sig = score_at(df, i, h4_lookup)
        if sig not in ("CONFIRMED_BUY", "CONFIRMED_SELL"): continue
        if i - last_signal_i < 5: continue

        direction     = "BUY" if sig == "CONFIRMED_BUY" else "SELL"
        atr           = df["atr"].iloc[i]
        outcome, hold = simulate_trade(df, i, direction, atr)
        last_signal_i = i

        if outcome == "WIN":    pnl = WIN_AMOUNT
        elif outcome == "LOSS": pnl = -LOSS_AMOUNT
        else:                   pnl = EXPIRED_AMOUNT

        equity += pnl
        if equity > peak_equity: peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        if drawdown > max_dd: max_dd = drawdown

        entry    = df["open"].iloc[i+1] if i+1 < len(df) else df["close"].iloc[i]
        lt       = lt_at(df, i)
        h4       = get_4h_conf(h4_lookup, df.index[i].strftime("%Y-%m-%d"))
        bb, be   = bnr_at(df, i)
        fb, fe   = fvg_at(df, i)
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


# ---- ANALYSE ----------------------------------------------------------------
def analyse(results, max_dd, peak_equity):
    if results.empty:
        print("No signals fired with current rules.")
        print("Try lowering MIN_SCORE or ATR_ENTRY_BAND at the top of the file.")
        return

    total    = len(results)
    wins     = (results["outcome"] == "WIN").sum()
    losses   = (results["outcome"] == "LOSS").sum()
    expired  = (results["outcome"] == "EXPIRED").sum()
    win_rate = wins / total * 100
    final_eq = results["equity"].iloc[-1]
    total_ret = (final_eq - STARTING_BALANCE) / STARTING_BALANCE * 100
    r_total  = (wins * 2) - losses
    avg_r    = r_total / total

    sep = "=" * 58
    print(sep)
    print("  GBP/USD BACKTEST RESULTS")
    print(f"  Period  : {results['date'].iloc[0]}  to  {results['date'].iloc[-1]}")
    print(f"  Account : ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade")
    print(sep)
    print(f"\n  Total Signals    : {total}  (~{round(total/2.5):.0f}/year)")
    print(f"  Win Rate         : {round(win_rate,1)}%  ({wins}W / {losses}L / {expired}E)")
    print(f"  Total R          : {r_total:+.0f}R  |  Avg: {avg_r:+.2f}R/trade")
    print(f"\n  Starting Balance : ${STARTING_BALANCE:>12,.2f}")
    print(f"  Final Balance    : ${final_eq:>12,.2f}")
    print(f"  Total Return     : {total_ret:>+.1f}%")
    print(f"  Peak Balance     : ${peak_equity:>12,.2f}")
    print(f"  Max Drawdown     : {max_dd:.1f}%  (${peak_equity - results['equity'].min():,.2f})")

    print(f"\n  --- By Market Condition ---")
    for lt in ["BULLISH","BEARISH","SIDEWAYS"]:
        sub = results[results["lt_trend"] == lt]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {lt:<10}: {len(sub):3d} trades | {round(wr,1):5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- BNR vs No-BNR ---")
    for flag, label in [(True,"With BNR   "),(False,"Without BNR")]:
        sub = results[results["bnr_active"]==flag]
        if len(sub):
            wr = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  {label}: {len(sub):3d} trades | {round(wr,1):5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- FVG vs No-FVG ---")
    for flag, label in [(True,"With FVG   "),(False,"Without FVG")]:
        sub = results[results["fvg_active"]==flag]
        if len(sub):
            wr = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  {label}: {len(sub):3d} trades | {round(wr,1):5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- Score Threshold ---")
    print(f"  {'Min Score':<12} {'Trades':>6} {'Win%':>7} {'P&L':>12} {'Return':>8}")
    print(f"  {'-'*50}")
    for t in [7,8,9,10,11,12]:
        sub = results[results[["bull_score","bear_score"]].max(axis=1) >= t]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            ret = pnl / STARTING_BALANCE * 100
            print(f"  {str(t)+'/10':<12} {len(sub):>6} {round(wr,1):>6}% ${pnl:>+10,.0f} {ret:>+7.1f}%")

    print(f"\n{sep}\n")


# ---- EQUITY CURVE HTML (encoding fixed) ------------------------------------
def generate_equity_curve(equity_curve, results, max_dd, peak_equity):
    final_eq  = results["equity"].iloc[-1]
    total_ret = (final_eq - STARTING_BALANCE) / STARTING_BALANCE * 100
    wins      = (results["outcome"] == "WIN").sum()
    losses    = (results["outcome"] == "LOSS").sum()
    expired   = (results["outcome"] == "EXPIRED").sum()
    total     = len(results)
    win_rate  = wins / total * 100

    dates    = [p["date"]   for p in equity_curve]
    equities = [p["equity"] for p in equity_curve]

    # Build trade log rows (last 30)
    trade_rows = ""
    for _, r in results.tail(30).iterrows():
        direction = "BUY" if r["direction"] == "BUY" else "SELL"
        outcome   = r["outcome"]
        oc        = "win" if outcome == "WIN" else "loss" if outcome == "LOSS" else "exp"
        pc        = "win" if r["pnl"] > 0 else "loss" if r["pnl"] < 0 else ""
        bnr       = "YES" if r["bnr_active"] else "-"
        fvg       = "YES" if r["fvg_active"] else "-"
        trade_rows += (
            f"<tr>"
            f"<td>{r['date']}</td>"
            f"<td>{direction}</td>"
            f"<td>{max(r['bull_score'],r['bear_score'])}/10</td>"
            f"<td>{r['lt_trend']}</td>"
            f"<td>{bnr}</td>"
            f"<td>{fvg}</td>"
            f"<td class='{oc}'>{outcome}</td>"
            f"<td class='{pc}'>${r['pnl']:+,.0f}</td>"
            f"<td>${r['equity']:,.0f}</td>"
            f"</tr>\n"
        )

    # Build market condition rows
    mc_rows = ""
    for lt in ["BULLISH","BEARISH","SIDEWAYS"]:
        sub = results[results["lt_trend"]==lt]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            pc  = "win" if pnl >= 0 else "loss"
            mc_rows += (
                f"<tr><td>{lt}</td><td>{len(sub)}</td>"
                f"<td>{wr:.1f}%</td>"
                f"<td class='{pc}'>${pnl:+,.0f}</td></tr>\n"
            )

    # Build score rows
    sc_rows = ""
    for t in [7,8,9,10,11,12]:
        sub = results[results[["bull_score","bear_score"]].max(axis=1) >= t]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            ret = pnl / STARTING_BALANCE * 100
            pc  = "win" if pnl >= 0 else "loss"
            sc_rows += (
                f"<tr><td>{t}/10</td><td>{len(sub)}</td>"
                f"<td>{wr:.1f}%</td>"
                f"<td class='{pc}'>{ret:+.1f}%</td></tr>\n"
            )

    bal_color   = "green" if final_eq >= STARTING_BALANCE else "red"
    ret_color   = "green" if total_ret >= 0 else "red"
    dd_color    = "amber" if max_dd < 15 else "red"
    generated   = datetime.now().strftime("%d %b %Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GBP/USD Backtest - Equity Curve</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px}}
h1{{font-size:20px;font-weight:600;color:#fff;margin-bottom:4px}}
.sub{{color:#7d8590;font-size:13px;margin-bottom:24px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.slabel{{font-size:11px;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.sval{{font-size:22px;font-weight:700}}
.green{{color:#3fb950}}.red{{color:#f85149}}.blue{{color:#58a6ff}}.amber{{color:#d29922}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:20px}}
.ctitle{{font-size:12px;font-weight:600;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:16px}}
canvas{{max-height:360px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:#7d8590;font-weight:500;padding:6px 8px;border-bottom:1px solid #30363d}}
td{{padding:7px 8px;border-bottom:1px solid #21262d}}
tr:last-child td{{border-bottom:none}}
.win{{color:#3fb950}}.loss{{color:#f85149}}.exp{{color:#d29922}}
.footer{{text-align:center;color:#7d8590;font-size:12px;margin-top:20px}}
.rules{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:20px;font-size:13px;color:#7d8590}}
.rules span{{color:#e6edf3;font-weight:600}}
</style>
</head>
<body>
<h1>GBP/USD Backtest Results</h1>
<p class="sub">{results['date'].iloc[0]} to {results['date'].iloc[-1]} &nbsp;|&nbsp; ${STARTING_BALANCE:,.0f} starting balance &nbsp;|&nbsp; ${RISK_PER_TRADE:,.0f} risk/trade (2%) &nbsp;|&nbsp; 1:2 R:R</p>

<div class="rules">
  Strategy rules: <span>Min score {MIN_SCORE}/10</span> &nbsp;|&nbsp;
  <span>Max hold {MAX_HOLD_CANDLES} candles</span> &nbsp;|&nbsp;
  <span>Entry within {ATR_ENTRY_BAND}x ATR of 50 EMA</span> &nbsp;|&nbsp;
  <span>SIDEWAYS market: BNR required</span>
</div>

<div class="stats">
  <div class="stat"><div class="slabel">Final Balance</div><div class="sval {bal_color}">${final_eq:,.0f}</div></div>
  <div class="stat"><div class="slabel">Total Return</div><div class="sval {ret_color}">{total_ret:+.1f}%</div></div>
  <div class="stat"><div class="slabel">Win Rate</div><div class="sval blue">{win_rate:.1f}%</div></div>
  <div class="stat"><div class="slabel">Total Trades</div><div class="sval blue">{total}</div></div>
  <div class="stat"><div class="slabel">Max Drawdown</div><div class="sval {dd_color}">{max_dd:.1f}%</div></div>
  <div class="stat"><div class="slabel">Peak Balance</div><div class="sval green">${peak_equity:,.0f}</div></div>
  <div class="stat"><div class="slabel">W / L / E</div><div class="sval">{wins}W {losses}L {expired}E</div></div>
  <div class="stat"><div class="slabel">Trades/Year</div><div class="sval blue">~{round(total/2.5):.0f}</div></div>
</div>

<div class="card">
  <div class="ctitle">Equity Curve</div>
  <canvas id="ec"></canvas>
</div>

<div class="card">
  <div class="ctitle">Drawdown (%)</div>
  <canvas id="dd"></canvas>
</div>

<div class="grid2">
  <div class="card">
    <div class="ctitle">By Market Condition</div>
    <table>
      <tr><th>Condition</th><th>Trades</th><th>Win%</th><th>P&amp;L</th></tr>
      {mc_rows}
    </table>
  </div>
  <div class="card">
    <div class="ctitle">Score Threshold</div>
    <table>
      <tr><th>Min Score</th><th>Trades</th><th>Win%</th><th>Return</th></tr>
      {sc_rows}
    </table>
  </div>
</div>

<div class="card">
  <div class="ctitle">Last 30 Trades</div>
  <table>
    <tr><th>Date</th><th>Dir</th><th>Score</th><th>LT</th><th>BNR</th><th>FVG</th><th>Outcome</th><th>P&amp;L</th><th>Balance</th></tr>
    {trade_rows}
  </table>
</div>

<p class="footer">GBP/USD Trading System v2.0 &nbsp;|&nbsp; Generated {generated}</p>

<script>
const dates={json.dumps(dates)};
const eq={json.dumps(equities)};
const dd=[];
let pk=eq[0];
for(const e of eq){{if(e>pk)pk=e;dd.push(-((pk-e)/pk*100));}}

const cfg=(id,ds,yFmt)=>new Chart(document.getElementById(id),{{
  type:'line',data:{{labels:dates,datasets:ds}},
  options:{{
    responsive:true,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{labels:{{color:'#7d8590',font:{{size:12}}}}}},
      tooltip:{{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
        titleColor:'#e6edf3',bodyColor:'#7d8590',
        callbacks:{{label:c=>yFmt(c)}}}}
    }},
    scales:{{
      x:{{ticks:{{color:'#7d8590',maxTicksLimit:10}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#7d8590',callback:yFmt}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});

cfg('ec',[
  {{label:'Equity',data:eq,borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.08)',
    borderWidth:2,fill:true,tension:0.3,pointRadius:0}},
  {{label:'Start',data:Array(dates.length).fill({STARTING_BALANCE}),
    borderColor:'#444c56',borderWidth:1,borderDash:[5,4],pointRadius:0,fill:false}}
],v=>typeof v==='number'?'$'+(v/1000).toFixed(0)+'k':v);

cfg('dd',[
  {{label:'Drawdown',data:dd,borderColor:'#f85149',backgroundColor:'rgba(248,81,73,0.1)',
    borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}}
],v=>typeof v==='number'?v.toFixed(1)+'%':v);
</script>
</body>
</html>"""

    with open("equity_curve.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Equity curve saved to equity_curve.html")


# ---- MAIN -------------------------------------------------------------------
if __name__ == "__main__":
    print("GBP/USD Backtester v2.1 - Tightened Rules")
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
        analyse(results, max_dd, peak_equity)
        results.to_csv("backtest_results.csv", index=False)
        print("Trade log saved to backtest_results.csv")
        generate_equity_curve(equity_curve, results, max_dd, peak_equity)
    else:
        print("\nNo signals fired with current rules.")
        print(f"Try lowering MIN_SCORE (currently {MIN_SCORE}) at the top of the file.")
