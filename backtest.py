import os
import json
import requests
import numpy as np
import pandas as pd
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

# ─── SETTINGS ────────────────────────────────────────────────────────────────
STARTING_BALANCE = 100_000.00
RISK_PER_TRADE   = 2_000.00
LOT_SIZE         = 100_000

PAIRS_TO_TEST = ["GBP_USD", "EUR_USD", "AUD_USD", "USD_JPY"]
COMMODITIES   = {"GBP_USD": "XAU_USD", "EUR_USD": "XAU_USD",
                 "AUD_USD": "XAU_USD", "USD_JPY": "WTICO_USD"}
PIP           = {"GBP_USD": 0.0001, "EUR_USD": 0.0001,
                 "AUD_USD": 0.0001, "USD_JPY": 0.01}

ADX_TREND_MIN  = 25
ADX_RANGE_MAX  = 20
RSI_BUY_MAX    = 58
RSI_BUY_MIN    = 38
RSI_SELL_MAX   = 62
RSI_SELL_MIN   = 42
RSI_OVERSOLD   = 32
RSI_OVERBOUGHT = 68
ATR_ENTRY_MULT = 0.5
ATR_STOP_MULT  = 1.5
RANGE_LOOKBACK = 30

RATES = {"GBP": 3.75, "USD": 3.625, "EUR": 2.50, "AUD": 4.10, "JPY": 0.50}
BASES = {"GBP_USD": "GBP", "EUR_USD": "EUR", "AUD_USD": "AUD", "USD_JPY": "USD"}
QUOTES= {"GBP_USD": "USD", "EUR_USD": "USD", "AUD_USD": "USD", "USD_JPY": "JPY"}


# ─── FETCH ───────────────────────────────────────────────────────────────────
def fetch_candles(instrument, granularity="D", count=800):
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

    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    pos_dm    = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm    = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_s      = df["tr"].ewm(com=13, adjust=False).mean()
    pdm_s     = pd.Series(pos_dm, index=df.index).ewm(com=13, adjust=False).mean()
    ndm_s     = pd.Series(neg_dm, index=df.index).ewm(com=13, adjust=False).mean()
    df["pdi"] = 100 * pdm_s / tr_s
    df["ndi"] = 100 * ndm_s / tr_s
    dx        = 100 * (df["pdi"] - df["ndi"]).abs() / (df["pdi"] + df["ndi"])
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    return df


# ─── BUILD 4H LOOKUP ─────────────────────────────────────────────────────────
def build_4h_lookup(df_4h):
    lookup = {}
    df_4h  = df_4h.copy()
    df_4h["date"] = df_4h.index.date
    for date, group in df_4h.groupby("date"):
        eod  = group[group.index.hour <= 20]
        last = (eod if not eod.empty else group).iloc[-1]
        lookup[str(date)] = {
            "close": last["close"], "ema50": last["ema50"],
            "ema200": last["ema200"], "rsi": last["rsi"],
        }
    return lookup


def get_4h_conf(lookup, date_str, direction):
    d = lookup.get(date_str)
    if not d: return False
    checks = 0
    if direction == "BUY":
        if d["close"] > d["ema50"]:  checks += 1
        if d["ema50"] > d["ema200"]: checks += 1
        if d["rsi"] < 35 or 35 <= d["rsi"] <= 68: checks += 1
    else:
        if d["close"] < d["ema50"]:  checks += 1
        if d["ema50"] < d["ema200"]: checks += 1
        if d["rsi"] > 65 or 32 <= d["rsi"] <= 65: checks += 1
    return checks >= 2


# ─── COMMODITY LOOKUP ────────────────────────────────────────────────────────
def build_commodity_lookup(comm_df):
    lookup = {}
    comm_df = comm_df.copy()
    comm_df["date"] = comm_df.index.date
    for date, group in comm_df.groupby("date"):
        last = group.iloc[-1]
        lookup[str(date)] = {
            "close": last["close"],
            "ema50": last["ema50"],
            "ema50_10ago": None,
        }
    # Fill 10-ago slope
    dates = sorted(lookup.keys())
    for idx, d in enumerate(dates):
        if idx >= 10:
            lookup[d]["ema50_10ago"] = lookup[dates[idx-10]]["ema50"]
    return lookup


def get_comm_aligned(comm_lookup, date_str, direction):
    d = comm_lookup.get(date_str)
    if not d: return True  # neutral — don't penalise
    if d["ema50_10ago"] is None: return True
    bullish = d["close"] > d["ema50"] and d["ema50"] > d["ema50_10ago"]
    if direction == "BUY":  return bullish
    if direction == "SELL": return not bullish
    return True


# ─── TIME FILTER ─────────────────────────────────────────────────────────────
def time_filter_at(date):
    wday  = date.weekday()
    month = date.month
    day   = date.day
    if wday == 4:                     return False, 1.0   # Friday
    if month == 12 and day >= 18:     return False, 1.0   # Year-end
    if month == 1  and day <= 5:      return False, 1.0   # New year
    if month == 8:                    return True,  0.5   # August
    return True, 1.0


# ─── RATE DIFFERENTIAL ───────────────────────────────────────────────────────
def rate_diff_modifier(pair, direction):
    base  = BASES[pair]
    quote = QUOTES[pair]
    diff  = RATES.get(base, 0) - RATES.get(quote, 0)
    if direction == "BUY"  and diff < -0.5: return 0.5
    if direction == "SELL" and diff >  0.5: return 0.5
    return 1.0


# ─── REGIME ──────────────────────────────────────────────────────────────────
def regime_at(df, i):
    if i < 4: return "AMBIGUOUS", "MIXED"
    adx  = df["adx"].iloc[i]
    pdi  = df["pdi"].iloc[i]
    ndi  = df["ndi"].iloc[i]
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]
    adx_rising = adx > df["adx"].iloc[i-4]

    if adx > ADX_TREND_MIN and adx_rising:
        if pdi > ndi and e50 > e200: return "TRENDING", "BULLISH"
        if ndi > pdi and e50 < e200: return "TRENDING", "BEARISH"
        return "AMBIGUOUS", "MIXED"
    if adx < ADX_RANGE_MAX:
        return "RANGING", "NEUTRAL"
    return "AMBIGUOUS", "MIXED"


# ─── STRATEGY A SIGNAL ───────────────────────────────────────────────────────
def strategy_a_at(df, i, h4_lookup, comm_lookup, pair):
    if i < 5: return None
    l    = df.iloc[i]
    prev = df.iloc[i-1]
    price = l["close"]
    e50   = l["ema50"]
    e200  = l["ema200"]
    rsi   = l["rsi"]
    atr   = l["atr"]
    adx   = l["adx"]
    date  = df.index[i].strftime("%Y-%m-%d")

    near_ema50  = abs(price - e50) <= ATR_ENTRY_MULT * atr
    adx_rising  = adx > df["adx"].iloc[i-4] if i >= 4 else False

    # BUY
    if (adx > ADX_TREND_MIN and adx_rising and near_ema50
            and e50 > e200 and price >= e50
            and RSI_BUY_MIN <= rsi <= RSI_BUY_MAX
            and prev["close"] > prev["open"]
            and get_4h_conf(h4_lookup, date, "BUY")
            and get_comm_aligned(comm_lookup, date, "BUY")):
        return "BUY"

    # SELL
    if (adx > ADX_TREND_MIN and adx_rising and near_ema50
            and e50 < e200 and price <= e50
            and RSI_SELL_MIN <= rsi <= RSI_SELL_MAX
            and prev["close"] < prev["open"]
            and get_4h_conf(h4_lookup, date, "SELL")
            and get_comm_aligned(comm_lookup, date, "SELL")):
        return "SELL"

    return None


# ─── STRATEGY B SIGNAL ───────────────────────────────────────────────────────
def strategy_b_at(df, i, h4_lookup, comm_lookup, pair):
    if i < RANGE_LOOKBACK: return None, None, None
    l      = df.iloc[i]
    recent = df.iloc[i-RANGE_LOOKBACK:i]
    price  = l["close"]
    rsi    = l["rsi"]
    atr    = l["atr"]
    rh     = recent["high"].max()
    rl     = recent["low"].min()
    date   = df.index[i].strftime("%Y-%m-%d")

    # BUY at support
    if (abs(price - rl) <= atr and rsi < RSI_OVERSOLD
            and l["close"] > l["open"]
            and get_4h_conf(h4_lookup, date, "BUY")
            and get_comm_aligned(comm_lookup, date, "BUY")):
        return "BUY", rh, rl

    # SELL at resistance
    if (abs(price - rh) <= atr and rsi > RSI_OVERBOUGHT
            and l["close"] < l["open"]
            and get_4h_conf(h4_lookup, date, "SELL")
            and get_comm_aligned(comm_lookup, date, "SELL")):
        return "SELL", rh, rl

    return None, None, None


# ─── SIMULATE TRADE ──────────────────────────────────────────────────────────
def simulate(df, i, direction, atr, strategy, rh=None, rl=None):
    if i+1 >= len(df): return "EXPIRED", 0, 0, 0

    entry = df["open"].iloc[i+1]

    if strategy == "A":
        stop   = entry - ATR_STOP_MULT * atr if direction=="BUY" else entry + ATR_STOP_MULT * atr
        target = entry + 2.5 * ATR_STOP_MULT * atr if direction=="BUY" else entry - 2.5 * ATR_STOP_MULT * atr
        max_hold = 25
    else:
        rm     = (rh + rl) / 2
        stop   = rl - atr if direction=="BUY" else rh + atr
        target = rm
        max_hold = 10

    for j in range(i+1, min(i+max_hold+1, len(df))):
        hi = df["high"].iloc[j]
        lo = df["low"].iloc[j]
        if direction == "BUY":
            if lo  <= stop:   return "LOSS", j-i, entry, stop
            if hi  >= target: return "WIN",  j-i, entry, target
        else:
            if hi  >= stop:   return "LOSS", j-i, entry, stop
            if lo  <= target: return "WIN",  j-i, entry, target

    return "EXPIRED", max_hold, entry, target


# ─── POSITION SIZE ───────────────────────────────────────────────────────────
def calc_lots(atr, price, pair, size_mod, comm_aligned, direction):
    stop_dist = ATR_STOP_MULT * atr
    if QUOTES[pair] == "USD":
        stop_val = stop_dist * LOT_SIZE
    else:
        stop_val = (stop_dist / price) * LOT_SIZE
    mod = size_mod
    if not comm_aligned:        mod *= 0.5
    if rate_diff_modifier(pair, direction) < 1.0: mod *= 0.5
    risk = RISK_PER_TRADE * mod
    return round(risk / max(stop_val, 1), 2)


# ─── RUN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest(all_data):
    results      = []
    equity       = STARTING_BALANCE
    peak         = STARTING_BALANCE
    max_dd       = 0.0
    equity_curve = [{"date": "start", "equity": equity}]
    open_positions = {}  # pair -> True if in trade
    last_signal    = {}  # pair -> candle index of last signal

    print(f"\nRunning backtest across {len(PAIRS_TO_TEST)} pairs...\n")

    # Get all signals across all pairs sorted by date
    all_signals = []

    for pair in PAIRS_TO_TEST:
        df          = all_data[pair]["daily"]
        h4_lookup   = all_data[pair]["h4_lookup"]
        comm_lookup = all_data[pair]["comm_lookup"]
        start_i     = max(205, RANGE_LOOKBACK + 5)

        for i in range(start_i, len(df)-1):
            can_trade, size_mod = time_filter_at(df.index[i].date())
            if not can_trade: continue
            if open_positions.get(pair): continue
            if i - last_signal.get(pair, -10) < 5: continue

            regime, direction_hint = regime_at(df, i)

            if regime == "TRENDING":
                sig = strategy_a_at(df, i, h4_lookup, comm_lookup, pair)
                if sig:
                    all_signals.append((df.index[i], pair, "A", sig, i, size_mod))

            elif regime == "RANGING":
                sig, rh, rl = strategy_b_at(df, i, h4_lookup, comm_lookup, pair)
                if sig:
                    all_signals.append((df.index[i], pair, "B", sig, i, size_mod, rh, rl))

    # Sort signals by date
    all_signals.sort(key=lambda x: x[0])

    for sig_data in all_signals:
        date, pair, strategy, direction, i, size_mod = sig_data[:6]
        rh = sig_data[6] if len(sig_data) > 6 else None
        rl = sig_data[7] if len(sig_data) > 7 else None

        # Portfolio rules: max 4 open, max 2 same direction
        if sum(open_positions.values()) >= 4: continue

        df    = all_data[pair]["daily"]
        atr   = df["atr"].iloc[i]
        price = df["close"].iloc[i]

        comm_lookup = all_data[pair]["comm_lookup"]
        date_str    = date.strftime("%Y-%m-%d")
        comm_ok     = get_comm_aligned(comm_lookup, date_str, direction)
        lots        = calc_lots(atr, price, pair, size_mod, comm_ok, direction)

        # Simulate
        outcome, hold, entry, exit_price = simulate(
            df, i, direction, atr, strategy, rh, rl
        )

        last_signal[pair] = i

        # P&L
        if QUOTES[pair] == "USD":
            pnl_per_lot = (exit_price - entry) * LOT_SIZE if direction=="BUY" else (entry - exit_price) * LOT_SIZE
        else:
            pnl_per_lot = ((exit_price - entry) / price * LOT_SIZE) if direction=="BUY" else ((entry - exit_price) / price * LOT_SIZE)

        if outcome == "EXPIRED":
            pnl = -200 * lots  # small time cost
        else:
            pnl = pnl_per_lot * lots

        equity += pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

        regime_val, _ = regime_at(df, i)

        results.append({
            "date":      date_str,
            "pair":      pair,
            "strategy":  strategy,
            "direction": direction,
            "regime":    regime_val,
            "comm_ok":   comm_ok,
            "lots":      lots,
            "entry":     round(entry, 5),
            "exit":      round(exit_price, 5),
            "outcome":   outcome,
            "pnl":       round(pnl, 2),
            "equity":    round(equity, 2),
            "dd_pct":    round(dd, 2),
            "hold":      hold,
        })

        equity_curve.append({"date": date_str, "equity": round(equity, 2)})

    return pd.DataFrame(results), equity_curve, max_dd, peak


# ─── ANALYSE ─────────────────────────────────────────────────────────────────
def analyse(results, max_dd, peak):
    if results.empty:
        print("No signals fired.")
        return

    total    = len(results)
    wins     = (results["outcome"] == "WIN").sum()
    losses   = (results["outcome"] == "LOSS").sum()
    expired  = (results["outcome"] == "EXPIRED").sum()
    win_rate = wins / total * 100
    final    = results["equity"].iloc[-1]
    ret      = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
    years    = 2.5

    sep = "=" * 60
    print(sep)
    print("  MULTI-PAIR BACKTEST RESULTS")
    print(f"  Period  : {results['date'].iloc[0]}  to  {results['date'].iloc[-1]}")
    print(f"  Account : ${STARTING_BALANCE:,.0f}  |  ${RISK_PER_TRADE:,.0f} base risk/trade")
    print(sep)
    print(f"\n  Total Signals  : {total}  (~{round(total/years):.0f}/year)")
    print(f"  Win Rate       : {round(win_rate,1)}%  ({wins}W / {losses}L / {expired}E)")
    print(f"\n  Starting       : ${STARTING_BALANCE:>12,.2f}")
    print(f"  Final          : ${final:>12,.2f}")
    print(f"  Return         : {ret:>+.1f}%")
    print(f"  Peak           : ${peak:>12,.2f}")
    print(f"  Max Drawdown   : {max_dd:.1f}%")

    print(f"\n  --- By Pair ---")
    for p in PAIRS_TO_TEST:
        sub = results[results["pair"]==p]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  {p:<10}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- By Strategy ---")
    for s, label in [("A","Trend Following"),("B","Mean Reversion")]:
        sub = results[results["strategy"]==s]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  Strategy {s} ({label:<16}): {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- By Regime ---")
    for reg in ["TRENDING","RANGING"]:
        sub = results[results["regime"]==reg]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  {reg:<10}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- Commodity Filter ---")
    for ok, label in [(True,"Aligned  "),(False,"Contrary ")]:
        sub = results[results["comm_ok"]==ok]
        if len(sub):
            wr  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl = sub["pnl"].sum()
            print(f"  {label}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n{sep}\n")


# ─── HTML EQUITY CURVE ───────────────────────────────────────────────────────
def generate_html(equity_curve, results, max_dd, peak):
    final = results["equity"].iloc[-1]
    ret   = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
    wins  = (results["outcome"]=="WIN").sum()
    total = len(results)
    wr    = wins/total*100 if total else 0
    losses= (results["outcome"]=="LOSS").sum()
    exp   = (results["outcome"]=="EXPIRED").sum()

    dates = [p["date"] for p in equity_curve]
    eqs   = [p["equity"] for p in equity_curve]

    # Drawdown series
    dds = []
    pk  = eqs[0]
    for e in eqs:
        if e > pk: pk = e
        dds.append(-((pk-e)/pk*100))

    # Trade rows (last 30)
    trows = ""
    for _, r in results.tail(30).iterrows():
        oc = "win" if r["outcome"]=="WIN" else "loss" if r["outcome"]=="LOSS" else "exp"
        pc = "win" if r["pnl"]>0 else "loss" if r["pnl"]<0 else ""
        trows += (f"<tr><td>{r['date']}</td><td>{r['pair'].replace('_','/')}</td>"
                  f"<td>{r['strategy']}</td><td>{r['direction']}</td><td>{r['regime']}</td>"
                  f"<td>{'Y' if r['comm_ok'] else 'N'}</td>"
                  f"<td class='{oc}'>{r['outcome']}</td>"
                  f"<td class='{pc}'>${r['pnl']:+,.0f}</td>"
                  f"<td>${r['equity']:,.0f}</td></tr>\n")

    # Pair summary rows
    prows = ""
    for p in PAIRS_TO_TEST:
        sub = results[results["pair"]==p]
        if len(sub):
            wr2  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl2 = sub["pnl"].sum()
            pc   = "win" if pnl2>=0 else "loss"
            prows += f"<tr><td>{p.replace('_','/')}</td><td>{len(sub)}</td><td>{wr2:.1f}%</td><td class='{pc}'>${pnl2:+,.0f}</td></tr>\n"

    # Strategy rows
    srows = ""
    for s, label in [("A","Trend"), ("B","Range")]:
        sub = results[results["strategy"]==s]
        if len(sub):
            wr2  = (sub["outcome"]=="WIN").sum()/len(sub)*100
            pnl2 = sub["pnl"].sum()
            pc   = "win" if pnl2>=0 else "loss"
            srows += f"<tr><td>Strategy {s} — {label}</td><td>{len(sub)}</td><td>{wr2:.1f}%</td><td class='{pc}'>${pnl2:+,.0f}</td></tr>\n"

    bal_c = "green" if final >= STARTING_BALANCE else "red"
    ret_c = "green" if ret >= 0 else "red"
    dd_c  = "amber" if max_dd < 20 else "red"
    gen   = datetime.now().strftime("%d %b %Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Multi-Pair Forex Backtest</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px}}
h1{{font-size:20px;font-weight:600;margin-bottom:4px}}
.sub{{color:#7d8590;font-size:13px;margin-bottom:24px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.sl{{font-size:11px;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.sv{{font-size:20px;font-weight:700}}
.green{{color:#3fb950}}.red{{color:#f85149}}.blue{{color:#58a6ff}}.amber{{color:#d29922}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:16px}}
.ct{{font-size:12px;font-weight:600;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}}
canvas{{max-height:340px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;color:#7d8590;font-weight:500;padding:5px 8px;border-bottom:1px solid #30363d}}
td{{padding:6px 8px;border-bottom:1px solid #21262d}}
tr:last-child td{{border-bottom:none}}
.win{{color:#3fb950}}.loss{{color:#f85149}}.exp{{color:#d29922}}
.footer{{text-align:center;color:#7d8590;font-size:11px;margin-top:16px}}
</style>
</head>
<body>
<h1>Multi-Pair Forex Backtest — Regime-Adaptive System</h1>
<p class="sub">{results['date'].iloc[0]} to {results['date'].iloc[-1]} &nbsp;|&nbsp;
4 Pairs: GBP/USD EUR/USD AUD/USD USD/JPY &nbsp;|&nbsp;
${STARTING_BALANCE:,.0f} account &nbsp;|&nbsp; ${RISK_PER_TRADE:,.0f} base risk/trade</p>

<div class="stats">
  <div class="stat"><div class="sl">Final Balance</div><div class="sv {bal_c}">${final:,.0f}</div></div>
  <div class="stat"><div class="sl">Total Return</div><div class="sv {ret_c}">{ret:+.1f}%</div></div>
  <div class="stat"><div class="sl">Win Rate</div><div class="sv blue">{wr:.1f}%</div></div>
  <div class="stat"><div class="sl">Total Trades</div><div class="sv blue">{total}</div></div>
  <div class="stat"><div class="sl">Max Drawdown</div><div class="sv {dd_c}">{max_dd:.1f}%</div></div>
  <div class="stat"><div class="sl">Peak Balance</div><div class="sv green">${peak:,.0f}</div></div>
  <div class="stat"><div class="sl">W / L / E</div><div class="sv">{wins}W {losses}L {exp}E</div></div>
  <div class="stat"><div class="sl">Trades/Year</div><div class="sv blue">~{round(total/2.5):.0f}</div></div>
</div>

<div class="card"><div class="ct">Equity Curve</div><canvas id="ec"></canvas></div>
<div class="card"><div class="ct">Drawdown (%)</div><canvas id="dd"></canvas></div>

<div class="grid2">
  <div class="card"><div class="ct">By Pair</div>
    <table><tr><th>Pair</th><th>Trades</th><th>Win%</th><th>P&amp;L</th></tr>{prows}</table>
  </div>
  <div class="card"><div class="ct">By Strategy</div>
    <table><tr><th>Strategy</th><th>Trades</th><th>Win%</th><th>P&amp;L</th></tr>{srows}</table>
  </div>
</div>

<div class="card"><div class="ct">Last 30 Trades</div>
  <table>
    <tr><th>Date</th><th>Pair</th><th>Strat</th><th>Dir</th><th>Regime</th><th>Comm</th><th>Result</th><th>P&amp;L</th><th>Balance</th></tr>
    {trows}
  </table>
</div>
<p class="footer">Regime-Adaptive Forex System v2.0 &nbsp;|&nbsp; Generated {gen}</p>

<script>
const dates={json.dumps(dates)};
const eq={json.dumps(eqs)};
const dd={json.dumps(dds)};

const cfg=(id,ds,yfmt)=>new Chart(document.getElementById(id),{{
  type:'line',data:{{labels:dates,datasets:ds}},
  options:{{responsive:true,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{legend:{{labels:{{color:'#7d8590',font:{{size:11}}}}}},
      tooltip:{{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
        titleColor:'#e6edf3',bodyColor:'#7d8590',
        callbacks:{{label:c=>yfmt(c)}}}}}},
    scales:{{
      x:{{ticks:{{color:'#7d8590',maxTicksLimit:10}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#7d8590',callback:yfmt}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});

cfg('ec',[
  {{label:'Equity',data:eq,borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.06)',
    borderWidth:2,fill:true,tension:0.3,pointRadius:0}},
  {{label:'Start',data:Array(dates.length).fill({STARTING_BALANCE}),
    borderColor:'#444c56',borderWidth:1,borderDash:[5,4],pointRadius:0,fill:false}}
],v=>typeof v==='number'?'$'+(v/1000).toFixed(0)+'k':v);

cfg('dd',[
  {{label:'Drawdown',data:dd,borderColor:'#f85149',backgroundColor:'rgba(248,81,73,0.08)',
    borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}}
],v=>typeof v==='number'?v.toFixed(1)+'%':v);
</script></body></html>"""

    with open("equity_curve.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Equity curve saved to equity_curve.html")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Pair Backtest v2.0 — Regime-Adaptive")
    print(f"Pairs: {', '.join(PAIRS_TO_TEST)}")
    print(f"Account: ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade\n")
    print("Fetching data from OANDA...")

    all_data = {}

    # Fetch commodity data
    comm_data = {}
    for comm in set(COMMODITIES.values()):
        try:
            cdf = fetch_candles(comm, "D", 850)
            cdf = calculate_indicators(cdf)
            comm_data[comm] = build_commodity_lookup(cdf)
            print(f"  {comm}: {len(cdf)} candles")
        except Exception as e:
            print(f"  {comm} failed: {e}")
            comm_data[comm] = {}

    # Fetch pair data
    for pair in PAIRS_TO_TEST:
        try:
            df    = fetch_candles(pair, "D",  850)
            df    = calculate_indicators(df)
            df_4h = fetch_candles(pair, "H4", 4250)
            df_4h = calculate_indicators(df_4h)
            h4_lk = build_4h_lookup(df_4h)
            comm  = COMMODITIES[pair]
            all_data[pair] = {
                "daily":       df,
                "h4_lookup":   h4_lk,
                "comm_lookup": comm_data.get(comm, {}),
            }
            print(f"  {pair}: {len(df)} daily  {len(df_4h)} 4H candles")
        except Exception as e:
            print(f"  {pair} failed: {e}")

    if not all_data:
        print("No data loaded. Check API key.")
    else:
        first_pair = list(all_data.keys())[0]
        first_date = all_data[first_pair]["daily"].index[0].strftime("%Y-%m-%d")
        last_date  = all_data[first_pair]["daily"].index[-1].strftime("%Y-%m-%d")
        print(f"\nData range: {first_date} to {last_date}\n")

        results, equity_curve, max_dd, peak = run_backtest(all_data)

        if not results.empty:
            analyse(results, max_dd, peak)
            results.to_csv("backtest_results.csv", index=False)
            print("Trade log saved to backtest_results.csv")
            generate_html(equity_curve, results, max_dd, peak)
        else:
            print("No signals fired. Check strategy thresholds.")
