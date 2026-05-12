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

# ─── ACCOUNT ─────────────────────────────────────────────────────────────────
STARTING_BALANCE = 100_000.00
RISK_PER_TRADE   = 2_000.00
LOT_SIZE         = 100_000

PAIRS_TO_TEST = ["GBP_USD", "EUR_USD", "AUD_USD", "USD_JPY"]
COMMODITIES   = {"GBP_USD": "XAU_USD", "EUR_USD": "XAU_USD",
                 "AUD_USD": "XAU_USD", "USD_JPY": "WTICO_USD"}
BASES  = {"GBP_USD": "GBP", "EUR_USD": "EUR", "AUD_USD": "AUD", "USD_JPY": "USD"}
QUOTES = {"GBP_USD": "USD", "EUR_USD": "USD", "AUD_USD": "USD", "USD_JPY": "JPY"}

# ─── STRATEGY THRESHOLDS ─────────────────────────────────────────────────────
# Regime
ADX_TREND_MIN  = 25    # ADX above = trending
ADX_RANGE_MAX  = 20    # ADX below = ranging

# Strategy A — entry band: price within N x ATR of 50 EMA
# 0.5 was too tight (never fires). 1.5 = within ~1 daily range of EMA = realistic pullback
ATR_ENTRY_BAND = 1.5

# Strategy A — RSI at pullback entry
RSI_BULL_LO = 35       # oversold limit for longs
RSI_BULL_HI = 65       # overbought limit for longs (don't buy when extended)
RSI_BEAR_LO = 35       # oversold limit for shorts
RSI_BEAR_HI = 65       # overbought limit for shorts

# Strategy B — mean reversion
RSI_OVERSOLD   = 35    # buy at range low when RSI here or lower
RSI_OVERBOUGHT = 65    # sell at range high when RSI here or higher
RANGE_LOOKBACK = 30

# Execution
ATR_STOP_MULT   = 1.5   # stop distance from entry
ATR_TARGET_MULT = 3.0   # Strategy A target (1:2 R:R)
GBP_ATR_TARGET  = 2.0   # GBP/USD tighter target (1:1.33 R:R)
MAX_HOLD_A      = 20    # max candles to hold trend trade
MAX_HOLD_B      = 12    # max candles to hold mean reversion trade
MIN_SIGNAL_GAP  = 5     # min candles between signals on same pair

RATES = {"GBP": 3.75, "USD": 3.625, "EUR": 2.50, "AUD": 4.10, "JPY": 0.50}


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

    # ADX — Wilder's smoothing
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


# ─── LOOKUPS ─────────────────────────────────────────────────────────────────
def build_4h_lookup(df_4h):
    lookup = {}
    df_4h  = df_4h.copy()
    df_4h["date"] = df_4h.index.date
    for date, group in df_4h.groupby("date"):
        eod  = group[group.index.hour <= 20]
        last = (eod if not eod.empty else group).iloc[-1]
        lookup[str(date)] = {
            "close":  last["close"],
            "ema50":  last["ema50"],
            "ema200": last["ema200"],
            "rsi":    last["rsi"],
        }
    return lookup


def build_commodity_lookup(comm_df):
    lookup = {}
    comm_df = comm_df.copy()
    comm_df["date"] = comm_df.index.date
    dates_sorted = []
    for date, group in comm_df.groupby("date"):
        last = group.iloc[-1]
        lookup[str(date)] = {"close": last["close"], "ema50": last["ema50"]}
        dates_sorted.append(str(date))
    dates_sorted.sort()
    for idx, d in enumerate(dates_sorted):
        if idx >= 10:
            lookup[d]["ema50_10ago"] = lookup[dates_sorted[idx-10]]["ema50"]
        else:
            lookup[d]["ema50_10ago"] = lookup[d]["ema50"]
    return lookup


# ─── CONFIRMATION HELPERS ────────────────────────────────────────────────────
def get_4h_conf(lookup, date_str, direction):
    """Returns True if 4H agrees, True if no data (neutral — don't block)."""
    d = lookup.get(date_str)
    if not d:
        return True   # no 4H data for this date = neutral, don't block
    checks = 0
    if direction == "BUY":
        if d["close"] > d["ema50"]:  checks += 1
        if d["ema50"] > d["ema200"]: checks += 1
        if d["rsi"] < 65:            checks += 1
    else:
        if d["close"] < d["ema50"]:  checks += 1
        if d["ema50"] < d["ema200"]: checks += 1
        if d["rsi"] > 35:            checks += 1
    return checks >= 2


def get_comm_aligned(comm_lookup, date_str, direction):
    """Returns True if commodity aligns, True if no data (neutral)."""
    d = comm_lookup.get(date_str)
    if not d or d.get("ema50_10ago") is None:
        return True
    bullish = d["close"] > d["ema50"] and d["ema50"] > d["ema50_10ago"]
    if direction == "BUY":  return bullish
    if direction == "SELL": return not bullish
    return True


# ─── TIME FILTER ─────────────────────────────────────────────────────────────
def time_filter_at(date):
    """Returns (can_trade, size_modifier)."""
    wday  = date.weekday()
    month = date.month
    day   = date.day
    if wday == 4:                  return False, 1.0   # Friday
    if month == 12 and day >= 18:  return False, 1.0   # Year-end
    if month == 1  and day <= 5:   return False, 1.0   # New year
    if month == 8:                 return True,  0.5   # August reduced
    return True, 1.0


# ─── REGIME ──────────────────────────────────────────────────────────────────
def regime_at(df, i):
    """Returns (regime, direction_hint)."""
    if i < 5: return "AMBIGUOUS", "MIXED"
    adx  = df["adx"].iloc[i]
    pdi  = df["pdi"].iloc[i]
    ndi  = df["ndi"].iloc[i]
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]

    # TRENDING: ADX > threshold, directional structure clear
    if adx > ADX_TREND_MIN:
        if pdi > ndi and e50 > e200: return "TRENDING", "BULLISH"
        if ndi > pdi and e50 < e200: return "TRENDING", "BEARISH"
        return "AMBIGUOUS", "MIXED"

    # RANGING: ADX below threshold
    if adx < ADX_RANGE_MAX:
        return "RANGING", "NEUTRAL"

    return "AMBIGUOUS", "MIXED"


# ─── STRATEGY A: TREND FOLLOWING ─────────────────────────────────────────────
def strategy_a_at(df, i, h4_lookup, comm_lookup):
    """
    BUY/SELL signal for trending market.
    Price must pull back toward 50 EMA within ATR_ENTRY_BAND.
    ADX > 25, EMA structure aligned, RSI not extreme, 4H confirms.
    """
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

    # Price pulled back toward 50 EMA (within ATR_ENTRY_BAND × ATR)
    near_ema50 = abs(price - e50) <= ATR_ENTRY_BAND * atr

    # BUY: bullish structure + pullback + RSI not overbought
    if (adx > ADX_TREND_MIN
            and e50 > e200
            and near_ema50
            and RSI_BULL_LO <= rsi <= RSI_BULL_HI
            and prev["close"] > prev["open"]   # prior candle bullish
            and get_4h_conf(h4_lookup, date, "BUY")):
        # Commodity is a size modifier, not a blocker
        return "BUY"

    # SELL: bearish structure + pullback + RSI not oversold
    if (adx > ADX_TREND_MIN
            and e50 < e200
            and near_ema50
            and RSI_BEAR_LO <= rsi <= RSI_BEAR_HI
            and prev["close"] < prev["open"]   # prior candle bearish
            and get_4h_conf(h4_lookup, date, "SELL")):
        return "SELL"

    return None


# ─── STRATEGY B: MEAN REVERSION ──────────────────────────────────────────────
def strategy_b_at(df, i, h4_lookup, comm_lookup):
    """
    BUY/SELL signal for ranging market.
    Price at range boundary + RSI extreme + reversal candle.
    """
    if i < RANGE_LOOKBACK + 2: return None, None, None
    l      = df.iloc[i]
    recent = df.iloc[i-RANGE_LOOKBACK:i]

    price = l["close"]
    rsi   = l["rsi"]
    atr   = l["atr"]
    rh    = recent["high"].max()
    rl    = recent["low"].min()
    date  = df.index[i].strftime("%Y-%m-%d")

    # BUY at range low — confirm 4H shows bearish conditions that created the low (fading them)
    if (price <= rl + atr
            and rsi <= RSI_OVERSOLD
            and l["close"] > l["open"]         # bullish reversal candle
            and get_4h_conf(h4_lookup, date, "SELL")):
        return "BUY", rh, rl

    # SELL at range high — confirm 4H shows bullish conditions that created the high (fading them)
    if (price >= rh - atr
            and rsi >= RSI_OVERBOUGHT
            and l["close"] < l["open"]         # bearish reversal candle
            and get_4h_conf(h4_lookup, date, "BUY")):
        return "SELL", rh, rl

    return None, None, None


# ─── SIMULATE TRADE ──────────────────────────────────────────────────────────
def simulate(df, i, direction, atr, strategy, rh=None, rl=None, target_mult=ATR_TARGET_MULT):
    """Enter at next candle open. Returns outcome, hold, entry, exit_price."""
    if i + 1 >= len(df): return "EXPIRED", 0, 0, 0

    entry = df["open"].iloc[i + 1]

    if strategy == "A":
        stop     = entry - ATR_STOP_MULT * atr if direction == "BUY" else entry + ATR_STOP_MULT * atr
        target   = entry + target_mult * atr if direction == "BUY" else entry - target_mult * atr
        max_hold = MAX_HOLD_A
    else:
        rm       = (rh + rl) / 2
        stop     = rl - atr if direction == "BUY" else rh + atr
        target   = rm
        max_hold = MAX_HOLD_B

    for j in range(i + 1, min(i + max_hold + 1, len(df))):
        hi = df["high"].iloc[j]
        lo = df["low"].iloc[j]
        if direction == "BUY":
            if lo  <= stop:   return "LOSS",    j - i, entry, stop
            if hi  >= target: return "WIN",     j - i, entry, target
        else:
            if hi  >= stop:   return "LOSS",    j - i, entry, stop
            if lo  <= target: return "WIN",     j - i, entry, target

    return "EXPIRED", max_hold, entry, target


# ─── POSITION SIZING ─────────────────────────────────────────────────────────
def calc_lots(atr, price, pair, size_mod, comm_aligned, direction):
    """Volatility-adjusted position size with commodity and rate modifiers."""
    stop_dist = ATR_STOP_MULT * atr
    if QUOTES[pair] == "USD":
        stop_val_per_lot = stop_dist * LOT_SIZE
    else:
        stop_val_per_lot = (stop_dist / price) * LOT_SIZE

    mod = size_mod
    if not comm_aligned:
        mod *= 0.5   # commodity headwind = half size

    # Rate differential check
    diff = RATES.get(BASES[pair], 0) - RATES.get(QUOTES[pair], 0)
    if (direction == "BUY" and diff < -0.5) or (direction == "SELL" and diff > 0.5):
        mod *= 0.5   # against rate differential = half size

    effective_risk = RISK_PER_TRADE * mod
    lots           = round(effective_risk / max(stop_val_per_lot, 1), 2)
    return max(lots, 0.01), round(effective_risk, 0)


# ─── BACKTEST ────────────────────────────────────────────────────────────────
def run_backtest(all_data):
    results       = []
    equity        = STARTING_BALANCE
    peak          = STARTING_BALANCE
    max_dd        = 0.0
    equity_curve  = [{"date": "start", "equity": equity}]
    last_signal   = {p: -MIN_SIGNAL_GAP for p in PAIRS_TO_TEST}

    # Count signals per pair for diagnostics
    signal_counts = {p: {"A_buy": 0, "A_sell": 0, "B_buy": 0, "B_sell": 0,
                          "skipped_gap": 0} for p in PAIRS_TO_TEST}

    start_i = 210

    print(f"Scanning from candle {start_i}...\n")

    # Process day by day across all pairs to enforce portfolio rules
    # Collect all potential signals with their date index
    all_signals = []

    for pair in PAIRS_TO_TEST:
        df          = all_data[pair]["daily"]
        h4_lookup   = all_data[pair]["h4_lookup"]
        comm_lookup = all_data[pair]["comm_lookup"]

        for i in range(start_i, len(df) - 1):
            can_trade, size_mod = time_filter_at(df.index[i].date())
            if not can_trade: continue

            # Enforce minimum gap between signals on same pair
            if i - last_signal[pair] < MIN_SIGNAL_GAP:
                signal_counts[pair]["skipped_gap"] += 1
                continue

            regime, direction_hint = regime_at(df, i)

            if regime == "TRENDING":
                sig = strategy_a_at(df, i, h4_lookup, comm_lookup)
                if sig:
                    label = f"A_{sig.lower()}"
                    if label in signal_counts[pair]:
                        signal_counts[pair][label] += 1
                    all_signals.append({
                        "ts": df.index[i], "pair": pair, "strategy": "A",
                        "direction": sig, "i": i, "size_mod": size_mod,
                    })

            elif regime == "RANGING":
                sig, rh, rl = strategy_b_at(df, i, h4_lookup, comm_lookup)
                if sig:
                    label = f"B_{sig.lower()}"
                    if label in signal_counts[pair]:
                        signal_counts[pair][label] += 1
                    all_signals.append({
                        "ts": df.index[i], "pair": pair, "strategy": "B",
                        "direction": sig, "i": i, "size_mod": size_mod,
                        "rh": rh, "rl": rl,
                    })

    # Sort by date
    all_signals.sort(key=lambda x: x["ts"])

    print(f"Total raw signals before portfolio filter: {len(all_signals)}")

    # Execute signals with portfolio rules
    open_pairs  = set()   # pairs currently in a trade
    open_buys   = 0
    open_sells  = 0

    for s in all_signals:
        pair      = s["pair"]
        direction = s["direction"]
        i         = s["i"]
        strategy  = s["strategy"]

        # Skip if already in a trade on this pair
        if pair in open_pairs: continue

        # Portfolio rules
        if len(open_pairs) >= 4: continue
        if direction == "BUY"  and open_buys  >= 2: continue
        if direction == "SELL" and open_sells >= 2: continue

        df          = all_data[pair]["daily"]
        comm_lookup = all_data[pair]["comm_lookup"]
        atr         = df["atr"].iloc[i]
        price       = df["close"].iloc[i]
        date_str    = df.index[i].strftime("%Y-%m-%d")

        comm_ok     = get_comm_aligned(comm_lookup, date_str, direction)
        lots, eff_r = calc_lots(atr, price, pair, s["size_mod"], comm_ok, direction)
        rh          = s.get("rh")
        rl          = s.get("rl")

        t_mult  = GBP_ATR_TARGET if pair == "GBP_USD" else ATR_TARGET_MULT
        outcome, hold, entry, exit_price = simulate(
            df, i, direction, atr, strategy, rh, rl, t_mult
        )

        # P&L calculation
        if QUOTES[pair] == "USD":
            pnl_per_lot = ((exit_price - entry) * LOT_SIZE
                           if direction == "BUY"
                           else (entry - exit_price) * LOT_SIZE)
        else:
            pnl_per_lot = (((exit_price - entry) / price * LOT_SIZE)
                           if direction == "BUY"
                           else ((entry - exit_price) / price * LOT_SIZE))

        if outcome == "EXPIRED":
            pnl = -200 * lots
        else:
            pnl = pnl_per_lot * lots

        equity += pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

        last_signal[pair] = i
        regime_val, _     = regime_at(df, i)

        results.append({
            "date":      date_str,
            "pair":      pair,
            "strategy":  strategy,
            "direction": direction,
            "regime":    regime_val,
            "comm_ok":   comm_ok,
            "lots":      lots,
            "eff_risk":  eff_r,
            "entry":     round(entry, 5),
            "exit":      round(exit_price, 5),
            "outcome":   outcome,
            "pnl":       round(pnl, 2),
            "equity":    round(equity, 2),
            "dd_pct":    round(dd, 2),
            "hold":      hold,
        })

        equity_curve.append({"date": date_str, "equity": round(equity, 2)})

    # Print signal diagnostics
    print("\nSignals fired per pair (before portfolio filter):")
    for p, c in signal_counts.items():
        total_p = sum(v for k, v in c.items() if k != "skipped_gap")
        print(f"  {p:<10}: {total_p:3d} total  "
              f"(A-buy:{c['A_buy']} A-sell:{c['A_sell']} "
              f"B-buy:{c['B_buy']} B-sell:{c['B_sell']} "
              f"skipped_gap:{c['skipped_gap']})")

    return pd.DataFrame(results), equity_curve, max_dd, peak


# ─── ANALYSE ─────────────────────────────────────────────────────────────────
def analyse(results, max_dd, peak):
    if results.empty:
        print("\nNo signals fired with current rules.")
        return

    total    = len(results)
    wins     = (results["outcome"] == "WIN").sum()
    losses   = (results["outcome"] == "LOSS").sum()
    expired  = (results["outcome"] == "EXPIRED").sum()
    win_rate = wins / total * 100
    final    = results["equity"].iloc[-1]
    ret      = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
    years    = 2.5

    sep = "=" * 62
    print(f"\n{sep}")
    print("  MULTI-PAIR BACKTEST RESULTS")
    print(f"  Period : {results['date'].iloc[0]}  to  {results['date'].iloc[-1]}")
    print(f"  Account: ${STARTING_BALANCE:,.0f}  |  ${RISK_PER_TRADE:,.0f} base risk/trade")
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
        sub = results[results["pair"] == p]
        if len(sub):
            wr  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {p:<10}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- By Strategy ---")
    for s, label in [("A", "Trend Following"), ("B", "Mean Reversion")]:
        sub = results[results["strategy"] == s]
        if len(sub):
            wr  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {s} ({label:<16}): {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- By Regime ---")
    for reg in ["TRENDING", "RANGING"]:
        sub = results[results["regime"] == reg]
        if len(sub):
            wr  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {reg:<10}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  --- Commodity Filter Impact ---")
    for ok, label in [(True, "Aligned  "), (False, "Contrary ")]:
        sub = results[results["comm_ok"] == ok]
        if len(sub):
            wr  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl = sub["pnl"].sum()
            print(f"  {label}: {len(sub):3d} trades | {wr:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n{sep}\n")


# ─── HTML EQUITY CURVE ───────────────────────────────────────────────────────
def generate_html(equity_curve, results, max_dd, peak):
    final  = results["equity"].iloc[-1]
    ret    = (final - STARTING_BALANCE) / STARTING_BALANCE * 100
    total  = len(results)
    wins   = (results["outcome"] == "WIN").sum()
    losses = (results["outcome"] == "LOSS").sum()
    exp    = (results["outcome"] == "EXPIRED").sum()
    wr     = wins / total * 100 if total else 0

    dates = [p["date"] for p in equity_curve]
    eqs   = [p["equity"] for p in equity_curve]

    dds = []
    pk  = eqs[0]
    for e in eqs:
        if e > pk: pk = e
        dds.append(-((pk - e) / pk * 100))

    trows = ""
    for _, r in results.tail(30).iterrows():
        oc = "win" if r["outcome"]=="WIN" else "loss" if r["outcome"]=="LOSS" else "exp"
        pc = "win" if r["pnl"]>0 else "loss" if r["pnl"]<0 else ""
        trows += (f"<tr>"
                  f"<td>{r['date']}</td><td>{r['pair'].replace('_','/')}</td>"
                  f"<td>{r['strategy']}</td><td>{r['direction']}</td>"
                  f"<td>{r['regime']}</td><td>{'Y' if r['comm_ok'] else 'N'}</td>"
                  f"<td class='{oc}'>{r['outcome']}</td>"
                  f"<td class='{pc}'>${r['pnl']:+,.0f}</td>"
                  f"<td>{r['lots']}</td><td>${r['equity']:,.0f}</td>"
                  f"</tr>\n")

    prows = ""
    for p in PAIRS_TO_TEST:
        sub = results[results["pair"] == p]
        if len(sub):
            wr2  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl2 = sub["pnl"].sum()
            pc   = "win" if pnl2 >= 0 else "loss"
            prows += (f"<tr><td>{p.replace('_','/')}</td><td>{len(sub)}</td>"
                      f"<td>{wr2:.1f}%</td><td class='{pc}'>${pnl2:+,.0f}</td></tr>\n")

    srows = ""
    for s, label in [("A","Trend"),("B","Range")]:
        sub = results[results["strategy"] == s]
        if len(sub):
            wr2  = (sub["outcome"] == "WIN").sum() / len(sub) * 100
            pnl2 = sub["pnl"].sum()
            pc   = "win" if pnl2 >= 0 else "loss"
            srows += (f"<tr><td>Strategy {s} — {label}</td><td>{len(sub)}</td>"
                      f"<td>{wr2:.1f}%</td><td class='{pc}'>${pnl2:+,.0f}</td></tr>\n")

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
<h1>Multi-Pair Forex Backtest — Regime-Adaptive v2.1</h1>
<p class="sub">
  {results['date'].iloc[0]} to {results['date'].iloc[-1]} &nbsp;|&nbsp;
  GBP/USD EUR/USD AUD/USD USD/JPY &nbsp;|&nbsp;
  ${STARTING_BALANCE:,.0f} account &nbsp;|&nbsp; ${RISK_PER_TRADE:,.0f} base risk/trade &nbsp;|&nbsp;
  Strategy A: Trend (ADX>{ADX_TREND_MIN}, ±{ATR_ENTRY_BAND}×ATR of 50EMA, 1:{ATR_TARGET_MULT/ATR_STOP_MULT:.0f} RR) &nbsp;|&nbsp;
  Strategy B: Range (ADX&lt;{ADX_RANGE_MAX}, boundary+RSI)
</p>

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
    <tr><th>Date</th><th>Pair</th><th>S</th><th>Dir</th><th>Regime</th>
        <th>Comm</th><th>Result</th><th>P&amp;L</th><th>Lots</th><th>Balance</th></tr>
    {trows}
  </table>
</div>
<p class="footer">Regime-Adaptive Forex System v2.1 &nbsp;|&nbsp; Generated {gen}</p>

<script>
const dates={json.dumps(dates)};
const eq={json.dumps(eqs)};
const dd={json.dumps(dds)};

const cfg=(id,ds,yfmt)=>new Chart(document.getElementById(id),{{
  type:'line',data:{{labels:dates,datasets:ds}},
  options:{{responsive:true,
    interaction:{{mode:'index',intersect:false}},
    plugins:{{
      legend:{{labels:{{color:'#7d8590',font:{{size:11}}}}}},
      tooltip:{{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,
        titleColor:'#e6edf3',bodyColor:'#7d8590',
        callbacks:{{label:c=>yfmt(c)}}}}
    }},
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
</script>
</body></html>"""

    with open("equity_curve.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Equity curve saved to equity_curve.html")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Pair Backtest v2.1 — Regime-Adaptive")
    print(f"Pairs: {', '.join(PAIRS_TO_TEST)}")
    print(f"Strategy A: ADX>{ADX_TREND_MIN}, entry within {ATR_ENTRY_BAND}x ATR of 50 EMA, 1:{ATR_TARGET_MULT/ATR_STOP_MULT:.0f} R:R")
    print(f"Strategy B: ADX<{ADX_RANGE_MAX}, range boundary + RSI {RSI_OVERSOLD}/{RSI_OVERBOUGHT}")
    print(f"Account: ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade\n")
    print("Fetching data from OANDA...")

    all_data  = {}
    comm_data = {}

    # Fetch commodity data
    for comm in set(COMMODITIES.values()):
        try:
            cdf = fetch_candles(comm, "D", 850)
            cdf = calculate_indicators(cdf)
            comm_data[comm] = build_commodity_lookup(cdf)
            print(f"  {comm}: {len(cdf)} candles loaded")
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
            print(f"  {pair}: {len(df)} daily + {len(df_4h)} 4H candles")
        except Exception as e:
            print(f"  {pair} failed: {e}")

    if not all_data:
        print("No data loaded. Check API key and .env file.")
    else:
        first = list(all_data.keys())[0]
        d0    = all_data[first]["daily"].index[0].strftime("%Y-%m-%d")
        d1    = all_data[first]["daily"].index[-1].strftime("%Y-%m-%d")
        print(f"\nData: {d0} to {d1}\n")

        results, equity_curve, max_dd, peak = run_backtest(all_data)

        if not results.empty:
            analyse(results, max_dd, peak)
            results.to_csv("backtest_results.csv", index=False)
            print("Trade log: backtest_results.csv")
            generate_html(equity_curve, results, max_dd, peak)
        else:
            print("\nZero signals fired. Check the diagnostic output above.")
