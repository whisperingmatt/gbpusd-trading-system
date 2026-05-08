"""
Multi-Pair Forex Backtester v3.0
Uses identical pair-specific configs as main.py.
Tests Strategy A (trend) and Strategy B (range) independently per pair.
"""

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
STARTING_BALANCE  = 100_000.00
RISK_PER_TRADE    = 2_000.00
LOT_SIZE          = 100_000
MIN_SIGNAL_GAP    = 5     # min candles between signals on same pair
MAX_OPEN_TRADES   = 4     # max simultaneous open positions
MAX_SAME_SIDE     = 2     # max same direction at once

# ─── RATES ───────────────────────────────────────────────────────────────────
RATES = {"GBP": 3.75, "USD": 3.625, "EUR": 2.50, "AUD": 4.10, "JPY": 0.50}

# ─── PAIR CONFIGS (identical to main.py) ─────────────────────────────────────
PAIR_CONFIGS = {
    "GBP_USD": {
        "name": "GBP/USD", "base": "GBP", "quote": "USD", "pip": 0.0001,
        "strategy": "trend", "adx_trend_min": 25, "adx_range_max": 20,
        "atr_entry_band": 1.5,
        "rsi_buy_lo": 35, "rsi_buy_hi": 65,
        "rsi_sell_lo": 35, "rsi_sell_hi": 65,
        "atr_stop": 1.5, "atr_target": 3.0, "max_hold": 20,
        "range_lookback": 40, "rsi_oversold": 33, "rsi_overbought": 67,
        "range_target": 0.50, "atr_range_stop": 1.0, "max_hold_range": 12,
        "commodity": "XAU_USD", "commodity_label": "Gold",
        "commodity_required": False, "commodity_contrary_mod": 0.5,
        "rate_base": "GBP", "rate_quote": "USD",
        "intervention_long_above": None, "intervention_short_below": None,
        "base_size_mod": 1.0, "require_risk_on": False,
    },
    "EUR_USD": {
        "name": "EUR/USD", "base": "EUR", "quote": "USD", "pip": 0.0001,
        "strategy": "auto", "adx_trend_min": 28, "adx_range_max": 22,
        "atr_entry_band": 1.5,
        "rsi_buy_lo": 35, "rsi_buy_hi": 60,
        "rsi_sell_lo": 40, "rsi_sell_hi": 65,
        "atr_stop": 1.5, "atr_target": 2.5, "max_hold": 18,
        "range_lookback": 40, "rsi_oversold": 32, "rsi_overbought": 68,
        "range_target": 0.45, "atr_range_stop": 1.0, "max_hold_range": 12,
        "commodity": "XAU_USD", "commodity_label": "Gold",
        "commodity_required": False, "commodity_contrary_mod": 0.5,
        "rate_base": "EUR", "rate_quote": "USD",
        "intervention_long_above": None, "intervention_short_below": None,
        "base_size_mod": 1.0, "require_risk_on": False,
    },
    "AUD_USD": {
        "name": "AUD/USD", "base": "AUD", "quote": "USD", "pip": 0.0001,
        "strategy": "auto", "adx_trend_min": 22, "adx_range_max": 18,
        "atr_entry_band": 1.5,
        "rsi_buy_lo": 35, "rsi_buy_hi": 62,
        "rsi_sell_lo": 38, "rsi_sell_hi": 65,
        "atr_stop": 1.5, "atr_target": 3.0, "max_hold": 20,
        "range_lookback": 35, "rsi_oversold": 33, "rsi_overbought": 67,
        "range_target": 0.50, "atr_range_stop": 1.0, "max_hold_range": 12,
        "commodity": "XAU_USD", "commodity_label": "Gold",
        "commodity_required": True, "commodity_contrary_mod": 0.0,
        "rate_base": "AUD", "rate_quote": "USD",
        "intervention_long_above": None, "intervention_short_below": None,
        "base_size_mod": 1.0, "require_risk_on": True,
    },
    "USD_JPY": {
        "name": "USD/JPY", "base": "USD", "quote": "JPY", "pip": 0.01,
        "strategy": "trend", "adx_trend_min": 25, "adx_range_max": 20,
        "atr_entry_band": 1.5,
        "rsi_buy_lo": 38, "rsi_buy_hi": 62,
        "rsi_sell_lo": 38, "rsi_sell_hi": 62,
        "atr_stop": 1.5, "atr_target": 2.5, "max_hold": 15,
        "range_lookback": 30, "rsi_oversold": 35, "rsi_overbought": 65,
        "range_target": 0.45, "atr_range_stop": 1.0, "max_hold_range": 10,
        "commodity": "WTICO_USD", "commodity_label": "Oil",
        "commodity_required": False, "commodity_contrary_mod": 0.5,
        "rate_base": "USD", "rate_quote": "JPY",
        "intervention_long_above": 153.0, "intervention_short_below": 143.0,
        "base_size_mod": 0.75, "require_risk_on": False,
    },
}


# ─── DATA ────────────────────────────────────────────────────────────────────
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
                "time": c["time"][:19], "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


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
        df["high"]-df["low"],
        (df["high"]-prev).abs(),
        (df["low"]-prev).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(com=13, adjust=False).mean()
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    pdm  = np.where((up>down)&(up>0), up, 0.0)
    ndm  = np.where((down>up)&(down>0), down, 0.0)
    tr_s = df["tr"].ewm(com=13, adjust=False).mean()
    df["pdi"] = 100*pd.Series(pdm, index=df.index).ewm(com=13, adjust=False).mean()/tr_s
    df["ndi"] = 100*pd.Series(ndm, index=df.index).ewm(com=13, adjust=False).mean()/tr_s
    dx        = 100*(df["pdi"]-df["ndi"]).abs()/(df["pdi"]+df["ndi"])
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
            "close": last["close"], "ema50": last["ema50"],
            "ema200": last["ema200"], "rsi": last["rsi"],
        }
    return lookup


def build_comm_lookup(comm_df):
    lookup = {}
    comm_df = comm_df.copy()
    comm_df["date"] = comm_df.index.date
    dates = []
    for date, grp in comm_df.groupby("date"):
        last = grp.iloc[-1]
        lookup[str(date)] = {"close": last["close"], "ema50": last["ema50"]}
        dates.append(str(date))
    dates.sort()
    for idx, d in enumerate(dates):
        if idx >= 10:
            lookup[d]["ema50_10ago"] = lookup[dates[idx-10]]["ema50"]
        else:
            lookup[d]["ema50_10ago"] = lookup[d]["ema50"]
    return lookup


# ─── SIGNAL HELPERS AT CANDLE i ──────────────────────────────────────────────
def get_regime_i(df, i, cfg):
    if i < 5: return "AMBIGUOUS", "MIXED"
    adx  = df["adx"].iloc[i]
    pdi  = df["pdi"].iloc[i]
    ndi  = df["ndi"].iloc[i]
    e50  = df["ema50"].iloc[i]
    e200 = df["ema200"].iloc[i]
    if adx >= cfg["adx_trend_min"]:
        if pdi>ndi and e50>e200: return "TRENDING", "BULLISH"
        if ndi>pdi and e50<e200: return "TRENDING", "BEARISH"
        return "AMBIGUOUS", "MIXED"
    if adx <= cfg["adx_range_max"]:
        return "RANGING", "NEUTRAL"
    return "AMBIGUOUS", "MIXED"


def get_4h_conf_i(h4_lk, date_str, direction):
    d = h4_lk.get(date_str)
    if not d: return True   # neutral
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


def get_comm_aligned_i(comm_lk, date_str, direction, cfg):
    d = comm_lk.get(date_str)
    if not d: return True, 1.0
    bullish = d["close"] > d["ema50"] and d.get("ema50_10ago", d["ema50"]) <= d["ema50"]
    if direction == "BUY":
        if bullish:  return True,  1.0
        return False, cfg["commodity_contrary_mod"]
    else:
        if not bullish: return True,  1.0
        return False, cfg["commodity_contrary_mod"]


def get_risk_on_i(df, i, direction):
    if i < 20: return True
    if direction == "BUY":
        return df["high"].iloc[i-10:i].max() > df["high"].iloc[i-20:i-10].max()
    else:
        return df["low"].iloc[i-10:i].min() < df["low"].iloc[i-20:i-10].min()


def check_iv_i(price, direction, cfg):
    if direction == "BUY" and cfg["intervention_long_above"] and price > cfg["intervention_long_above"]:
        return False
    if direction == "SELL" and cfg["intervention_short_below"] and price < cfg["intervention_short_below"]:
        return False
    return True


def get_rate_mod_i(cfg, direction):
    diff = RATES.get(cfg["rate_base"], 0) - RATES.get(cfg["rate_quote"], 0)
    if direction == "BUY"  and diff < -0.5: return 0.5
    if direction == "SELL" and diff >  0.5: return 0.5
    return 1.0


def time_filter_i(date):
    wday  = date.weekday()
    month = date.month
    day   = date.day
    if wday == 4:                 return False, 1.0
    if month == 12 and day >= 18: return False, 1.0
    if month == 1  and day <= 5:  return False, 1.0
    if month == 8:                return True,  0.5
    return True, 1.0


# ─── STRATEGY A SIGNAL AT CANDLE i ───────────────────────────────────────────
def strat_a_at(df, i, h4_lk, comm_lk, cfg):
    if i < 5: return None, 1.0
    l    = df.iloc[i]
    prev = df.iloc[i-1]
    price = l["close"];  e50 = l["ema50"];  e200 = l["ema200"]
    rsi   = l["rsi"];    atr = l["atr"];    adx  = l["adx"]
    date  = df.index[i].strftime("%Y-%m-%d")
    near  = abs(price - e50) <= cfg["atr_entry_band"] * atr

    for direction in ("BUY", "SELL"):
        if direction == "BUY":
            if not (adx >= cfg["adx_trend_min"] and e50>e200 and near
                    and cfg["rsi_buy_lo"] <= rsi <= cfg["rsi_buy_hi"]
                    and prev["close"] > prev["open"]): continue
        else:
            if not (adx >= cfg["adx_trend_min"] and e50<e200 and near
                    and cfg["rsi_sell_lo"] <= rsi <= cfg["rsi_sell_hi"]
                    and prev["close"] < prev["open"]): continue

        if not get_4h_conf_i(h4_lk, date, direction): continue
        comm_ok, comm_mod = get_comm_aligned_i(comm_lk, date, direction, cfg)
        if not comm_ok and comm_mod == 0.0: continue
        if cfg["require_risk_on"] and not get_risk_on_i(df, i, direction): continue
        if not check_iv_i(price, direction, cfg): continue

        return direction, comm_mod

    return None, 1.0


# ─── STRATEGY B SIGNAL AT CANDLE i ───────────────────────────────────────────
def strat_b_at(df, i, h4_lk, comm_lk, cfg):
    if i < cfg["range_lookback"] + 2: return None, 1.0, None, None
    l      = df.iloc[i]
    recent = df.iloc[i-cfg["range_lookback"]:i]
    price  = l["close"];  rsi = l["rsi"];  atr = l["atr"]
    rh     = recent["high"].max();  rl  = recent["low"].min()
    date   = df.index[i].strftime("%Y-%m-%d")
    bull_c = l["close"] > l["open"]
    bear_c = l["close"] < l["open"]

    if price <= rl + atr and rsi <= cfg["rsi_oversold"] and bull_c:
        if not get_4h_conf_i(h4_lk, date, "BUY"): return None,1.0,None,None
        comm_ok, comm_mod = get_comm_aligned_i(comm_lk, date, "BUY", cfg)
        if not comm_ok and comm_mod == 0.0: return None,1.0,None,None
        return "BUY", comm_mod, rh, rl

    if price >= rh - atr and rsi >= cfg["rsi_overbought"] and bear_c:
        if not get_4h_conf_i(h4_lk, date, "SELL"): return None,1.0,None,None
        comm_ok, comm_mod = get_comm_aligned_i(comm_lk, date, "SELL", cfg)
        if not comm_ok and comm_mod == 0.0: return None,1.0,None,None
        return "SELL", comm_mod, rh, rl

    return None, 1.0, None, None


# ─── SIMULATE TRADE ──────────────────────────────────────────────────────────
def simulate(df, i, direction, atr, strategy, cfg, rh=None, rl=None):
    if i+1 >= len(df): return "EXPIRED", 0, 0, 0
    entry = df["open"].iloc[i+1]

    if strategy == "A":
        stop     = entry - cfg["atr_stop"]*atr if direction=="BUY" else entry + cfg["atr_stop"]*atr
        target   = entry + cfg["atr_target"]*atr if direction=="BUY" else entry - cfg["atr_target"]*atr
        max_hold = cfg["max_hold"]
    else:
        rm       = (rh + rl) / 2 if rh and rl else entry
        stop     = rl - atr*cfg["atr_range_stop"] if direction=="BUY" else rh + atr*cfg["atr_range_stop"]
        target   = rm
        max_hold = cfg["max_hold_range"]

    for j in range(i+1, min(i+max_hold+1, len(df))):
        hi = df["high"].iloc[j];  lo = df["low"].iloc[j]
        if direction == "BUY":
            if lo <= stop:   return "LOSS", j-i, entry, stop
            if hi >= target: return "WIN",  j-i, entry, target
        else:
            if hi >= stop:   return "LOSS", j-i, entry, stop
            if lo <= target: return "WIN",  j-i, entry, target

    return "EXPIRED", max_hold, entry, target


# ─── POSITION SIZING ─────────────────────────────────────────────────────────
def calc_lots(atr, price, cfg, direction, time_mod, comm_mod):
    stop_dist = cfg["atr_stop"] * atr
    stop_val  = (stop_dist * LOT_SIZE
                 if cfg["quote"] == "USD"
                 else stop_dist / price * LOT_SIZE)
    rate_mod  = get_rate_mod_i(cfg, direction)
    total_mod = cfg["base_size_mod"] * time_mod * comm_mod * rate_mod
    total_mod = max(total_mod, 0.0)
    risk      = RISK_PER_TRADE * total_mod
    lots      = round(risk / max(stop_val, 1), 2)
    return max(lots, 0.01), round(risk, 0), round(total_mod, 2)


# ─── P&L CALCULATION ─────────────────────────────────────────────────────────
def calc_pnl(entry, exit_price, direction, lots, price, cfg, outcome):
    if outcome == "EXPIRED":
        return -150 * lots  # small time-cost for expired trades

    if cfg["quote"] == "USD":
        raw = (exit_price - entry) * LOT_SIZE if direction=="BUY" else (entry - exit_price) * LOT_SIZE
    else:
        raw = ((exit_price - entry) / price * LOT_SIZE if direction=="BUY"
               else (entry - exit_price) / price * LOT_SIZE)
    return raw * lots


# ─── BACKTEST ────────────────────────────────────────────────────────────────
def run_backtest(all_data):
    results      = []
    equity       = STARTING_BALANCE
    peak         = STARTING_BALANCE
    max_dd       = 0.0
    equity_curve = [{"date": "start", "equity": equity}]
    last_sig     = {p: -MIN_SIGNAL_GAP for p in PAIR_CONFIGS}

    # Collect all candidate signals across all pairs
    candidates = []

    for pair, cfg in PAIR_CONFIGS.items():
        df       = all_data[pair]["daily"]
        h4_lk    = all_data[pair]["h4_lookup"]
        comm_lk  = all_data[pair]["comm_lookup"]
        start_i  = max(210, cfg["range_lookback"] + 5)

        print(f"  Scanning {cfg['name']} ({len(df)-start_i} candles)...")
        a_cnt = b_cnt = 0

        for i in range(start_i, len(df)-1):
            can_trade, time_mod = time_filter_i(df.index[i].date())
            if not can_trade: continue
            if i - last_sig[pair] < MIN_SIGNAL_GAP: continue

            regime, direction_hint = get_regime_i(df, i, cfg)
            strategy_type = cfg["strategy"]

            sig = direction = comm_mod = rh = rl = None

            if strategy_type == "trend" or (strategy_type == "auto" and regime == "TRENDING"):
                direction, comm_mod = strat_a_at(df, i, h4_lk, comm_lk, cfg)
                if direction:
                    sig = "A"
                    a_cnt += 1

            if sig is None and (strategy_type == "auto" and regime == "RANGING"):
                direction, comm_mod, rh, rl = strat_b_at(df, i, h4_lk, comm_lk, cfg)
                if direction:
                    sig = "B"
                    b_cnt += 1

            if sig and direction:
                candidates.append({
                    "ts": df.index[i], "pair": pair, "strategy": sig,
                    "direction": direction, "i": i, "time_mod": time_mod,
                    "comm_mod": comm_mod or 1.0, "rh": rh, "rl": rl,
                    "regime": regime,
                })

        print(f"    Strategy A: {a_cnt} signals | Strategy B: {b_cnt} signals")

    # Sort by date for portfolio rule enforcement
    candidates.sort(key=lambda x: x["ts"])
    print(f"\nTotal candidates: {len(candidates)}")

    open_pairs = {}   # pair -> candle index of open trade close
    open_buys  = 0
    open_sells = 0

    for c in candidates:
        pair      = c["pair"]
        direction = c["direction"]
        i         = c["i"]
        strategy  = c["strategy"]
        cfg       = PAIR_CONFIGS[pair]
        df        = all_data[pair]["daily"]

        # Skip if pair already in a trade
        if pair in open_pairs: continue

        # Portfolio rules
        if len(open_pairs) >= MAX_OPEN_TRADES: continue
        if direction == "BUY"  and open_buys  >= MAX_SAME_SIDE: continue
        if direction == "SELL" and open_sells >= MAX_SAME_SIDE: continue

        atr   = df["atr"].iloc[i]
        price = df["close"].iloc[i]

        lots, eff_risk, total_mod = calc_lots(
            atr, price, cfg, direction, c["time_mod"], c["comm_mod"]
        )

        outcome, hold, entry, exit_price = simulate(
            df, i, direction, atr, strategy, cfg, c["rh"], c["rl"]
        )

        pnl = calc_pnl(entry, exit_price, direction, lots, price, cfg, outcome)

        equity += pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

        last_sig[pair] = i
        if direction == "BUY":  open_buys += 1
        if direction == "SELL": open_sells += 1

        date_str = df.index[i].strftime("%Y-%m-%d")
        results.append({
            "date":       date_str,
            "pair":       pair,
            "strategy":   strategy,
            "direction":  direction,
            "regime":     c["regime"],
            "time_mod":   c["time_mod"],
            "comm_mod":   c["comm_mod"],
            "total_mod":  total_mod,
            "lots":       lots,
            "eff_risk":   eff_risk,
            "entry":      round(entry, 5),
            "exit":       round(exit_price, 5),
            "outcome":    outcome,
            "pnl":        round(pnl, 2),
            "equity":     round(equity, 2),
            "dd_pct":     round(dd, 2),
            "hold":       hold,
        })

        equity_curve.append({"date": date_str, "equity": round(equity, 2)})

        # Remove from open positions when trade closes
        # (simplified — we remove immediately since we simulate full hold)
        if direction == "BUY":  open_buys  = max(0, open_buys - 1)
        if direction == "SELL": open_sells = max(0, open_sells - 1)

    return pd.DataFrame(results), equity_curve, max_dd, peak


# ─── ANALYSE ─────────────────────────────────────────────────────────────────
def analyse(results, max_dd, peak):
    if results.empty:
        print("No signals fired.")
        return

    total    = len(results)
    wins     = (results["outcome"]=="WIN").sum()
    losses   = (results["outcome"]=="LOSS").sum()
    expired  = (results["outcome"]=="EXPIRED").sum()
    wr       = wins/total*100
    final    = results["equity"].iloc[-1]
    ret      = (final-STARTING_BALANCE)/STARTING_BALANCE*100
    years    = 2.5

    sep = "="*64
    print(f"\n{sep}")
    print("  MULTI-PAIR BACKTEST v3.0 — PAIR-SPECIFIC STRATEGY")
    print(f"  Period  : {results['date'].iloc[0]}  to  {results['date'].iloc[-1]}")
    print(f"  Account : ${STARTING_BALANCE:,.0f}  |  ${RISK_PER_TRADE:,.0f} base risk")
    print(sep)
    print(f"\n  Total    : {total} trades  (~{round(total/years):.0f}/yr)")
    print(f"  Win Rate : {wr:.1f}%  ({wins}W / {losses}L / {expired}E)")
    print(f"\n  Start    : ${STARTING_BALANCE:>12,.2f}")
    print(f"  Final    : ${final:>12,.2f}")
    print(f"  Return   : {ret:>+.1f}%")
    print(f"  Peak     : ${peak:>12,.2f}")
    print(f"  Max DD   : {max_dd:.1f}%")

    print(f"\n  ─── By Pair ──────────────────────────────────────────")
    for p, cfg in PAIR_CONFIGS.items():
        sub = results[results["pair"]==p]
        if not len(sub): continue
        wr2 = (sub["outcome"]=="WIN").sum()/len(sub)*100
        pnl = sub["pnl"].sum()
        a   = (sub["strategy"]=="A").sum()
        b   = (sub["strategy"]=="B").sum()
        print(f"  {cfg['name']:<10}: {len(sub):3d} trades (A:{a} B:{b}) | {wr2:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  ─── By Strategy ──────────────────────────────────────")
    for s, label in [("A","Trend Following"),("B","Mean Reversion")]:
        sub = results[results["strategy"]==s]
        if not len(sub): continue
        wr2 = (sub["outcome"]=="WIN").sum()/len(sub)*100
        pnl = sub["pnl"].sum()
        print(f"  {s} ({label:<16}): {len(sub):3d} trades | {wr2:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  ─── By Regime ────────────────────────────────────────")
    for reg in ["TRENDING","RANGING"]:
        sub = results[results["regime"]==reg]
        if not len(sub): continue
        wr2 = (sub["outcome"]=="WIN").sum()/len(sub)*100
        pnl = sub["pnl"].sum()
        print(f"  {reg:<10}: {len(sub):3d} trades | {wr2:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  ─── Commodity Filter Impact ──────────────────────────")
    for ok, label in [(1.0,"Aligned  "),(0.5,"Headwind "),(0.0,"Blocked  ")]:
        sub = results[results["comm_mod"]==ok]
        if not len(sub): continue
        wr2 = (sub["outcome"]=="WIN").sum()/len(sub)*100
        pnl = sub["pnl"].sum()
        print(f"  {label}: {len(sub):3d} trades | {wr2:5.1f}% win | P&L ${pnl:>+,.0f}")

    print(f"\n  ─── Pair-Specific Insights ───────────────────────────")
    for p, cfg in PAIR_CONFIGS.items():
        sub = results[results["pair"]==p]
        if len(sub) < 3: continue
        wins_p   = (sub["outcome"]=="WIN").sum()
        losses_p = (sub["outcome"]=="LOSS").sum()
        exp_p    = (sub["outcome"]=="EXPIRED").sum()
        wr_p     = wins_p/len(sub)*100
        avg_win  = sub[sub["outcome"]=="WIN"]["pnl"].mean() if wins_p else 0
        avg_loss = sub[sub["outcome"]=="LOSS"]["pnl"].mean() if losses_p else 0
        print(f"  {cfg['name']}: WR {wr_p:.0f}% | avg win ${avg_win:+,.0f} | avg loss ${avg_loss:+,.0f} | "
              f"exp cost ${sub[sub['outcome']=='EXPIRED']['pnl'].sum():+,.0f}")

    print(f"\n{sep}\n")


# ─── HTML EQUITY CURVE ───────────────────────────────────────────────────────
def generate_html(equity_curve, results, max_dd, peak):
    final  = results["equity"].iloc[-1]
    ret    = (final-STARTING_BALANCE)/STARTING_BALANCE*100
    total  = len(results)
    wins   = (results["outcome"]=="WIN").sum()
    losses = (results["outcome"]=="LOSS").sum()
    exp    = (results["outcome"]=="EXPIRED").sum()
    wr     = wins/total*100 if total else 0

    dates = [p["date"] for p in equity_curve]
    eqs   = [p["equity"] for p in equity_curve]

    dds = []
    pk  = eqs[0]
    for e in eqs:
        if e > pk: pk = e
        dds.append(-((pk-e)/pk*100))

    # Per-pair equity series
    pair_series = {}
    for p in PAIR_CONFIGS:
        sub = results[results["pair"]==p].sort_values("date")
        if len(sub):
            series = []
            running = STARTING_BALANCE
            for _, r in sub.iterrows():
                running += r["pnl"]
                series.append({"date": r["date"], "val": round(running-STARTING_BALANCE,0)})
            pair_series[p] = series

    # Build trade log rows (last 40)
    trows = ""
    for _, r in results.tail(40).iterrows():
        oc = "win" if r["outcome"]=="WIN" else "loss" if r["outcome"]=="LOSS" else "exp"
        pc = "win" if r["pnl"]>0 else "loss" if r["pnl"]<0 else ""
        pname = PAIR_CONFIGS[r["pair"]]["name"]
        trows += (f"<tr><td>{r['date']}</td><td>{pname}</td><td>{r['strategy']}</td>"
                  f"<td>{r['direction']}</td><td>{r['regime'][:3]}</td>"
                  f"<td>{r['lots']}</td>"
                  f"<td class='{oc}'>{r['outcome']}</td>"
                  f"<td class='{pc}'>${r['pnl']:+,.0f}</td>"
                  f"<td>${r['equity']:,.0f}</td></tr>\n")

    # Summary rows per pair
    prows = ""
    for p, cfg in PAIR_CONFIGS.items():
        sub = results[results["pair"]==p]
        if not len(sub): continue
        wr2  = (sub["outcome"]=="WIN").sum()/len(sub)*100
        pnl2 = sub["pnl"].sum()
        a    = (sub["strategy"]=="A").sum()
        b    = (sub["strategy"]=="B").sum()
        pc   = "win" if pnl2>=0 else "loss"
        prows += (f"<tr><td>{cfg['name']}</td><td>{len(sub)}</td>"
                  f"<td>{a}/{b}</td><td>{wr2:.1f}%</td>"
                  f"<td class='{pc}'>${pnl2:+,.0f}</td></tr>\n")

    # Pair colours
    PAIR_COLORS = {
        "GBP_USD": "#3fb950",
        "EUR_USD": "#58a6ff",
        "AUD_USD": "#d29922",
        "USD_JPY": "#ff7b72",
    }

    # Series data for pair chart
    pair_chart_datasets = []
    for p, cfg in PAIR_CONFIGS.items():
        if p not in pair_series: continue
        ser  = pair_series[p]
        dates_p = [s["date"] for s in ser]
        vals_p  = [s["val"] for s in ser]
        pair_chart_datasets.append({
            "label": cfg["name"],
            "dates": dates_p,
            "vals":  vals_p,
            "color": PAIR_COLORS.get(p, "#888"),
        })

    bal_c = "green" if final>=STARTING_BALANCE else "red"
    ret_c = "green" if ret>=0 else "red"
    dd_c  = "amber" if max_dd<20 else "red"
    gen   = datetime.now().strftime("%d %b %Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Multi-Pair Forex Backtest v3.0</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:24px}}
h1{{font-size:20px;font-weight:600;margin-bottom:4px}}
h2{{font-size:14px;font-weight:600;margin:20px 0 12px;color:#e6edf3}}
.sub{{color:#7d8590;font-size:13px;margin-bottom:24px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px}}
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
.pill{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
.pill-a{{background:rgba(63,185,80,0.15);color:#3fb950}}
.pill-b{{background:rgba(88,166,255,0.15);color:#58a6ff}}
.footer{{text-align:center;color:#7d8590;font-size:11px;margin-top:16px}}
</style>
</head>
<body>
<h1>Multi-Pair Forex Backtest — Pair-Specific v3.0</h1>
<p class="sub">
  {results['date'].iloc[0]} to {results['date'].iloc[-1]} &nbsp;|&nbsp;
  GBP/USD (Trend) &nbsp; EUR/USD (Auto) &nbsp; AUD/USD (Trend+Gold) &nbsp; USD/JPY (Trend+IV) &nbsp;|&nbsp;
  ${STARTING_BALANCE:,.0f} account &nbsp;|&nbsp; ${RISK_PER_TRADE:,.0f} base risk/trade
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

<div class="card"><div class="ct">Portfolio Equity Curve</div><canvas id="ec"></canvas></div>
<div class="card"><div class="ct">Per-Pair Cumulative P&amp;L</div><canvas id="pp"></canvas></div>
<div class="card"><div class="ct">Drawdown (%)</div><canvas id="dd"></canvas></div>

<div class="grid2">
  <div class="card"><div class="ct">By Pair</div>
    <table>
      <tr><th>Pair</th><th>Trades</th><th>A/B</th><th>Win%</th><th>P&amp;L</th></tr>
      {prows}
    </table>
  </div>
  <div class="card"><div class="ct">Strategy Config</div>
    <table>
      <tr><th>Pair</th><th>Strategy</th><th>ADX Trend</th><th>ADX Range</th><th>Commodity</th></tr>
      {''.join(f"<tr><td>{cfg['name']}</td><td>{cfg['strategy'].upper()}</td><td>>{cfg['adx_trend_min']}</td><td>&lt;{cfg['adx_range_max']}</td><td>{'REQ' if cfg['commodity_required'] else 'pref'} {cfg['commodity_label']}</td></tr>" for p,cfg in PAIR_CONFIGS.items())}
    </table>
  </div>
</div>

<div class="card"><div class="ct">Last 40 Trades</div>
  <table>
    <tr><th>Date</th><th>Pair</th><th>S</th><th>Dir</th><th>Reg</th><th>Lots</th><th>Result</th><th>P&amp;L</th><th>Balance</th></tr>
    {trows}
  </table>
</div>

<p class="footer">Multi-Pair Forex System v3.0 &nbsp;|&nbsp; Generated {gen}</p>

<script>
const dates={json.dumps(dates)};
const eq={json.dumps(eqs)};
const dds={json.dumps(dds)};

const mkChart=(id,ds,yfmt)=>new Chart(document.getElementById(id),{{
  type:'line',data:{{labels:dates,datasets:ds}},
  options:{{
    responsive:true,
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

mkChart('ec',[
  {{label:'Portfolio',data:eq,borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.05)',
    borderWidth:2,fill:true,tension:0.3,pointRadius:0}},
  {{label:'Start',data:Array(dates.length).fill({STARTING_BALANCE}),
    borderColor:'#333',borderWidth:1,borderDash:[5,4],pointRadius:0,fill:false}}
],v=>typeof v==='number'?'$'+(v/1000).toFixed(0)+'k':v);

// Per-pair P&L chart
const PDATA={json.dumps(pair_chart_datasets)};
const ppDs=PDATA.map(s=>{{
  // Map pair dates to full date axis
  const m={{}};
  s.dates.forEach((d,i)=>m[d]=s.vals[i]);
  let last=0;
  const data=dates.map(d=>{{if(m[d]!==undefined)last=m[d];return last;}});
  return {{label:s.label,data,borderColor:s.color,borderWidth:2,
    fill:false,tension:0.3,pointRadius:0}};
}});
mkChart('pp',ppDs,v=>typeof v==='number'?'$'+(v>=0?'+':'')+v.toLocaleString():v);

mkChart('dd',[
  {{label:'Drawdown',data:dds,borderColor:'#f85149',backgroundColor:'rgba(248,81,73,0.08)',
    borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}}
],v=>typeof v==='number'?v.toFixed(1)+'%':v);
</script>
</body></html>"""

    with open("equity_curve.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML saved to equity_curve.html")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Pair Backtest v3.0 — Pair-Specific Strategy")
    print(f"Account: ${STARTING_BALANCE:,.0f}  |  Risk: ${RISK_PER_TRADE:,.0f}/trade base\n")

    for p, cfg in PAIR_CONFIGS.items():
        print(f"  {cfg['name']:<10}: {cfg['strategy'].upper()} | "
              f"ADX trend>{cfg['adx_trend_min']} range<{cfg['adx_range_max']} | "
              f"{cfg['commodity_label']} {'REQUIRED' if cfg['commodity_required'] else 'preferred'} | "
              f"base ×{cfg['base_size_mod']}")
    print()
    print("Fetching data from OANDA...")

    all_data  = {}
    comm_data = {}

    for comm in set(cfg["commodity"] for cfg in PAIR_CONFIGS.values()):
        try:
            cdf = fetch_candles(comm, "D", 850)
            cdf = calculate_indicators(cdf)
            comm_data[comm] = build_comm_lookup(cdf)
            print(f"  {comm}: {len(cdf)} candles")
        except Exception as e:
            print(f"  {comm}: FAILED — {e}")
            comm_data[comm] = {}

    for pair, cfg in PAIR_CONFIGS.items():
        try:
            df    = fetch_candles(pair, "D",  850)
            df    = calculate_indicators(df)
            df_4h = fetch_candles(pair, "H4", 4250)
            df_4h = calculate_indicators(df_4h)
            all_data[pair] = {
                "daily":      df,
                "h4_lookup":  build_4h_lookup(df_4h),
                "comm_lookup": comm_data.get(cfg["commodity"], {}),
            }
            print(f"  {cfg['name']}: {len(df)} daily + {len(df_4h)} 4H")
        except Exception as e:
            print(f"  {cfg['name']}: FAILED — {e}")

    if not all_data:
        print("No data. Check API key.")
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
            print("Zero signals fired. Check diagnostic output above.")
