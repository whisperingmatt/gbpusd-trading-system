#!/usr/bin/env python3
"""Log a paper trade result to docs/trades.json."""

import json
import sys
from datetime import datetime
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "trades.json"
PAIRS       = ["GBP/USD", "EUR/USD", "AUD/USD", "USD/JPY"]


def load():
    with open(TRADES_FILE) as f:
        return json.load(f)


def save(data):
    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def ask(label, choices=None, default=None):
    while True:
        hint = ""
        if choices:
            hint = f" [{'/'.join(choices)}]"
        elif default is not None:
            hint = f" [{default}]"
        raw = input(f"  {label}{hint}: ").strip()
        if not raw and default is not None:
            return default
        if not raw:
            print("    (required)")
            continue
        if choices:
            match = [c for c in choices if c.upper() == raw.upper()]
            if not match:
                print(f"    Enter one of: {', '.join(choices)}")
                continue
            return match[0]
        return raw


def ask_float(label, allow_negative=False):
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            v = float(raw)
            if not allow_negative and v < 0:
                print("    Enter a positive number (use negative P&L for losses)")
                continue
            return v
        except ValueError:
            print("    Enter a valid number (e.g. 1.27345 or -850.00)")


def main():
    if not TRADES_FILE.exists():
        print(f"Error: {TRADES_FILE} not found.")
        sys.exit(1)

    data   = load()
    trades = data["trades"]
    start  = data["starting_balance"]
    current = trades[-1]["balance"] if trades else start
    closed = [t for t in trades if t.get("outcome") not in (None, "OPEN")]
    wins   = [t for t in closed if t.get("outcome") == "WIN"]
    open_  = [t for t in trades if t.get("outcome") == "OPEN"]

    print()
    print("─" * 52)
    print("  Log Paper Trade")
    print("─" * 52)
    print(f"  Starting balance  : ${start:>12,.2f}  ({data.get('start_date','')})")
    print(f"  Current balance   : ${current:>12,.2f}")
    ret = (current - start) / start * 100
    print(f"  Total return      : {ret:+.2f}%")
    print(f"  Trades logged     : {len(trades)}  ({len(closed)} closed · {len(open_)} open)")
    if closed:
        print(f"  Win rate          : {len(wins)/len(closed)*100:.0f}%  ({len(wins)}/{len(closed)})")
    print("─" * 52)
    print()

    today = datetime.now().strftime("%Y-%m-%d")
    date  = ask("Date (YYYY-MM-DD)", default=today)
    pair  = ask("Pair", choices=PAIRS)
    direction = ask("Direction", choices=["BUY", "SELL"])
    strategy  = ask("Strategy", choices=["A", "B"])
    entry  = ask_float("Entry price")
    stop   = ask_float("Stop price")
    target = ask_float("Target price")
    outcome = ask("Outcome", choices=["WIN", "LOSS", "BREAKEVEN", "OPEN"])

    pnl = 0.0
    if outcome != "OPEN":
        pnl = ask_float("P&L in USD (negative for loss)", allow_negative=True)

    notes = ask("Notes", default="")

    new_balance = round(current + pnl, 2)
    strat_name  = "A — Trend Following" if strategy == "A" else "B — Mean Reversion"

    trade = {
        "id":        len(trades) + 1,
        "date":      date,
        "pair":      pair,
        "direction": direction,
        "strategy":  strat_name,
        "entry":     entry,
        "stop":      stop,
        "target":    target,
        "outcome":   outcome,
        "pnl":       pnl,
        "balance":   new_balance,
        "notes":     notes,
    }

    print()
    print("─" * 52)
    print("  Trade to log:")
    for k, v in trade.items():
        if v or v == 0:
            print(f"    {k:<12} : {v}")
    print("─" * 52)

    confirm = ask("Save? ", choices=["Y", "N"])
    if confirm.upper() != "Y":
        print("  Cancelled.")
        return

    trades.append(trade)
    data["trades"] = trades
    save(data)

    closed2 = [t for t in trades if t.get("outcome") not in (None, "OPEN")]
    wins2   = [t for t in closed2 if t.get("outcome") == "WIN"]
    print()
    print("  Trade logged.")
    print(f"    New balance  : ${new_balance:>12,.2f}")
    print(f"    Total return : {(new_balance - start) / start * 100:+.2f}%")
    if closed2:
        print(f"    Win rate     : {len(wins2)/len(closed2)*100:.0f}%  ({len(wins2)}/{len(closed2)})")
    print()
    print("  Commit & push to update the dashboard:")
    print('    git add docs/trades.json && git commit -m "Add trade" && git push')
    print()


if __name__ == "__main__":
    main()
