#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py
- Marks open positions to market and closes by rule.
- Sends Telegram summary per run.

Inputs:
  data/positions_open.csv  (from memecoin_previral.py)
  data/trades.csv          (master ledger)

Outputs:
  data/positions_open.csv  (updated)
  data/trades.csv          (updated)
"""

import os, time, math, pathlib
from typing import List, Dict
import requests, pandas as pd

ROOT = pathlib.Path(".")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)
OPEN = DATA / "positions_open.csv"
TRADES = DATA / "trades.csv"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "memecoin-backtest/1.0"})

# Strategy params (can be env-overridden)
TP_MULT = float(os.getenv("TP_MULT", "2.0"))   # take-profit at 2x
SL_MULT = float(os.getenv("SL_MULT", "0.6"))   # stop-loss at -40%
MAX_HOLD_HRS = float(os.getenv("MAX_HOLD_HRS", "12.0"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def http_get(url: str, timeout=18):
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[http] {url} -> {e}")
        return None

def fetch_pair_price(chain: str, pair: str) -> float:
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"
    js = http_get(url)
    if not js: return None
    pairs = js.get("pairs") or []
    if not pairs: return None
    p = pairs[0]
    return float(p.get("priceUsd") or p.get("price") or 0) or None

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        SESSION.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=15)
    except Exception as e:
        print(f"[telegram] {e}")

def read_csv_safe(path: pathlib.Path) -> pd.DataFrame:
    if path.exists():
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame()

def main():
    open_df = read_csv_safe(OPEN)
    if open_df.empty:
        print("[backtest] No open positions.")
        send_telegram("📉 Backtest: No open positions.")
        return

    # ensure needed cols
    for col in ["trade_id","opened_utc","chain","pair_address","entry_price","qty","usd_alloc","tp_mult","sl_mult","max_hold_hours","status"]:
        if col not in open_df.columns:
            print(f"[backtest] Missing column in positions_open.csv: {col}")
            return

    closed_rows: List[Dict] = []
    remain_rows: List[Dict] = []
    realized_pnl = 0.0

    # iterate open positions
    for _, r in open_df.iterrows():
        chain = str(r["chain"]).lower()
        pair  = str(r["pair_address"])
        entry = float(r["entry_price"])
        qty   = float(r["qty"])
        tp    = float(r.get("tp_mult", TP_MULT))
        sl    = float(r.get("sl_mult", SL_MULT))
        maxh  = float(r.get("max_hold_hours", MAX_HOLD_HRS))

        # compute holding time
        # opened_utc is not parsed here; we just skip exact hours calc if missing
        held_hours = None
        try:
            # crude parse: treat opened_utc as epoch if convertible, else skip
            # (optional: you can store opened_epoch in scanner to make this exact)
            held_hours = float(r.get("held_hours", "nan"))
        except Exception:
            held_hours = None

        price = fetch_pair_price(chain, pair)
        if not price or price <= 0:
            # keep open if we cannot price it now
            remain_rows.append(dict(r))
            continue

        up_mult = price / entry
        should_close = False
        reason = ""
        if up_mult >= tp:
            should_close = True; reason = f"TP hit ({up_mult:.2f}x)"
        elif up_mult <= sl:
            should_close = True; reason = f"SL hit ({up_mult:.2f}x)"
        elif held_hours is not None and held_hours >= maxh:
            should_close = True; reason = f"MaxHold {held_hours:.1f}h"

        if should_close:
            exit_value = price * qty
            pnl = exit_value - float(r["usd_alloc"])
            realized_pnl += pnl
            closed_rows.append({
                "trade_id": r["trade_id"], "status": "closed",
                "opened_utc": r["opened_utc"], "closed_utc": now_iso(),
                "chain": r["chain"], "dex": r.get("dex",""),
                "symbol": r.get("symbol",""), "quote": r.get("quote",""),
                "pair_address": pair, "entry_price": entry, "exit_price": price,
                "qty": qty, "usd_alloc": r["usd_alloc"], "pnl_usd": pnl, "reason": reason
            })
        else:
            # keep position open; update optional held_hours if desired
            rr = dict(r)
            remain_rows.append(rr)

    # write back
    pd.DataFrame(remain_rows).to_csv(OPEN, index=False)
    if closed_rows:
        trades_df = read_csv_safe(TRADES)
        if trades_df.empty:
            pd.DataFrame(closed_rows).to_csv(TRADES, index=False)
        else:
            pd.concat([trades_df, pd.DataFrame(closed_rows)], ignore_index=True).to_csv(TRADES, index=False)

    # Telegram summary
    if closed_rows:
        winners = sum(1 for z in closed_rows if z["pnl_usd"] > 0)
        losers  = len(closed_rows) - winners
        total = len(closed_rows)
        msg = [
            "✅ *Backtest Run Summary*",
            f"Closed trades: *{total}*  (W:{winners} / L:{losers})",
            f"Realized PnL: *${realized_pnl:,.2f}*",
        ]
        # show a couple lines
        for z in closed_rows[:5]:
            msg.append(f"- {z['symbol']} [{z['chain']}] {z['reason']}: PnL ${z['pnl_usd']:.2f}")
        send_telegram("\n".join(msg))
        print("\n".join(msg))
    else:
        send_telegram("📉 Backtest: No exits this run.")
        print("[backtest] No exits this run.")

if __name__ == "__main__":
    main()
