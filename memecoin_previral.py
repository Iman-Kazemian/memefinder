#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memecoin_previral.py
- Finds pre-viral memes, alerts to Telegram.
- Persists NEW finds and opens $10 virtual positions for backtesting.

Files created:
  data/discoveries.csv      (append-only, one row per new discovery)
  data/positions_open.csv   (current open positions)
  data/trades.csv           (master ledger: open and closed trades)
  runs/last_scan.csv        (all candidates passing gates this run)
"""

import os, re, json, time, math, argparse, pathlib
from typing import Dict, List, Optional
import requests, pandas as pd

# --------- IO & utils --------- #
ROOT = pathlib.Path(".")
DATA = ROOT / "data"
RUNS = ROOT / "runs"
DATA.mkdir(exist_ok=True, parents=True)
RUNS.mkdir(exist_ok=True, parents=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "memecoin-previral/1.2"})

def now_ts() -> int: return int(time.time())
def now_iso() -> str: return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def http_get(url: str, params=None, headers=None, timeout: int = 20):
    try:
        r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[http] GET fail: {url} -> {e}")
        return None

def to_float(x, default=None):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default

def pct_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    try: return f"{float(x):+0.1f}%"
    except Exception: return str(x)

# --------- seeds --------- #
def fetch_coingecko_trending_public() -> List[str]:
    print("[*] Seeds: CoinGecko trending…")
    js = http_get("https://api.coingecko.com/api/v3/search/trending")
    out = []
    if js and isinstance(js.get("coins"), list):
        for c in js["coins"]:
            item = c.get("item") or {}
            sym = item.get("symbol")
            if sym: out.append(sym.upper())
    # de-dupe
    return list(dict.fromkeys(out))

def fetch_birdeye_solana_trending(birdeye_key: str, limit: int = 50) -> List[str]:
    print("[*] Seeds: Birdeye Solana trending (optional)…")
    if not birdeye_key: return []
    url = "https://public-api.birdeye.so/defi/v3/token/trending"
    headers = {"x-api-key": birdeye_key}
    params = {"chain": "solana", "limit": limit}
    js = http_get(url, params=params, headers=headers)
    out = []
    if js and isinstance(js.get("data"), dict):
        for t in js["data"].get("tokens", []):
            sym = (t.get("symbol") or "").upper()
            if sym: out.append(sym)
    return list(dict.fromkeys(out))

# --------- Dexscreener --------- #
def dexscreener_search_pairs_by_symbol(symbol: str, limit_per_symbol: int = 50) -> List[dict]:
    js = http_get("https://api.dexscreener.com/latest/dex/search", params={"q": symbol})
    if not js: return []
    pairs = js.get("pairs") or []
    return pairs[:limit_per_symbol]

def fetch_dexscreener_boosts_map() -> Dict[str, int]:
    print("[*] Fetching boosts map…")
    js = http_get("https://api.dexscreener.com/latest/dex/boosts")
    m = {}
    if js and isinstance(js.get("boosts"), list):
        for b in js["boosts"]:
            pa = b.get("pairAddress")
            cnt = b.get("boosts") or b.get("boostCount") or b.get("count")
            if pa and isinstance(cnt, (int, float)): m[pa] = int(cnt)
    return m

def normalize_pair(p: dict) -> Optional[dict]:
    try:
        chain = (p.get("chainId") or p.get("chain") or "").lower()
        dex   = (p.get("dexId") or p.get("dex") or "").lower()
        base  = (p.get("baseToken", {}) or {}).get("symbol") or p.get("baseSymbol")
        quote = (p.get("quoteToken", {}) or {}).get("symbol") or p.get("quoteSymbol")
        base_addr = (p.get("baseToken", {}) or {}).get("address")
        pair_addr = p.get("pairAddress") or p.get("pairId") or p.get("address")

        liq_usd = to_float((p.get("liquidity", {}) or {}).get("usd"))
        vol24   = to_float((p.get("volume", {}) or {}).get("h24"))
        # price
        price_usd = to_float(p.get("priceUsd")) or to_float(p.get("price"))

        tx5 = (p.get("txns", {}) or {}).get("m5") or {}
        tx1 = (p.get("txns", {}) or {}).get("h1") or {}
        txns5m = to_float(tx5.get("buys"), 0) + to_float(tx5.get("sells"), 0)
        txns1h = to_float(tx1.get("buys"), 0) + to_float(tx1.get("sells"), 0)
        buys5m = to_float(tx5.get("buys"), 0)
        sells5m = to_float(tx5.get("sells"), 0)
        buy_ratio_5m = (buys5m / max(1.0, buys5m + sells5m)) if (buys5m + sells5m) > 0 else None

        chg5m = to_float((p.get("priceChange", {}) or {}).get("m5"))
        chg1h = to_float((p.get("priceChange", {}) or {}).get("h1"))
        chg24 = to_float((p.get("priceChange", {}) or {}).get("h24"))

        age_hours = None
        created_ms = p.get("pairCreatedAt")
        if created_ms:
            try: age_hours = (now_ts()*1000 - int(created_ms)) / 1000 / 3600.0
            except Exception: age_hours = None

        return {
            "chain": chain, "dex": dex,
            "base_symbol": (base or "").upper(), "quote_symbol": (quote or "").upper(),
            "base_address": base_addr, "pair_address": pair_addr,
            "price_usd": price_usd, "liq_usd": liq_usd, "vol24_usd": vol24,
            "txns5m": txns5m, "txns1h": txns1h, "buy_ratio_5m": buy_ratio_5m,
            "chg5m": chg5m, "chg1h": chg1h, "chg24h": chg24, "age_hours": age_hours
        }
    except Exception:
        return None

# --------- score & gates --------- #
def compute_viral_score(df: pd.DataFrame, boosts_map: Dict[str, int]) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df["boosts"] = df["pair_address"].map(lambda x: boosts_map.get(str(x), 0))
    df["accel"] = (df["txns1h"].fillna(0.0) / (df["txns5m"].fillna(0.0).clip(lower=1.0))).clip(upper=12.0)
    df["turnover_ratio"] = (df["vol24_usd"].fillna(0.0) / df["liq_usd"].fillna(1.0)).replace([math.inf, -math.inf], 0)

    br = df["buy_ratio_5m"].fillna(0.5).clip(0, 1)
    accel = df["accel"].fillna(1.0).clip(0, 12)
    tovr = df["turnover_ratio"].fillna(0.0).clip(0, 10)
    boosts = df["boosts"].fillna(0.0).clip(0, 100)
    chg5 = df["chg5m"].fillna(0.0).clip(-50, 50) / 50.0
    chg1 = df["chg1h"].fillna(0.0).clip(-50, 50) / 50.0
    chg24 = df["chg24h"].fillna(0.0).clip(-200, 200) / 200.0

    df["ViralScore"] = (
        0.35 * br
      + 0.25 * (accel / 4.0)
      + 0.20 * (tovr / 2.0)
      + 0.10 * (boosts / 10.0)
      + 0.05 * chg5
      + 0.03 * chg1
      + 0.02 * chg24
    ).clip(0, 1.0)
    return df

MEME_PATTERN_DEFAULT = r"(PEPE|DOGE|SHIB|BONK|WIF|FLOKI|INU|MOON|PUMP|MEME|CAT|DOG|HAMSTER|PANDA|KEK|SOON|GIGA|PENG|LUNA|BOZO|DEGEN|FART|BABY|ELON|TRUMP|BIDEN|HARRY|HUSKY|FROG|FROGE|KIRK|MOG|WOJAK|RUG|PONZI|GOAT|TURTLE|NINJA|BASED)$"

def is_memeish(symbol: str, name: Optional[str] = None, pattern: Optional[str] = None) -> bool:
    s = (symbol or "").upper(); nm = (name or "").upper()
    pat = re.compile(pattern or MEME_PATTERN_DEFAULT, re.IGNORECASE)
    if pat.search(s) or pat.search(nm): return True
    obvious = ["PEPE","DOGE","SHIB","BONK","WIF","INU","MEME","MOON","PUMP","BASED","PENG","FROG","MOG"]
    return any(x in s for x in obvious) or any(x in nm for x in obvious)

def audit(df: pd.DataFrame, tag: str, enabled: bool):
    if enabled: print(f"[audit] {tag}: {len(df)} rows")

def apply_quality_gates(df: pd.DataFrame, *, chains, only_quotes, exclude_dex,
                        meme_only, meme_regex, min_liq, max_liq, min_vol24, abs_vol24_floor,
                        min_turnover, age_min, age_max, min_accel, min_buys5m,
                        min_txns1h, min_txns5m, debug: bool) -> pd.DataFrame:
    x = df.copy()
    audit(x, "start", debug)

    if chains:
        chains_lc = [c.strip().lower() for c in chains if c.strip()]
        x = x[x["chain"].isin(chains_lc)]
        audit(x, "after chains", debug)

    if exclude_dex:
        ex = set(d.strip().lower() for d in exclude_dex if d.strip())
        if ex: x = x[~x["dex"].isin(ex)]
        audit(x, "after exclude_dex", debug)

    if only_quotes:
        oq = set(q.upper().strip() for q in only_quotes if q.strip())
        if oq: x = x[x["quote_symbol"].isin(oq)]
        audit(x, "after only_quotes", debug)

    if meme_only:
        x = x[x["base_symbol"].apply(lambda s: is_memeish(s, None, meme_regex))]
        audit(x, "after meme-only", debug)

    # fill nums
    for c in ["liq_usd","vol24_usd","txns5m","txns1h","buy_ratio_5m","accel"]:
        x[c] = x[c].fillna(0.0)
    x["age_hours"] = x["age_hours"].fillna(float("nan"))

    if min_liq is not None: x = x[x["liq_usd"] >= float(min_liq)]
    if max_liq is not None and max_liq > 0: x = x[x["liq_usd"] <= float(max_liq)]
    audit(x, "after liq bounds", debug)

    if abs_vol24_floor is not None: x = x[x["vol24_usd"] >= float(abs_vol24_floor)]
    if min_vol24 is not None: x = x[x["vol24_usd"] >= float(min_vol24)]
    audit(x, "after vol floors", debug)

    x["turnover_ratio"] = (x["vol24_usd"] / x["liq_usd"].replace(0, float("inf"))).replace([math.inf,-math.inf],0)
    if min_turnover is not None: x = x[x["turnover_ratio"] >= float(min_turnover)]
    audit(x, "after turnover", debug)

    if age_min is not None: x = x[(x["age_hours"].isna()) | (x["age_hours"] >= float(age_min))]
    if age_max is not None: x = x[(x["age_hours"].isna()) | (x["age_hours"] <= float(age_max))]
    audit(x, "after age", debug)

    if min_accel is not None: x = x[x["accel"] >= float(min_accel)]
    if min_buys5m is not None: x = x[x["buy_ratio_5m"] >= float(min_buys5m)]
    if min_txns1h is not None: x = x[x["txns1h"] >= float(min_txns1h)]
    if min_txns5m is not None: x = x[x["txns5m"] >= float(min_txns5m)]
    audit(x, "after momentum/txns", debug)
    return x

# --------- Telegram --------- #
def send_telegram(msg: str, token: str, chat_id: str):
    if not token or not chat_id: return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}
        SESSION.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"[telegram] failed: {e}")

def dexscreener_link(chain: str, pair: str) -> str:
    return f"https://dexscreener.com/{(chain or '').lower()}/{pair}"

def build_group_alert(df: pd.DataFrame, top: int = 10) -> str:
    rows = []
    for _, r in df.head(top).iterrows():
        rows.append(
            "• *{sym}* [{chain}/{dex}] — VS {vs:.2f} | Accel×{acc:.2f} | Buys5m {br:.2f}\n"
            "  Liq ${liq:,.0f} | Vol24 ${vol:,.0f} | Δ5m {c5} | Δ1h {c1} | Δ24h {c24}\n"
            "  Age {age}h | [Dexscreener]({link})".format(
                sym=r["base_symbol"], chain=r["chain"], dex=r["dex"], vs=r["ViralScore"],
                acc=r["accel"], br=r["buy_ratio_5m"], liq=r["liq_usd"], vol=r["vol24_usd"],
                c5=pct_str(r["chg5m"]), c1=pct_str(r["chg1h"]), c24=pct_str(r["chg24h"]),
                age=int(r["age_hours"]) if pd.notnull(r["age_hours"]) else -1,
                link=dexscreener_link(r["chain"], r["pair_address"])
            )
        )
    return "🔥 *Pre-Viral Watchlist*\n" + "\n".join(rows)

# cooldown cache (to avoid re-alert spam)
CACHE_FILE = DATA / "alert_cache.json"

def load_cache():
    if CACHE_FILE.exists():
        try: return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def save_cache(cache):
    try: CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception: pass

# --------- persistence helpers --------- #
DISC = DATA / "discoveries.csv"
OPEN = DATA / "positions_open.csv"
TRADES = DATA / "trades.csv"

def read_csv_safe(path: pathlib.Path) -> pd.DataFrame:
    if path.exists():
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame()

def append_csv(path: pathlib.Path, rowdicts: List[dict]):
    df = pd.DataFrame(rowdicts)
    if path.exists() and path.stat().st_size > 0:
        df0 = pd.read_csv(path)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(path, index=False)

# --------- main flow --------- #
def seed_symbols(args) -> List[str]:
    seeds = fetch_coingecko_trending_public()
    if args.birdeye_key: seeds += fetch_birdeye_solana_trending(args.birdeye_key, limit=min(100, args.seed_boosts_limit))
    seeds = [s for s in seeds if s]
    return list(dict.fromkeys(seeds))

def expand_pairs_from_seeds(seeds: List[str], per_symbol: int) -> List[dict]:
    print("[*] Expanding via Dexscreener search + Boosts…")
    out = []
    for s in seeds:
        ps = dexscreener_search_pairs_by_symbol(s, limit_per_symbol=per_symbol)
        if ps: out.extend(ps)
        time.sleep(0.05)
    return out

def run_once(args):
    seeds = seed_symbols(args)
    if not seeds:
        print("[!] No seeds gathered; aborting."); return

    raw = expand_pairs_from_seeds(seeds, per_symbol=max(20, min(100, args.seed_boosts_limit)))
    df = pd.DataFrame([normalize_pair(p) for p in raw if normalize_pair(p) is not None])

    boosts_map = fetch_dexscreener_boosts_map()
    df = compute_viral_score(df, boosts_map)

    # Apply gates
    chains = [c.strip() for c in (args.chains or "").split(",") if c.strip()]
    only_quotes = [q.strip() for q in (args.only_quotes or "").split(",") if q.strip()]
    exclude_dex = [d.strip() for d in (args.exclude_dex or "").split(",") if d.strip()]
    df = apply_quality_gates(
        df=df, chains=chains, only_quotes=only_quotes, exclude_dex=exclude_dex,
        meme_only=args.meme_only, meme_regex=args.meme_regex or None,
        min_liq=args.min_liquidity, max_liq=args.max_liquidity,
        min_vol24=args.min_vol24, abs_vol24_floor=args.abs_vol24_floor,
        min_turnover=args.min_turnover_ratio, age_min=args.age_min, age_max=args.age_max,
        min_accel=args.min_accel, min_buys5m=args.min_buys5m,
        min_txns1h=args.min_txns1h, min_txns5m=args.min_txns5m, debug=args.debug
    )

    if df is None or df.empty:
        if args.debug: print("[debug] after gates: 0 rows")
        print("[!] No candidates.\n\nTop picks (why):")
        return

    if "ViralScore" in df.columns:
        df = df.sort_values("ViralScore", ascending=False).copy()

    # Save run snapshot
    df.to_csv(RUNS / "last_scan.csv", index=False)

    # Pretty print small table
    show_cols = ["ViralScore","chain","dex","base_symbol","quote_symbol","liq_usd","vol24_usd",
                 "txns5m","txns1h","buy_ratio_5m","accel","chg5m","chg1h","chg24h","age_hours","pair_address"]
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        print(df[show_cols].head(args.top).to_string(index=False))

    # ---------- Telegram + NEW discoveries persistence ---------- #
    token = args.telegram_token
    chat  = args.telegram_chat

    alerts = df[
        (df["ViralScore"] >= args.alert_threshold)
        & (df["accel"] >= args.alert_accel)
        & (df["buy_ratio_5m"] >= args.alert_buys)
    ].copy()

    # cooldown de-dupe (by chain|pair)
    if not alerts.empty and (token and chat):
        alerts["__id"] = alerts[["chain","pair_address"]].astype(str).agg("|".join, axis=1)
        cache = load_cache()
        cutoff = now_ts() - int(args.cooldown_hours*3600)
        fresh = alerts[alerts["__id"].map(lambda k: cache.get(k, 0) < cutoff)].copy()

        if not fresh.empty:
            # 1) TELEGRAM
            msg = build_group_alert(fresh, top=min(len(fresh), args.top))
            send_telegram(msg, token, chat)
            print(f"[telegram] sent grouped alert for {len(fresh)} coins")

            # 2) DISCOVERIES LOG
            discovery_rows = []
            for _, r in fresh.iterrows():
                discovery_rows.append({
                    "ts_utc": now_iso(),
                    "chain": r["chain"], "dex": r["dex"],
                    "symbol": r["base_symbol"], "quote": r["quote_symbol"],
                    "pair_address": r["pair_address"],
                    "price_usd": r.get("price_usd", None),
                    "liq_usd": r.get("liq_usd", None),
                    "vol24_usd": r.get("vol24_usd", None),
                    "ViralScore": r.get("ViralScore", None)
                })
            append_csv(DISC, discovery_rows)

            # 3) OPEN $10 POSITIONS (if not already open)
            open_df = read_csv_safe(OPEN)
            open_ids = set()
            if not open_df.empty:
                open_df["__id"] = open_df[["chain","pair_address"]].astype(str).agg("|".join, axis=1)
                open_ids = set(open_df["__id"].tolist())

            trade_rows = []
            new_pos_rows = []
            for _, r in fresh.iterrows():
                _id = f"{r['chain']}|{r['pair_address']}"
                if _id in open_ids:  # already open
                    continue
                entry_price = to_float(r.get("price_usd"), None)
                if not entry_price or entry_price <= 0:
                    # If price missing, skip opening
                    continue
                usd_alloc = 10.0
                qty = usd_alloc / entry_price
                trade_id = f"{int(time.time())}-{r['chain']}-{r['pair_address']}"

                new_pos_rows.append({
                    "trade_id": trade_id,
                    "opened_utc": now_iso(),
                    "chain": r["chain"], "dex": r["dex"],
                    "symbol": r["base_symbol"], "quote": r["quote_symbol"],
                    "pair_address": r["pair_address"],
                    "entry_price": entry_price,
                    "qty": qty,
                    "usd_alloc": usd_alloc,
                    "tp_mult": 2.0,
                    "sl_mult": 0.6,
                    "max_hold_hours": 12.0,
                    "status": "open"
                })

                trade_rows.append({
                    "trade_id": trade_id, "status": "open",
                    "opened_utc": now_iso(), "closed_utc": "",
                    "chain": r["chain"], "dex": r["dex"],
                    "symbol": r["base_symbol"], "quote": r["quote_symbol"],
                    "pair_address": r["pair_address"],
                    "entry_price": entry_price, "exit_price": "",
                    "qty": qty, "usd_alloc": usd_alloc, "pnl_usd": ""
                })

            if new_pos_rows:
                # update open and master trades
                if open_df.empty:
                    pd.DataFrame(new_pos_rows).to_csv(OPEN, index=False)
                else:
                    pd.concat([open_df, pd.DataFrame(new_pos_rows)], ignore_index=True).to_csv(OPEN, index=False)

                trades_df = read_csv_safe(TRADES)
                if trades_df.empty:
                    pd.DataFrame(trade_rows).to_csv(TRADES, index=False)
                else:
                    pd.concat([trades_df, pd.DataFrame(trade_rows)], ignore_index=True).to_csv(TRADES, index=False)

            # 4) cooldown update
            tnow = now_ts()
            for k in fresh["__id"].tolist():
                cache[k] = tnow
            save_cache(cache)

# --------- CLI --------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Find pre-viral meme coins; alert & persist new finds.")
    ap.add_argument("--chains", type=str, default="solana,base,bsc")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed-boosts-limit", type=int, default=200)

    # filters
    ap.add_argument("--meme-only", action="store_true")
    ap.add_argument("--meme-regex", type=str, default="")
    ap.add_argument("--only-quotes", type=str, default="USDC,SOL,WETH,USDT,USD1")
    ap.add_argument("--exclude-dex", type=str, default="")
    ap.add_argument("--min-liquidity", type=float, default=30000)
    ap.add_argument("--max-liquidity", type=float, default=300000)
    ap.add_argument("--min-vol24", type=float, default=30000)
    ap.add_argument("--abs-vol24-floor", type=float, default=30000)
    ap.add_argument("--min-turnover-ratio", type=float, default=0.003)
    ap.add_argument("--age-min", type=float, default=2)
    ap.add_argument("--age-max", type=float, default=120)
    ap.add_argument("--min-accel", type=float, default=1.10)
    ap.add_argument("--min-buys5m", type=float, default=0.60)
    ap.add_argument("--min-txns1h", type=int, default=20)
    ap.add_argument("--min-txns5m", type=int, default=2)

    # telegram
    ap.add_argument("--telegram-token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN",""))
    ap.add_argument("--telegram-chat",  type=str, default=os.getenv("TELEGRAM_CHAT_ID",""))
    ap.add_argument("--alert-threshold", type=float, default=0.45)
    ap.add_argument("--alert-accel", type=float, default=1.2)
    ap.add_argument("--alert-buys", type=float, default=0.6)
    ap.add_argument("--cooldown-hours", type=float, default=6.0)

    # optional birdeye
    ap.add_argument("--birdeye-key", type=str, default=os.getenv("BIRDEYE_API_KEY",""))

    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    run_once(args)

if __name__ == "__main__":
    main()
