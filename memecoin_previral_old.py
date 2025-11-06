#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memecoin_previral.py
Scan DEX pairs for early, liquid, momentum-leaning meme coins and push alerts to Telegram.

Data sources (no keys required):
- CoinGecko trending (public endpoint)
- Dexscreener search + pairs + boosts (public endpoints)

Optional:
- Birdeye Solana trending (requires BIRDEYE_API_KEY env or --birdeye-key)

CLI examples:
  python memecoin_previral.py --chains solana,base,bsc --top 20 \
    --meme-only --age-min 2 --age-max 120 \
    --min-liquidity 30000 --max-liquidity 300000 \
    --min-vol24 30000 --min-turnover-ratio 0.003 --abs-vol24-floor 30000 \
    --min-accel 1.15 --min-buys5m 0.60 --min-txns1h 20 --min-txns5m 2 \
    --only-quotes USDC,SOL,WETH,USDT \
    --telegram-token <BOT_TOKEN> --telegram-chat <CHAT_ID> --debug
"""

import os
import re
import json
import time
import math
import argparse
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd

# ------------------------------- utils -------------------------------- #

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "memecoin-previral/1.0"})

def http_get(url: str, params: dict = None, headers: dict = None, timeout: int = 20):
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
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    try:
        return f"{float(x):+0.1f}%"
    except Exception:
        return str(x)

def now_ts():
    return int(time.time())

# -------------------------- sources: seeds ----------------------------- #

def fetch_coingecko_trending_public() -> List[str]:
    """
    Public CG trending: returns a list of token symbols to seed searches.
    """
    print("[*] Seeds: CoinGecko trending…")
    url = "https://api.coingecko.com/api/v3/search/trending"
    js = http_get(url)
    out = []
    if js and isinstance(js.get("coins"), list):
        for c in js["coins"]:
            item = c.get("item") or {}
            sym = item.get("symbol")
            if sym:
                out.append(sym.upper())
    return list(dict.fromkeys(out))  # de-dupe, preserve order

def fetch_birdeye_solana_trending(birdeye_key: str, limit: int = 50) -> List[str]:
    """
    Optional seeds from Birdeye Solana trending (requires API key).
    """
    print("[*] Seeds: Birdeye Solana trending (optional)…")
    if not birdeye_key:
        return []
    url = "https://public-api.birdeye.so/defi/v3/token/trending"
    headers = {"x-api-key": birdeye_key}
    params = {"chain": "solana", "limit": limit}
    js = http_get(url, params=params, headers=headers)
    out = []
    if js and isinstance(js.get("data"), dict):
        for t in js["data"].get("tokens", []):
            sym = (t.get("symbol") or "").upper()
            if sym:
                out.append(sym)
    return list(dict.fromkeys(out))

# ------------------------ dex expansion & boosts ----------------------- #

def dexscreener_search_pairs_by_symbol(symbol: str, limit_per_symbol: int = 50) -> List[dict]:
    """
    Use Dexscreener search to pull pairs for a ticker text. We keep the JSON for parsing later.
    """
    url = "https://api.dexscreener.com/latest/dex/search"
    js = http_get(url, params={"q": symbol})
    if not js:
        return []
    pairs = js.get("pairs") or []
    # Keep only first `limit_per_symbol` to avoid explosion
    return pairs[:limit_per_symbol]

def fetch_dexscreener_boosts_map() -> Dict[str, int]:
    """
    Fetch boosts info from Dexscreener and return a map pairAddress -> boostsCount
    """
    print("[*] Fetching boosts map…")
    url = "https://api.dexscreener.com/latest/dex/boosts"
    js = http_get(url)
    boosts_map = {}
    if js and isinstance(js.get("boosts"), list):
        for b in js["boosts"]:
            # Structure sometimes: {"pairAddress":"...", "chainId":"solana","boosts":123}
            pa = b.get("pairAddress")
            cnt = b.get("boosts") or b.get("boostCount") or b.get("count")
            if pa and isinstance(cnt, (int, float)):
                boosts_map[pa] = int(cnt)
    return boosts_map

# --------------------------- parsing helpers --------------------------- #

def normalize_pair(p: dict) -> Optional[dict]:
    """
    Convert Dexscreener pair JSON to a flat row with typed numeric fields.
    """
    try:
        chain = (p.get("chainId") or p.get("chain") or "").lower()
        dex = (p.get("dexId") or p.get("dex") or "").lower()
        base = (p.get("baseToken", {}) or {}).get("symbol") or p.get("baseSymbol")
        quote = (p.get("quoteToken", {}) or {}).get("symbol") or p.get("quoteSymbol")
        base_addr = (p.get("baseToken", {}) or {}).get("address")
        pair_addr = p.get("pairAddress") or p.get("pairId") or p.get("address")

        liq_usd = to_float((p.get("liquidity", {}) or {}).get("usd"))
        vol24 = to_float((p.get("volume", {}) or {}).get("h24"))

        # txns: sometimes in p.get("txns", {"h1":{"buys":..,"sells":..}, "m5":...})
        txns5m_obj = (p.get("txns", {}) or {}).get("m5") or {}
        txns1h_obj = (p.get("txns", {}) or {}).get("h1") or {}
        txns5m = to_float(txns5m_obj.get("buys"), 0) + to_float(txns5m_obj.get("sells"), 0)
        txns1h = to_float(txns1h_obj.get("buys"), 0) + to_float(txns1h_obj.get("sells"), 0)
        buys5m = to_float(txns5m_obj.get("buys"), 0)
        sells5m = to_float(txns5m_obj.get("sells"), 0)
        buy_ratio_5m = (buys5m / max(1.0, buys5m + sells5m)) if (buys5m + sells5m) > 0 else None

        chg5m = to_float((p.get("priceChange", {}) or {}).get("m5"))
        chg1h = to_float((p.get("priceChange", {}) or {}).get("h1"))
        chg24 = to_float((p.get("priceChange", {}) or {}).get("h24"))

        # age: sometimes "pairCreatedAt" (ms)
        age_hours = None
        created_ms = p.get("pairCreatedAt")
        if created_ms:
            try:
                age_hours = (now_ts()*1000 - int(created_ms)) / 1000 / 3600.0
            except Exception:
                age_hours = None

        return {
            "chain": chain,
            "dex": dex,
            "base_symbol": (base or "").upper(),
            "quote_symbol": (quote or "").upper(),
            "base_address": base_addr,
            "pair_address": pair_addr,
            "liq_usd": liq_usd,
            "vol24_usd": vol24,
            "txns5m": txns5m,
            "txns1h": txns1h,
            "buy_ratio_5m": buy_ratio_5m,
            "chg5m": chg5m,
            "chg1h": chg1h,
            "chg24h": chg24,
            "age_hours": age_hours,
        }
    except Exception:
        return None

# --------------------------- scoring logic ----------------------------- #

def compute_viral_score(df: pd.DataFrame, boosts_map: Dict[str, int]) -> pd.DataFrame:
    """
    Combine normalized features into a simple "ViralScore".
    Factors: boost count, 5m buy ratio, acceleration (txns1h / max(1, txns5m)),
             turnover (vol24 / liq), price changes (small weight).
    """
    if df.empty:
        return df

    df = df.copy()

    # attach boosts
    df["boosts"] = df["pair_address"].map(lambda x: boosts_map.get(str(x), 0))

    # features
    df["accel"] = (df["txns1h"].fillna(0.0) / (df["txns5m"].fillna(0.0).clip(lower=1.0))).clip(upper=12.0)
    df["turnover_ratio"] = (df["vol24_usd"].fillna(0.0) / df["liq_usd"].fillna(1.0)).replace([math.inf, -math.inf], 0)

    # bounded transforms
    br = df["buy_ratio_5m"].fillna(0.5).clip(0, 1)
    accel = df["accel"].fillna(1.0).clip(0, 12)
    tovr = df["turnover_ratio"].fillna(0.0).clip(0, 10)
    boosts = df["boosts"].fillna(0.0).clip(0, 100)

    chg5 = df["chg5m"].fillna(0.0).clip(-50, 50) / 50.0
    chg1 = df["chg1h"].fillna(0.0).clip(-50, 50) / 50.0
    chg24 = df["chg24h"].fillna(0.0).clip(-200, 200) / 200.0

    # weighted sum (tuned to favor fresh heat + buy imbalance + social boosts)
    df["ViralScore"] = (
        0.35 * br
        + 0.25 * (accel / 4.0)              # normalize 0..~3
        + 0.20 * (tovr / 2.0)               # normalize 0..~5
        + 0.10 * (boosts / 10.0)            # 0..10 boosts → up to +0.1
        + 0.05 * chg5
        + 0.03 * chg1
        + 0.02 * chg24
    ).clip(0, 1.0)

    return df

# -------------------------- quality gates ------------------------------ #

MEME_PATTERN_DEFAULT = r"(PEPE|DOGE|SHIB|BONK|WIF|FLOKI|INU|MOON|PUMP|MEME|CAT|DOG|HAMSTER|PANDA|KEK|SOON|GIGA|PENG|LUNA|BOZO|DEGEN|FART|BABY|ELON|TRUMP|BIDEN|HARRY|HUSKY|FROG|FROGE|KIRK|MOG|WOJAK|RUG|PONZI|GOAT|TURTLE|NINJA|NARD|BASED|BASED|BASED|BASED)$"

def is_memeish(symbol: str, name: Optional[str] = None, pattern: Optional[str] = None) -> bool:
    s = (symbol or "").upper()
    nm = (name or "").upper()
    pat = re.compile(pattern or MEME_PATTERN_DEFAULT, re.IGNORECASE)
    # heuristic: short noisy tickers with vowels or common meme tokens
    if pat.search(s) or pat.search(nm):
        return True
    # extra: contains obvious words
    obvious = ["PEPE", "DOGE", "SHIB", "BONK", "WIF", "INU", "MEME", "MOON", "PUMP", "BASED", "PENG", "FROG", "MOG"]
    return any(x in s for x in obvious) or any(x in nm for x in obvious)

def audit(df: pd.DataFrame, tag: str, enabled: bool):
    if enabled:
        print(f"[audit] {tag}: {len(df)} rows")

def apply_quality_gates(
    df: pd.DataFrame,
    chains: List[str],
    only_quotes: Optional[List[str]],
    exclude_dex: Optional[List[str]],
    meme_only: bool,
    meme_regex: Optional[str],
    min_liq: float,
    max_liq: float,
    min_vol24: float,
    abs_vol24_floor: float,
    min_turnover: float,
    age_min: float,
    age_max: float,
    min_accel: float,
    min_buys5m: float,
    min_txns1h: int,
    min_txns5m: int,
    debug: bool
) -> pd.DataFrame:

    x = df.copy()
    audit(x, "start", debug)

    # chain gate
    if chains:
        chains_lc = [c.strip().lower() for c in chains if c.strip()]
        x = x[x["chain"].isin(chains_lc)]
        audit(x, "after chains", debug)

    # dex exclusion
    if exclude_dex:
        ex = set(d.strip().lower() for d in exclude_dex if d.strip())
        if ex:
            x = x[~x["dex"].isin(ex)]
            audit(x, "after exclude_dex", debug)

    # quote gate
    if only_quotes:
        oq = set(q.upper().strip() for q in only_quotes if q.strip())
        if oq:
            x = x[x["quote_symbol"].isin(oq)]
            audit(x, "after only_quotes", debug)

    # meme-only
    if meme_only:
        x = x[x["base_symbol"].apply(lambda s: is_memeish(s, None, meme_regex))]
        audit(x, "after meme-only", debug)

    # numeric sanity
    x["liq_usd"] = x["liq_usd"].fillna(0.0)
    x["vol24_usd"] = x["vol24_usd"].fillna(0.0)
    x["txns5m"] = x["txns5m"].fillna(0.0)
    x["txns1h"] = x["txns1h"].fillna(0.0)
    x["buy_ratio_5m"] = x["buy_ratio_5m"].fillna(0.0)
    x["accel"] = x["accel"].fillna(0.0)
    x["age_hours"] = x["age_hours"].fillna(float("nan"))

    # liquidity/volume/turnover gates
    if min_liq is not None:
        x = x[x["liq_usd"] >= float(min_liq)]
    if max_liq is not None and max_liq > 0:
        x = x[x["liq_usd"] <= float(max_liq)]
    audit(x, "after liq bounds", debug)

    if abs_vol24_floor is not None:
        x = x[x["vol24_usd"] >= float(abs_vol24_floor)]
    if min_vol24 is not None:
        x = x[x["vol24_usd"] >= float(min_vol24)]
    audit(x, "after vol floors", debug)

    # turnover ratio gate
    x["turnover_ratio"] = (x["vol24_usd"] / x["liq_usd"].replace(0, float("inf"))).replace([math.inf, -math.inf], 0)
    if min_turnover is not None:
        x = x[x["turnover_ratio"] >= float(min_turnover)]
    audit(x, "after turnover", debug)

    # age gate
    if age_min is not None:
        x = x[(x["age_hours"].isna()) | (x["age_hours"] >= float(age_min))]
    if age_max is not None:
        x = x[(x["age_hours"].isna()) | (x["age_hours"] <= float(age_max))]
    audit(x, "after age", debug)

    # momentum & flow
    if min_accel is not None:
        x = x[x["accel"] >= float(min_accel)]
    if min_buys5m is not None:
        x = x[x["buy_ratio_5m"] >= float(min_buys5m)]
    if min_txns1h is not None:
        x = x[x["txns1h"] >= float(min_txns1h)]
    if min_txns5m is not None:
        x = x[x["txns5m"] >= float(min_txns5m)]
    audit(x, "after momentum/txns", debug)

    return x

# --------------------------- Telegram ---------------------------------- #

def send_telegram(msg: str, token: str, chat_id: str):
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}
        SESSION.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"[telegram] failed: {e}")

def dexscreener_link(chain: str, pair: str) -> str:
    c = (chain or "").lower()
    return f"https://dexscreener.com/{c}/{pair}"

def build_group_alert(df: pd.DataFrame, top: int = 10) -> str:
    rows = []
    for _, r in df.head(top).iterrows():
        rows.append(
            "• *{sym}* [{chain}/{dex}] — VS {vs:.2f} | Accel×{acc:.2f} | Buys5m {br:.2f}\n"
            "  Liq ${liq:,.0f} | Vol24 ${vol:,.0f} | Δ5m {c5} | Δ1h {c1} | Δ24h {c24}\n"
            "  Age {age}h | [Dexscreener]({link})".format(
                sym=r["base_symbol"],
                chain=r["chain"], dex=r["dex"],
                vs=r["ViralScore"], acc=r["accel"], br=r["buy_ratio_5m"],
                liq=r["liq_usd"], vol=r["vol24_usd"],
                c5=pct_str(r["chg5m"]), c1=pct_str(r["chg1h"]), c24=pct_str(r["chg24h"]),
                age=int(r["age_hours"]) if pd.notnull(r["age_hours"]) else -1,
                link=dexscreener_link(r["chain"], r["pair_address"])
            )
        )
    return "🔥 *Pre-Viral Watchlist*\n" + "\n".join(rows)

# cooldown cache (avoid re-alerting same pair for N hours)
CACHE_FILE = "alert_cache.json"
COOLDOWN_HOURS = 6

def load_cache(path=CACHE_FILE):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache, path=CACHE_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

# ------------------------------ main ----------------------------------- #

def seed_symbols(args) -> List[str]:
    seeds = []
    seeds += fetch_coingecko_trending_public()
    if args.birdeye_key:
        seeds += fetch_birdeye_solana_trending(args.birdeye_key, limit=min(100, args.seed_boosts_limit))
    seeds = [s for s in seeds if s]
    return list(dict.fromkeys(seeds))

def expand_pairs_from_seeds(seeds: List[str], per_symbol: int) -> List[dict]:
    print("[*] Expanding via Dexscreener search + Boosts…")
    pairs = []
    for s in seeds:
        ps = dexscreener_search_pairs_by_symbol(s, limit_per_symbol=per_symbol)
        if ps:
            pairs.extend(ps)
        # tiny sleep to be polite
        time.sleep(0.05)
    return pairs

def run_once(args):
    seeds = seed_symbols(args)
    # fallback if no seeds
    if not seeds:
        print("[!] No seeds gathered; aborting.")
        return

    pairs_raw = expand_pairs_from_seeds(seeds, per_symbol=max(20, min(100, args.seed_boosts_limit)))
    df = pd.DataFrame([normalize_pair(p) for p in pairs_raw if normalize_pair(p) is not None])
    if args.debug:
        print(f"[debug] seeds->pairs (raw): {len(df)} rows")

    boosts_map = fetch_dexscreener_boosts_map()
    df = compute_viral_score(df, boosts_map)
    if args.debug:
        print(f"[debug] scored before gates: {len(df)} rows")
        # show a few rows
        with pd.option_context("display.max_columns", 20, "display.width", 200):
            print(df.head(10).to_string(index=False))

    # Apply gates
    chains = [c.strip() for c in (args.chains or "").split(",") if c.strip()]
    only_quotes = [q.strip() for q in (args.only_quotes or "").split(",") if q.strip()]
    exclude_dex = [d.strip() for d in (args.exclude_dex or "").split(",") if d.strip()]
    meme_regex = args.meme_regex if args.meme_regex else None

    df = apply_quality_gates(
        df=df,
        chains=chains,
        only_quotes=only_quotes,
        exclude_dex=exclude_dex,
        meme_only=args.meme_only,
        meme_regex=meme_regex,
        min_liq=args.min_liquidity,
        max_liq=args.max_liquidity,
        min_vol24=args.min_vol24,
        abs_vol24_floor=args.abs_vol24_floor,
        min_turnover=args.min_turnover_ratio,
        age_min=args.age_min,
        age_max=args.age_max,
        min_accel=args.min_accel,
        min_buys5m=args.min_buys5m,
        min_txns1h=args.min_txns1h,
        min_txns5m=args.min_txns5m,
        debug=args.debug
    )

    # Empty guard (prevents KeyError in sort/print)
    if df is None or df.empty:
        if args.debug:
            print("[debug] after gates: 0 rows")
        print("[!] No candidates.\n\nTop picks (why):")
        return

    # Sort & take top
    if "ViralScore" in df.columns:
        df = df.sort_values("ViralScore", ascending=False).copy()

    if args.debug:
        print(f"[debug] after gates: {len(df)} rows")

    # Pretty print small table
    show_cols = [
        "ViralScore","seed","chain","dex","base_symbol","quote_symbol",
        "liq_usd","vol24_usd","txns5m","txns1h","buy_ratio_5m","accel",
        "chg5m","chg1h","chg24h","age_hours","pair_address"
    ]
    # attach 'seed' column if not present
    if "seed" not in df.columns:
        df["seed"] = "coingecko"

    out = df[show_cols].head(args.top).copy()
    # format $ values & pct for printing
    def money(x): 
        return f"${x:,.0f}" if pd.notnull(x) else "—"
    out["liq_usd"] = out["liq_usd"].apply(money)
    out["vol24_usd"] = out["vol24_usd"].apply(money)
    out["chg5m"] = out["chg5m"].apply(pct_str)
    out["chg1h"] = out["chg1h"].apply(pct_str)
    out["chg24h"] = out["chg24h"].apply(pct_str)

    print(out.to_string(index=False))

    # Console top-picks summary
    print("\nTop picks (why):")
    for _, r in df.head(args.top).iterrows():
        age_str = f" | age {r['age_hours']:.1f}h" if pd.notnull(r["age_hours"]) else ""
        print(f" - {r['base_symbol']} [{r['chain']}/{r['dex']}] ViralScore={r['ViralScore']:.2f} "
              f"| liq≈${r['liq_usd']:,.0f} vol24≈${r['vol24_usd']:,.0f} "
              f"| accel×{r['accel']:.2f} buys5m={r['buy_ratio_5m']:.2f} "
              f"| Δ5m {pct_str(r['chg5m'])} Δ1h {pct_str(r['chg1h'])} Δ24h {pct_str(r['chg24h'])}{age_str}")

    # ---------------- Telegram alerts ---------------- #
    if args.telegram_token and args.telegram_chat:
        alerts = df[
            (df["ViralScore"] >= args.alert_threshold)
            & (df["accel"] >= args.alert_accel)
            & (df["buy_ratio_5m"] >= args.alert_buys)
        ].copy()

        if not alerts.empty:
            # cooldown de-dupe
            alerts["__id"] = alerts[["chain","pair_address"]].astype(str).agg("|".join, axis=1)
            cache = load_cache()
            cutoff = now_ts() - int(args.cooldown_hours*3600)
            alerts = alerts[alerts["__id"].map(lambda k: cache.get(k, 0) < cutoff)].copy()

            if not alerts.empty:
                # grouped alert
                msg = build_group_alert(alerts, top=min(len(alerts), args.top))
                send_telegram(msg, args.telegram_token, args.telegram_chat)
                print(f"[telegram] sent grouped alert for {len(alerts)} coins")

                # update cache
                tnow = now_ts()
                for _, r in alerts.iterrows():
                    cache[r["__id"]] = tnow
                save_cache(cache)

# -------------------------------- CLI ---------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Find pre-viral meme coins with liquidity & momentum.")
    ap.add_argument("--chains", type=str, default="solana,base", help="comma list of chains (e.g., solana,base,bsc)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed-boosts-limit", type=int, default=200, help="per-symbol search cap (20..100 recommended)")

    # filters
    ap.add_argument("--meme-only", action="store_true")
    ap.add_argument("--meme-regex", type=str, default="", help="custom regex for meme tickers")
    ap.add_argument("--only-quotes", type=str, default="USDC,SOL,WETH,USDT")
    ap.add_argument("--exclude-dex", type=str, default="", help="comma list of dex ids to exclude (e.g., pumpswap)")
    ap.add_argument("--min-liquidity", type=float, default=20000)
    ap.add_argument("--max-liquidity", type=float, default=400000)
    ap.add_argument("--min-vol24", type=float, default=20000)
    ap.add_argument("--abs-vol24-floor", type=float, default=15000)
    ap.add_argument("--min-turnover-ratio", type=float, default=0.002)
    ap.add_argument("--age-min", type=float, default=2)
    ap.add_argument("--age-max", type=float, default=240000, help="hours; set ~120 for true pre-viral")
    ap.add_argument("--min-accel", type=float, default=1.05)
    ap.add_argument("--min-buys5m", type=float, default=0.55)
    ap.add_argument("--min-txns1h", type=int, default=0)
    ap.add_argument("--min-txns5m", type=int, default=0)

    # telegram
    ap.add_argument("--telegram-token", type=str, default="8340970668:AAErYSWVmr2gv99dMb78jyLV9G1nJk3eVZk")
    ap.add_argument("--telegram-chat", type=str, default="-5020507132")
    ap.add_argument("--alert-threshold", type=float, default=0.45)
    ap.add_argument("--alert-accel", type=float, default=1.2)
    ap.add_argument("--alert-buys", type=float, default=0.6)
    ap.add_argument("--cooldown-hours", type=float, default=6.0)

    # optional birdeye
    ap.add_argument("--birdeye-key", type=str, default=os.getenv("BIRDEYE_API_KEY", ""))

    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    run_once(args)

if __name__ == "__main__":
    main()
