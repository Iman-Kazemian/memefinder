#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memecoin_previral.py
Scan DEX pairs for early, liquid, momentum-leaning meme coins and (a) push alerts to Telegram
and (b) append structured signals to signals/signals.csv (committed by the Action).

Public data:
- CoinGecko trending (no key)
- Dexscreener search + boosts (no key)

Optional:
- Birdeye Solana trending (BIRDEYE_API_KEY via env or --birdeye-key)

Signals CSV schema (append-only):
ts_iso,unix_ms,chain,dex,base_symbol,quote_symbol,pair_address,base_address,liq_usd,vol24_usd,
txns5m,txns1h,buy_ratio_5m,accel,chg5m,chg1h,chg24h,age_hours,turnover_ratio,boosts,viral_score
"""
import os, re, json, time, math, argparse, csv
from typing import List, Dict, Optional
import requests
import pandas as pd

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "memecoin-previral/2.0"})

def http_get(url: str, params=None, headers=None, timeout=20):
    try:
        r = SESSION.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
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

def now_ts_s() -> int:
    return int(time.time())

def coingecko_trending_symbols() -> List[str]:
    print("[*] Seeds: CoinGecko trending…")
    js = http_get("https://api.coingecko.com/api/v3/search/trending")
    out = []
    if js and isinstance(js.get("coins"), list):
        for c in js["coins"]:
            item = c.get("item") or {}
            sym = (item.get("symbol") or "").upper().strip()
            if sym:
                out.append(sym)
    if not out:  # keep pipeline alive
        out = ["WIF","BONK","PEPE","DOGE","PENGU","FROG","TRUMP","BODEN"]
    return list(dict.fromkeys(out))

def birdeye_trending_symbols(key: str, limit=50) -> List[str]:
    if not key: 
        return []
    print("[*] Seeds: Birdeye Solana trending (optional)…")
    js = http_get(
        "https://public-api.birdeye.so/defi/v3/token/trending",
        params={"chain":"solana","limit":limit},
        headers={"x-api-key": key}
    )
    out = []
    if js and isinstance(js.get("data"), dict):
        for t in js["data"].get("tokens", []):
            s = (t.get("symbol") or "").upper().strip()
            if s: out.append(s)
    return list(dict.fromkeys(out))

def dexscreener_search(q: str) -> List[dict]:
    return (http_get("https://api.dexscreener.com/latest/dex/search", params={"q": q}) or {}).get("pairs", []) or []

def fetch_boosts_map() -> Dict[str, int]:
    print("[*] Fetching boosts map…")
    js = http_get("https://api.dexscreener.com/latest/dex/boosts")
    m = {}
    if js and isinstance(js.get("boosts"), list):
        for b in js["boosts"]:
            pa = b.get("pairAddress")
            cnt = b.get("boosts") or b.get("boostCount") or b.get("count")
            if pa and isinstance(cnt, (int,float)):
                m[str(pa)] = int(cnt)
    return m

def normalize_pair(p: dict) -> Optional[dict]:
    try:
        chain = (p.get("chainId") or p.get("chain") or "").lower()
        dex = (p.get("dexId") or p.get("dex") or "").lower()
        base_sym = (p.get("baseToken", {}) or {}).get("symbol") or p.get("baseSymbol") or ""
        quote_sym = (p.get("quoteToken", {}) or {}).get("symbol") or p.get("quoteSymbol") or ""
        base_addr = (p.get("baseToken", {}) or {}).get("address")
        pair_addr = p.get("pairAddress") or p.get("pairId") or p.get("address")
        liq = to_float((p.get("liquidity", {}) or {}).get("usd"), 0.0)
        vol24 = to_float((p.get("volume", {}) or {}).get("h24"), 0.0)
        tx5 = (p.get("txns", {}) or {}).get("m5") or {}
        tx1 = (p.get("txns", {}) or {}).get("h1") or {}
        buys5, sells5 = to_float(tx5.get("buys"),0.0), to_float(tx5.get("sells"),0.0)
        txns5m = (buys5 or 0.0) + (sells5 or 0.0)
        txns1h = (to_float(tx1.get("buys"),0.0) or 0.0) + (to_float(tx1.get("sells"),0.0) or 0.0)
        buy_ratio_5m = (buys5 / max(1.0, (buys5 or 0.0) + (sells5 or 0.0))) if (buys5 or 0.0) + (sells5 or 0.0) > 0 else 0.0
        chg5 = to_float((p.get("priceChange", {}) or {}).get("m5"), 0.0)
        chg1 = to_float((p.get("priceChange", {}) or {}).get("h1"), 0.0)
        chg24 = to_float((p.get("priceChange", {}) or {}).get("h24"), 0.0)
        age_h = None
        created_ms = p.get("pairCreatedAt")
        if created_ms:
            try:
                age_h = max(0.0, (now_ts_s()*1000 - int(created_ms)) / 1000 / 3600.0)
            except Exception:
                age_h = None
        return {
            "chain": chain,
            "dex": dex,
            "base_symbol": base_sym.upper(),
            "quote_symbol": quote_sym.upper(),
            "base_address": base_addr,
            "pair_address": pair_addr,
            "liq_usd": liq,
            "vol24_usd": vol24,
            "txns5m": txns5m,
            "txns1h": txns1h,
            "buy_ratio_5m": buy_ratio_5m,
            "chg5m": chg5,
            "chg1h": chg1,
            "chg24h": chg24,
            "age_hours": age_h,
        }
    except Exception:
        return None

MEME_REGEX_DEFAULT = r"(PEPE|DOGE|SHIB|BONK|WIF|FLOKI|INU|MOON|PUMP|MEME|CAT|DOG|HAM|PANDA|KEK|SOON|GIGA|PENG|FROG|MOG|ELON|TRUMP|BODEN|WOJAK|PONKE|HOPPY|DEGEN|BOZO|RUG|GOAT|TURTLE|BASED)"

def is_memeish(symbol: str, name: Optional[str] = None, custom: Optional[str] = None) -> bool:
    s = (symbol or "").upper()
    nm = (name or "").upper()
    pat = re.compile(custom or MEME_REGEX_DEFAULT, re.IGNORECASE)
    if pat.search(s) or pat.search(nm): return True
    return False

def compute_score(df: pd.DataFrame, boosts: Dict[str,int]) -> pd.DataFrame:
    if df.empty: return df
    x = df.copy()
    x["boosts"] = x["pair_address"].map(lambda k: boosts.get(str(k), 0))
    x["turnover_ratio"] = (x["vol24_usd"].fillna(0.0) / x["liq_usd"].replace(0, float("inf"))).replace([math.inf,-math.inf],0)
    x["accel"] = (x["txns5m"].fillna(0.0) * 12.0 / x["txns1h"].replace(0, 1e-9)).clip(0, 12)

    br = x["buy_ratio_5m"].fillna(0.5).clip(0,1)
    accel = x["accel"].fillna(1.0).clip(0,12)/4.0
    tov = x["turnover_ratio"].fillna(0).clip(0,10)/2.0
    bst = x["boosts"].fillna(0).clip(0,100)/10.0
    c5 = x["chg5m"].fillna(0).clip(-50,50)/50.0
    c1 = x["chg1h"].fillna(0).clip(-50,50)/50.0
    c24 = x["chg24h"].fillna(0).clip(-200,200)/200.0

    x["ViralScore"] = (0.35*br + 0.25*accel + 0.20*tov + 0.10*bst + 0.05*c5 + 0.03*c1 + 0.02*c24).clip(0,1)
    return x

def audit(df: pd.DataFrame, tag: str, dbg: bool):
    if dbg: print(f"[audit] {tag}: {len(df)} rows")

def apply_gates(df: pd.DataFrame, *, chains, only_quotes, exclude_dex, meme_only, meme_regex,
                min_liq, max_liq, min_vol24, abs_vol24_floor, min_turnover, age_min, age_max,
                min_accel, min_buys5m, min_txns1h, min_txns5m, debug) -> pd.DataFrame:
    x = df.copy()
    audit(x, "start", debug)

    if chains:
        cc = [c.strip().lower() for c in chains if c.strip()]
        x = x[x["chain"].isin(cc)]
        audit(x, "after chains", debug)

    if exclude_dex:
        ex = set(d.strip().lower() for d in exclude_dex if d.strip())
        if ex:
            x = x[~x["dex"].isin(ex)]
            audit(x, "after exclude_dex", debug)

    if only_quotes:
        oq = set(q.upper().strip() for q in only_quotes if q.strip())
        if oq:
            x = x[x["quote_symbol"].isin(oq)]
            audit(x, "after only_quotes", debug)

    if meme_only:
        x = x[x["base_symbol"].apply(lambda s: is_memeish(s, None, meme_regex))]
        audit(x, "after meme-only", debug)

    # numeric fills
    for c in ["liq_usd","vol24_usd","txns5m","txns1h","buy_ratio_5m","accel","age_hours","turnover_ratio"]:
        if c not in x.columns: x[c]=None
    x["liq_usd"]=x["liq_usd"].fillna(0.0); x["vol24_usd"]=x["vol24_usd"].fillna(0.0)
    x["txns5m"]=x["txns5m"].fillna(0.0); x["txns1h"]=x["txns1h"].fillna(0.0)
    x["buy_ratio_5m"]=x["buy_ratio_5m"].fillna(0.0)
    x["accel"]=x["accel"].fillna(0.0)
    x["age_hours"]=x["age_hours"].astype(float)

    if min_liq is not None: x = x[x["liq_usd"] >= float(min_liq)]
    if max_liq: x = x[x["liq_usd"] <= float(max_liq)]
    audit(x, "after liq bounds", debug)

    if abs_vol24_floor is not None: x = x[x["vol24_usd"] >= float(abs_vol24_floor)]
    if min_vol24 is not None: x = x[x["vol24_usd"] >= float(min_vol24)]
    audit(x, "after vol floors", debug)

    x["turnover_ratio"] = (x["vol24_usd"] / x["liq_usd"].replace(0,float("inf"))).replace([math.inf,-math.inf],0)
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

# ---------- signals.csv persistence (repo file) ----------
def append_signal_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts_iso","unix_ms","chain","dex","base_symbol","quote_symbol","pair_address","base_address",
            "liq_usd","vol24_usd","txns5m","txns1h","buy_ratio_5m","accel","chg5m","chg1h","chg24h",
            "age_hours","turnover_ratio","boosts","viral_score"
        ])
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def send_telegram(msg: str, token: str, chat_id: str):
    if not token or not chat_id: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=15
        )
    except Exception as e:
        print(f"[telegram] failed: {e}")

def ds_link(chain: str, pair: str) -> str:
    return f"https://dexscreener.com/{(chain or '').lower()}/{pair}"

def build_group_alert(df: pd.DataFrame, top=10) -> str:
    rows=[]
    for _, r in df.head(top).iterrows():
        rows.append(
            "• *{sym}* [{chain}/{dex}] — VS {vs:.2f} | Accel×{acc:.2f} | Buys5m {br:.2f}\n"
            "  Liq ${liq:,.0f} | Vol24 ${vol:,.0f} | Δ5m {c5} | Δ1h {c1} | Δ24h {c24}\n"
            "  Age {age}h | [Dexscreener]({link})".format(
                sym=r["base_symbol"], chain=r["chain"], dex=r["dex"],
                vs=r["ViralScore"], acc=r["accel"], br=r["buy_ratio_5m"],
                liq=r["liq_usd"], vol=r["vol24_usd"],
                c5=pct_str(r["chg5m"]), c1=pct_str(r["chg1h"]), c24=pct_str(r["chg24h"]),
                age=int(r["age_hours"]) if pd.notnull(r["age_hours"]) else -1,
                link=ds_link(r["chain"], r["pair_address"])
            )
        )
    return "🔥 *Pre-Viral Watchlist*\n" + "\n".join(rows)

CACHE_FILE = "alert_cache.json"

def load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE,"r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cache(c):
    try:
        with open(CACHE_FILE,"w",encoding="utf-8") as f:
            json.dump(c,f)
    except Exception:
        pass

def run_once(args):
    # seeds
    seeds = coingecko_trending_symbols()
    seeds += birdeye_trending_symbols(args.birdeye_key, limit=min(100, args.seed_boosts_limit))
    seeds = list(dict.fromkeys([s for s in seeds if s]))
    if not seeds:
        print("[!] No seeds gathered; aborting.")
        return

    # expand
    print("[*] Expanding via Dexscreener search + Boosts…")
    raw=[]
    per_symbol = max(20, min(100, args.seed_boosts_limit))
    for s in seeds:
        ps = dexscreener_search(s)[:per_symbol]
        raw.extend(ps)
        time.sleep(0.05)
    df = pd.DataFrame([normalize_pair(p) for p in raw if normalize_pair(p) is not None])
    if args.debug:
        print(f"[debug] seeds->pairs (raw): {len(df)} rows")

    boosts = fetch_boosts_map()  # if 404 or empty, map={}
    df = compute_score(df, boosts)
    if args.debug:
        print(f"[debug] scored before gates: {len(df)} rows")
        with pd.option_context("display.max_columns", 20, "display.width", 220):
            print(df.head(10).to_string(index=False))

    # filters
    chains = [c.strip() for c in (args.chains or "").split(",") if c.strip()]
    only_quotes = [q.strip() for q in (args.only_quotes or "").split(",") if q.strip()]
    exclude_dex = [d.strip() for d in (args.exclude_dex or "").split(",") if d.strip()]

    df = apply_gates(
        df=df, chains=chains, only_quotes=only_quotes, exclude_dex=exclude_dex,
        meme_only=args.meme_only, meme_regex=args.meme_regex or None,
        min_liq=args.min_liquidity, max_liq=args.max_liquidity, min_vol24=args.min_vol24,
        abs_vol24_floor=args.abs_vol24_floor, min_turnover=args.min_turnover_ratio,
        age_min=args.age_min, age_max=args.age_max,
        min_accel=args.min_accel, min_buys5m=args.min_buys5m, min_txns1h=args.min_txns1h,
        min_txns5m=args.min_txns5m, debug=args.debug
    )

    if df.empty:
        if args.debug: print("[debug] after gates: 0 rows")
        print("[!] No candidates.\n\nTop picks (why):")
        return

    df = df.sort_values("ViralScore", ascending=False).copy()
    if args.debug: print(f"[debug] after gates: {len(df)} rows")

    # print top
    show_cols = ["ViralScore","chain","dex","base_symbol","quote_symbol","liq_usd","vol24_usd",
                 "txns5m","txns1h","buy_ratio_5m","accel","chg5m","chg1h","chg24h","age_hours","pair_address"]
    out = df[show_cols].head(args.top).copy()
    def money(x): return f"${x:,.0f}" if pd.notnull(x) else "—"
    out["liq_usd"]=out["liq_usd"].apply(money); out["vol24_usd"]=out["vol24_usd"].apply(money)
    out["chg5m"]=out["chg5m"].apply(pct_str); out["chg1h"]=out["chg1h"].apply(pct_str); out["chg24h"]=out["chg24h"].apply(pct_str)
    print(out.to_string(index=False))

    print("\nTop picks (why):")
    for _, r in df.head(args.top).iterrows():
        age = f" | age {r['age_hours']:.1f}h" if pd.notnull(r["age_hours"]) else ""
        print(f" - {r['base_symbol']} [{r['chain']}/{r['dex']}] ViralScore={r['ViralScore']:.2f} "
              f"| liq≈${r['liq_usd']:,.0f} vol24≈${r['vol24_usd']:,.0f} "
              f"| accel×{r['accel']:.2f} buys5m={r['buy_ratio_5m']:.2f} "
              f"| Δ5m {pct_str(r['chg5m'])} Δ1h {pct_str(r['chg1h'])} Δ24h {pct_str(r['chg24h'])}{age}")

    # ---- write signals ----
    ts = now_ts_s()
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    rows=[]
    for _, r in df.head(args.top).iterrows():
        rows.append({
            "ts_iso": ts_iso, "unix_ms": ts*1000,
            "chain": r["chain"], "dex": r["dex"], "base_symbol": r["base_symbol"],
            "quote_symbol": r["quote_symbol"], "pair_address": r["pair_address"],
            "base_address": r.get("base_address"),
            "liq_usd": float(r["liq_usd"]), "vol24_usd": float(r["vol24_usd"]),
            "txns5m": float(r["txns5m"]), "txns1h": float(r["txns1h"]),
            "buy_ratio_5m": float(r["buy_ratio_5m"]), "accel": float(r["accel"]),
            "chg5m": float(r["chg5m"]), "chg1h": float(r["chg1h"]), "chg24h": float(r["chg24h"]),
            "age_hours": float(r["age_hours"]) if pd.notnull(r["age_hours"]) else None,
            "turnover_ratio": float(r["turnover_ratio"]), "boosts": int(r["boosts"]),
            "viral_score": float(r["ViralScore"]),
        })
    append_signal_csv(os.path.join("signals","signals.csv"), rows)

    # ---- telegram (cooldown per pair) ----
    if args.telegram_token and args.telegram_chat:
        candidates = df[
            (df["ViralScore"] >= args.alert_threshold)
            & (df["accel"] >= args.alert_accel)
            & (df["buy_ratio_5m"] >= args.alert_buys)
        ].copy()
        if not candidates.empty:
            candidates["__id"] = candidates[["chain","pair_address"]].astype(str).agg("|".join, axis=1)
            cache = load_cache()
            cutoff = now_ts_s() - int(args.cooldown_hours*3600)
            candidates = candidates[candidates["__id"].map(lambda k: cache.get(k,0) < cutoff)]
            if not candidates.empty:
                msg = build_group_alert(candidates, top=min(len(candidates), args.top))
                send_telegram(msg, args.telegram_token, args.telegram_chat)
                print(f"[telegram] sent grouped alert for {len(candidates)} coins")
                tnow = now_ts_s()
                for _, r in candidates.iterrows():
                    cache[r["__id"]] = tnow
                save_cache(cache)

def parse_args():
    ap = argparse.ArgumentParser(description="Find pre-viral meme coins and log signals.")
    ap.add_argument("--chains", type=str, default="solana,base,bsc")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed-boosts-limit", type=int, default=200)

    ap.add_argument("--meme-only", action="store_true")
    ap.add_argument("--meme-regex", type=str, default="")
    ap.add_argument("--only-quotes", type=str, default="USDC,SOL,WETH,USDT")
    ap.add_argument("--exclude-dex", type=str, default="")
    ap.add_argument("--min-liquidity", type=float, default=20000)
    ap.add_argument("--max-liquidity", type=float, default=400000)
    ap.add_argument("--min-vol24", type=float, default=20000)
    ap.add_argument("--abs-vol24-floor", type=float, default=15000)
    ap.add_argument("--min-turnover-ratio", type=float, default=0.002)
    ap.add_argument("--age-min", type=float, default=2)
    ap.add_argument("--age-max", type=float, default=240000)
    ap.add_argument("--min-accel", type=float, default=1.05)
    ap.add_argument("--min-buys5m", type=float, default=0.55)
    ap.add_argument("--min-txns1h", type=int, default=0)
    ap.add_argument("--min-txns5m", type=int, default=0)

    ap.add_argument("--telegram-token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN",""))
    ap.add_argument("--telegram-chat", type=str, default=os.getenv("TELEGRAM_CHAT_ID",""))
    ap.add_argument("--alert-threshold", type=float, default=0.45)
    ap.add_argument("--alert-accel", type=float, default=1.2)
    ap.add_argument("--alert-buys", type=float, default=0.6)
    ap.add_argument("--cooldown-hours", type=float, default=6.0)

    ap.add_argument("--birdeye-key", type=str, default=os.getenv("BIRDEYE_API_KEY",""))
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    run_once(parse_args())
