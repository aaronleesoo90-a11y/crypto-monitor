# app.py — Crypto Monitor (adaptive risk, 0–100 sentiment, ARIMA, optional Google Sheets logging)

from __future__ import annotations
import time, re, warnings, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

import ccxt, requests, feedparser, ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA

# Optional Google Sheets (graceful fallback if not installed/configured)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except Exception:
    GSHEETS_AVAILABLE = False

warnings.filterwarnings("ignore")

# =================== Page / Styles ===================
st.set_page_config(page_title="Crypto Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1200px}
h1,h2,h3{margin-top:.6rem}
.card{border-radius:14px;padding:12px 14px;border:1px solid rgba(0,0,0,.06);box-shadow:0 8px 20px rgba(0,0,0,.05)}
.chip{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.85rem;font-weight:600}
.chip.Low{background:#e9f9ee;color:#106b35}
.chip.Medium{background:#fff7e6;color:#8a5300}
.chip.High{background:#ffefef;color:#9b1b1b}
.badge-pos{color:#0b7a34;font-weight:700}
.badge-neg{color:#b00020;font-weight:700}
.small{color:#6b7280;font-size:.85rem}
.dataframe td,.dataframe th{padding:6px 8px}
</style>
""", unsafe_allow_html=True)

# =================== Utils ===================
analyzer = SentimentIntensityAnalyzer()

@dataclass
class Bracket:
    entry: float; tp: float; sl: float

def fmt(x: float) -> str:
    try:
        if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return "-"
        return f"{x:.6f}" if x < 1000 else f"{x:.2f}"
    except: return "-"

def usd(x: float) -> str:
    try:
        if x >= 1e9: return f"${x/1e9:.2f}B"
        if x >= 1e6: return f"${x/1e6:.2f}M"
        if x >= 1e3: return f"${x/1e3:.2f}k"
        return f"${x:.2f}"
    except: return "-"

def tf_to_freq(tf: str) -> str:
    return {"15m":"15min","1h":"H","4h":"4H"}.get(tf.lower().strip(), tf)

def tf_to_minutes(tf: str) -> int:
    return {"15m":15, "1h":60, "4h":240}.get(tf.lower().strip(), 60)

def to_0_100(signed: float) -> int:
    v = int(round((float(signed)+1.0)*50))
    return max(0, min(100, v))

# =================== Sidebar ===================
with st.sidebar:
    st.title("Settings")
    exchange_id = st.text_input("Exchange", value=st.session_state.get("exchange_id","coinbase"))
    quote = st.text_input("Quote", value=st.session_state.get("quote","USDC"))
    st.caption("Tip: coinbase/USDC or binance/USDT")

    st.subheader("Auto‑refresh (Live tab)")
    rate = st.selectbox("Interval", ["Off","10s","30s","60s"], index=1)
    rate_sec = {"Off":0,"10s":10,"30s":30,"60s":60}[rate]

    st.subheader("News (optional)")
    cp_token = st.text_input("CryptoPanic token", value=st.session_state.get("cp_token",""), type="password")

    st.subheader("Google Sheets (optional)")
    sheet_id_input = st.text_input("Spreadsheet ID", value=st.session_state.get("gsheet_id",""))
    st.caption("Put service account JSON in secrets as 'gsheets_service_account'. Share the Sheet with that client email.")

    st.subheader("Advanced")
    with st.expander("Show advanced"):
        price_ttl = st.number_input("Price TTL (s)", 3, value=int(st.session_state.get("price_ttl",10)))
        news_ttl  = st.number_input("News TTL (s)", 10, value=int(st.session_state.get("news_ttl",90)))
        st.caption("Shorter TTL = fresher data, more API calls.")
        st.markdown("---")
        st.caption("Trading allocations")
        total_capital = st.number_input("Total capital", 0.0, value=float(st.session_state.get("capital",100.0)), step=10.0)
        fee_pct = st.number_input("Round-trip fee %", 0.0, value=float(st.session_state.get("fee_pct",0.10)), step=0.05)
        alloc_low  = st.slider("Low %", 0,100, int(st.session_state.get("alloc_low",40)))
        alloc_med  = st.slider("Med %", 0,100, int(st.session_state.get("alloc_med",40)))
        alloc_high = st.slider("High %",0,100, int(st.session_state.get("alloc_high",20)))
        if alloc_low+alloc_med+alloc_high != 100:
            st.warning("Allocations should sum to 100%.")
        st.markdown("---")
        sentiment_impact = st.slider("Sentiment impact (max shift)", 0.0, 0.05, float(st.session_state.get("sent_w",0.02)), 0.005)
        bias_floor = st.slider("Ignore bias below (|x|)", 0.0, 0.5, float(st.session_state.get("bias_floor",0.10)), 0.05)
        if st.button("Refresh now"):
            st.cache_data.clear(); st.success("Cleared caches.")

st.session_state.update({
    "exchange_id":exchange_id, "quote":quote, "cp_token":cp_token,
    "price_ttl":locals().get("price_ttl",10), "news_ttl":locals().get("news_ttl",90),
    "capital":locals().get("total_capital",100.0), "fee_pct":locals().get("fee_pct",0.10),
    "alloc_low":locals().get("alloc_low",40), "alloc_med":locals().get("alloc_med",40), "alloc_high":locals().get("alloc_high",20),
    "sent_w":locals().get("sentiment_impact",0.02), "bias_floor":locals().get("bias_floor",0.10),
    "gsheet_id": sheet_id_input
})

# =================== Google Sheets helpers ===================
SHEET_HEADERS = [
    "ts_utc","kind","symbol","timeframe","last_price",
    "horizon_steps","horizon_tf","pred_end_price","pred_change_pct",
    "sentiment_bias","risk_bucket","entry","tp","sl",
    "due_ts_utc","status","actual_price","realized_change_pct","correct"
]

def _open_logs_sheet(sheet_id: str):
    if not GSHEETS_AVAILABLE or not sheet_id:
        return None
    try:
        sa = st.secrets.get("gsheets_service_account", None)
        if not sa: return None
        if isinstance(sa, str):
            try: sa = json.loads(sa)
            except Exception: return None
        creds = Credentials.from_service_account_info(sa, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        try:
            ws = sh.worksheet("logs")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="logs", rows="1000", cols=str(len(SHEET_HEADERS)+2))
            ws.append_row(SHEET_HEADERS)
        if ws.row_values(1) != SHEET_HEADERS:
            try: ws.delete_rows(1)
            except Exception: pass
            ws.insert_row(SHEET_HEADERS, 1)
        return ws
    except Exception:
        return None

def _append_log(ws, row: dict):
    if ws is None: return
    try:
        ws.append_row([row.get(h,"") for h in SHEET_HEADERS])
    except Exception:
        pass

def _reconcile_logs(ws, ex_id: str):
    if ws is None: return
    try:
        data = ws.get_all_records()
        now = pd.Timestamp.utcnow()
        for i, r in enumerate(data, start=2):
            if r.get("kind")!="forecast" or r.get("status"):
                continue
            due_ts = pd.to_datetime(r.get("due_ts_utc",""), utc=True, errors="coerce")
            if pd.isna(due_ts) or now < due_ts: 
                continue
            symbol = r.get("symbol",""); tf = r.get("horizon_tf","1h")
            try:
                limit = 120 if tf=="15m" else (72 if tf=="1h" else 36)
                df = fetch_ohlcv_df(ex_id, symbol, timeframe=tf, limit=limit, _ttl_key=int(time.time()))
                ts = df.index
                near = ts[np.argmin(np.abs((ts - due_ts).values))]
                actual = float(df.loc[near,"close"])
                p0 = float(r.get("last_price") or 0)
                realized = (actual/p0 - 1.0)*100.0 if p0>0 else None
                pred_change = float(r.get("pred_change_pct") or 0.0)
                correct = (realized is not None) and (np.sign(realized)==np.sign(pred_change))
                ws.update(f"P{i}:R{i}", [[ "closed", actual, realized, "TRUE" if correct else "FALSE" ]])
            except Exception:
                continue
    except Exception:
        pass

# =================== Data (cached with dynamic keys) ===================
@st.cache_data(show_spinner=False)
def make_exchange(ex_id: str):
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    ex.load_markets(); return ex

@st.cache_data(show_spinner=False)
def top_symbols_by_dollar_volume(ex_id: str, quote: str, _ttl_key:int, top_n: int=40) -> List[str]:
    ex = make_exchange(ex_id)
    syms = [s for s,m in ex.markets.items()
            if m.get("spot") and m.get("active") and m.get("quote","").upper()==quote.upper()]
    if not syms: return []
    tickers = ex.fetch_tickers(syms)
    rows=[]
    for sym,t in tickers.items():
        last = t.get("last") or t.get("close") or 0
        base_vol = t.get("baseVolume") or 0
        rows.append((sym, (last or 0)*(base_vol or 0)))
    rows.sort(key=lambda x:x[1], reverse=True)
    return [s for s,_ in rows[:top_n]]

@st.cache_data(show_spinner=False)
def fetch_ohlcv_df(ex_id: str, symbol: str, timeframe: str, limit: int, _ttl_key:int) -> pd.DataFrame:
    ex = make_exchange(ex_id)
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()

@st.cache_data(show_spinner=False)
def fetch_ticker(ex_id: str, symbol: str, _ttl_key:int) -> Dict:
    return make_exchange(ex_id).fetch_ticker(symbol)

@st.cache_data(show_spinner=False)
def latest_news(cp_token: Optional[str], limit: int, _ttl_key:int) -> List[Dict]:
    out=[]
    if cp_token:
        try:
            url="https://cryptopanic.com/api/v1/posts/"
            params={"auth_token":cp_token,"public":"true","kind":"news","regions":"en","page":1}
            r=requests.get(url,params=params,timeout=10); r.raise_for_status()
            for it in r.json().get("results",[])[:limit]:
                title=it.get("title",""); link=it.get("url",""); ts=it.get("published_at","")
                vs=analyzer.polarity_scores(title)["compound"]
                out.append({"source":"CryptoPanic","title":title,"url":link,"ts":ts,"sentiment":vs})
        except: pass
    feeds=[
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
    ]
    for src in feeds:
        try:
            fp=feedparser.parse(src)
            for e in fp.entries[:30]:
                title=e.get("title",""); link=e.get("link",""); ts=e.get("published","") or e.get("updated","")
                vs=analyzer.polarity_scores(title)["compound"]
                out.append({"source":src,"title":title,"url":link,"ts":ts,"sentiment":vs})
        except: continue
    return out[:limit]

# =================== Analytics / Scoring ===================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy()
    if len(d)<60: return d.assign(sma20=np.nan,sma50=np.nan,rsi14=np.nan,bb_bw=np.nan,atr=np.nan,atr_pct=np.nan)
    d["sma20"]=d["close"].rolling(20).mean()
    d["sma50"]=d["close"].rolling(50).mean()
    d["rsi14"]=ta.momentum.RSIIndicator(d["close"],window=14).rsi()
    bb=ta.volatility.BollingerBands(d["close"],window=20,window_dev=2)
    d["bb_bw"]=(bb.bollinger_hband()-bb.bollinger_lband())/d["sma20"]
    atr=ta.volatility.AverageTrueRange(d["high"],d["low"],d["close"],window=14)
    d["atr"]=atr.average_true_range(); d["atr_pct"]=100.0*d["atr"]/d["close"]
    return d

def compute_sentiment_bias(news_items: List[Dict]) -> float:
    if not news_items: return 0.0
    num=den=0.0
    for n in news_items[:100]:
        w=max(0.1, abs(n.get("sentiment",0.0)))
        num+=w*n.get("sentiment",0.0); den+=w
    return num/den if den else 0.0

def composite_score(dfi: pd.DataFrame, news_bias: float=0.0) -> float:
    if dfi.empty or "sma20" not in dfi.columns: return 0.0
    r=dfi.iloc[-1]; s=0.0
    s += 0.30 if r["sma20"]>r["sma50"] else -0.30
    s += 0.20 if r["close"]>r["sma20"] else -0.05
    s += 0.20*np.tanh(r["volume"]/max(1e-9, dfi["volume"].tail(20).mean()) - 1.0)
    s += 0.10*np.tanh((r.get("rsi14",50)-50)/15)
    s += 0.20*np.tanh(news_bias)
    return float(np.clip(s,-1.0,1.0))

def bracket_from_atr(entry: float, atr: float, bucket: str) -> Bracket:
    k={"Low":(1.2,0.8),"Medium":(1.8,1.0)}.get(bucket,(2.5,1.2))
    return Bracket(entry, entry + k[0]*atr, entry - k[1]*atr)

# ---- ADAPTIVE RISK: quantiles + guardrails (adds 'bucket' and 'risk_score') ----
def assign_risk(rows: List[dict]) -> List[dict]:
    if not rows: return rows
    df = pd.DataFrame(rows).copy()

    # normalize names we rely on
    if "dollar_volume" not in df.columns and "Dollar Volume" in df.columns:
        df["dollar_volume"] = pd.to_numeric(df["Dollar Volume"], errors="coerce")
    if "atr_pct" not in df.columns and "ATR %" in df.columns:
        df["atr_pct"] = pd.to_numeric(df["ATR %"], errors="coerce")

    if "atr_pct" not in df.columns or "dollar_volume" not in df.columns:
        df["bucket"]="High"; df["risk_score"]=50.0
        return df.to_dict("records")

    df["atr_pct"]=pd.to_numeric(df["atr_pct"], errors="coerce")
    df["dollar_volume"]=pd.to_numeric(df["dollar_volume"], errors="coerce").fillna(0.0)
    df["dv_log"]=np.log10(df["dollar_volume"].clip(lower=0)+1.0)

    z_atr=(df["atr_pct"]-df["atr_pct"].mean())/(df["atr_pct"].std()+1e-9)
    z_dv =(df["dv_log"] -df["dv_log"].mean()) /(df["dv_log"].std() +1e-9)
    risk_index = z_atr - z_dv

    rmin, rmax = np.nanmin(risk_index), np.nanmax(risk_index)
    df["risk_score"] = 50.0 if rmax==rmin else 100.0*(risk_index - rmin)/(rmax - rmin)

    if len(df) >= 3:
        q1,q2 = np.nanquantile(risk_index,[0.33,0.66])
        df["bucket"]=np.where(risk_index<=q1,"Low", np.where(risk_index<=q2,"Medium","High"))
    else:
        df["bucket"]="High"

    # guardrails
    df.loc[(df["atr_pct"]>15) | (df["dollar_volume"]<100_000), "bucket"]="High"
    df.loc[(df["atr_pct"]<2.0) & (df["dollar_volume"]>20_000_000), "bucket"]="Low"

    return df.drop(columns=["dv_log"]).to_dict("records")

# =================== Forecast (statsmodels) ===================
def auto_arima_statsmodels(y: pd.Series, max_p=3, max_d=2, max_q=3) -> Tuple[Tuple[int,int,int], ARIMA]:
    best_aic=np.inf; best=None; best_model=None
    y=pd.Series(y).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    for d in range(max_d+1):
        for p in range(max_p+1):
            for q in range(max_q+1):
                if p==q==0 and d==0: continue
                try:
                    res=ARIMA(y, order=(p,d,q)).fit()
                    if res.aic < (best_aic if best_aic is not None else np.inf):
                        best_aic=res.aic; best=(p,d,q); best_model=res
                except: continue
    if best_model is None:
        best=(1,1,1); best_model=ARIMA(y, order=best).fit()
    return best, best_model

def forecast_series(df_xy: pd.DataFrame, periods:int, freq:str) -> pd.DataFrame:
    s=df_xy[["ds","y"]].dropna().sort_values("ds")
    y=s["y"].astype(float).values
    _, model = auto_arima_statsmodels(y)
    fc = model.get_forecast(steps=periods)
    conf = fc.conf_int(alpha=0.2)  # 80% band
    idx = pd.date_range(s["ds"].iloc[-1], periods=periods+1, freq=freq)[1:]
    return pd.DataFrame({"ds":idx,"yhat":fc.predicted_mean,"lo":conf.iloc[:,0].values,"hi":conf.iloc[:,1].values})

# =================== Tabs ===================
tab_overview, tab_live, tab_news, tab_predict = st.tabs(["Overview","Live","News","Predict"])

# cache-busters (respect TTLs)
price_key = int(time.time() // max(1, st.session_state["price_ttl"]))
news_key  = int(time.time() // max(1, st.session_state["news_ttl"]))

# ---------- Overview ----------
with tab_overview:
    st.markdown("## Overview")

    # Global sentiment
    news = latest_news(st.session_state["cp_token"], limit=80, _ttl_key=news_key)
    global_bias = compute_sentiment_bias(news)

    c1,c2 = st.columns([1,3])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Sentiment (0–100)", f"{to_0_100(global_bias)}")
        st.caption(f"Internal bias: {global_bias:+.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Headlines (sample)")
        for n in news[:6]:
            st.write(f"- [{n['title']}]({n['url']}) — {to_0_100(n['sentiment'])}/100")
        st.markdown("</div>", unsafe_allow_html=True)

    # Markets
    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, price_key, 40)
    except Exception as e:
        st.error(f"Markets failed: {e}"); syms=[]
    default_watch = [s for s in syms if any(x in s for x in ["BTC/","ETH/","SOL/"])][:5]
    watch = st.multiselect(f"Watchlist ({quote.upper()})", options=syms, default=default_watch)

    # Chart
    if watch:
        primary = watch[0]
        ind = compute_indicators(fetch_ohlcv_df(exchange_id, primary, "1h", 400, price_key))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind.index, y=ind["close"], mode="lines", name="Price"))
        if ind["sma20"].notna().any(): fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"], mode="lines", name="SMA20"))
        if ind["sma50"].notna().any(): fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], mode="lines", name="SMA50"))
        fig.update_layout(title=f"{primary} 1h", xaxis_title="Time", yaxis_title="Price", margin=dict(l=10,r=10,t=36,b=8))
        st.plotly_chart(fig, use_container_width=True)

    # Table rows (no bucket yet)
    rows=[]
    for sym in watch:
        try:
            dfi = compute_indicators(fetch_ohlcv_df(exchange_id, sym, "1h", 400, price_key))
            t = fetch_ticker(exchange_id, sym, price_key)
            last = float(t.get("last") or t.get("close") or dfi["close"].iloc[-1])
            dv = float((t.get("last") or 0) * (t.get("baseVolume") or 0))
            score = composite_score(dfi, global_bias)
            atr = float(dfi["atr"].iloc[-1]) if "atr" in dfi else np.nan
            rows.append({
                "Symbol": sym,
                "Score (−1..+1)": score,
                "Score (0–100)": to_0_100(score),
                "Last": last,
                "ATR": atr,
                "atr_pct": float(dfi["atr_pct"].iloc[-1]) if "atr_pct" in dfi else np.nan,
                "Dollar Volume": dv,
                "Entry": None, "TP": None, "SL": None
            })
        except Exception as e:
            st.warning(f"{sym}: {e}")

    # Adaptive risk + brackets
    if rows:
        rows = assign_risk(rows)
        for r in rows:
            bucket = r.get("bucket","High")
            if pd.notna(r.get("ATR")) and pd.notna(r.get("Last")):
                br = bracket_from_atr(r["Last"], r["ATR"], bucket)
                r["Entry"], r["TP"], r["SL"] = br.entry, br.tp, br.sl

        df = pd.DataFrame(rows)
        disp = df.copy()
        disp["Risk"] = df["bucket"].map(lambda b: f'<span class="chip {b}">{b}</span>')
        disp["Risk Score"] = df["risk_score"].map(lambda x: f"{x:.0f}")
        disp["Last"] = df["Last"].map(fmt)
        disp["ATR"] = df["ATR"].map(fmt)
        disp["ATR %"] = df["atr_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        disp["Entry"] = df["Entry"].map(fmt)
        disp["TP"] = df["TP"].map(fmt)
        disp["SL"] = df["SL"].map(fmt)
        def _colored(s: float) -> str:
            cls = "badge-pos" if s >= 0 else "badge-neg"
            return f'<span class="{cls}">{s:+.3f}</span>'
        disp["Score (−1..+1)"] = df["Score (−1..+1)"].map(_colored)
        disp["Dollar Volume"] = df["Dollar Volume"].map(usd)
        disp = disp[["Symbol","Risk","Risk Score","Score (0–100)","Score (−1..+1)","Last","ATR","ATR %","Entry","TP","SL","Dollar Volume"]]
        st.markdown(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Picks + optional Google Sheets log
        st.markdown("### Picks (top per bucket)")
        cols = st.columns(3)
        ws = _open_logs_sheet(st.session_state.get("gsheet_id",""))
        for i,b in enumerate(["Low","Medium","High"]):
            with cols[i]:
                sub = df[df["bucket"]==b]
                st.markdown(f"**{b} Risk**")
                if sub.empty:
                    st.caption("No candidate.")
                else:
                    r = sub.sort_values(["Score (0–100)","risk_score"], ascending=[False,True]).iloc[0]
                    alloc = {"Low":st.session_state["alloc_low"],"Medium":st.session_state["alloc_med"],"High":st.session_state["alloc_high"]}[b]
                    budget = st.session_state["capital"] * (alloc/100.0)
                    qty = 0.0 if r["Last"]<=0 else budget/r["Last"]
                    fees = budget * (st.session_state["fee_pct"]/100.0)
                    st.write(f"{r['Symbol']} — Score {int(r['Score (0–100)'])}/100, Risk {int(r['risk_score'])}/100")
                    st.write(f"Entry {fmt(r['Entry'])} • TP {fmt(r['TP'])} • SL {fmt(r['SL'])}")
                    st.caption(f"Budget {usd(budget)} • Qty ~ {fmt(qty)} • Est fees {usd(fees)}")
                    _append_log(ws, {
                        "ts_utc": pd.Timestamp.utcnow().isoformat(),
                        "kind": "pick",
                        "symbol": r["Symbol"], "timeframe": "1h", "last_price": r["Last"],
                        "horizon_steps": "", "horizon_tf": "", "pred_end_price": "", "pred_change_pct": "",
                        "sentiment_bias": global_bias, "risk_bucket": b,
                        "entry": r["Entry"], "tp": r["TP"], "sl": r["SL"],
                        "due_ts_utc":"", "status":"", "actual_price":"", "realized_change_pct":"", "correct":""
                    })

        if st.button("Reconcile results (forecasts)"):
            ws = _open_logs_sheet(st.session_state.get("gsheet_id",""))
            _reconcile_logs(ws, exchange_id)
            st.success("Reconciled (where due).")

# ---------- Live ----------
with tab_live:
    st.markdown("## Live")
    if rate_sec:
        st.markdown(f"<script>setTimeout(()=>window.location.reload(), {rate_sec*1000});</script>", unsafe_allow_html=True)

    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, price_key, 40)
    except Exception as e:
        st.error(f"Markets failed: {e}"); syms=[]
    default_watch = [s for s in syms if any(x in s for x in ["BTC/","ETH/","SOL/"])][:5]
    watch_live = st.multiselect("Watchlist", options=syms, default=default_watch, key="live_watch")

    rows=[]
    for sym in watch_live:
        try:
            t = fetch_ticker(exchange_id, sym, price_key)
            rows.append({"Symbol":sym, "Last": float(t.get("last") or t.get("close") or 0), "24h %": t.get("percentage")})
        except Exception as e: st.warning(f"{sym}: {e}")
    if rows:
        df=pd.DataFrame(rows); df["Last"]=df["Last"].map(fmt)
        st.dataframe(df, use_container_width=True)

# ---------- News ----------
with tab_news:
    st.markdown("## News")
    data = latest_news(st.session_state["cp_token"], limit=100, _ttl_key=news_key)
    bias_here = compute_sentiment_bias(data)
    st.metric("Global Sentiment (0–100)", f"{to_0_100(bias_here)}")
    st.metric("Bias (internal)", f"{bias_here:+.3f}")
    for n in data:
        st.write(f"- [{n['title']}]({n['url']}) — {to_0_100(n['sentiment'])}/100")

# ---------- Predict (ARIMA + sentiment) ----------
COIN_ALIASES = {
    "BTC":["btc","bitcoin"], "ETH":["eth","ethereum"], "SOL":["sol","solana"],
    "XRP":["xrp","ripple"], "ADA":["ada","cardano"], "DOGE":["doge","dogecoin"]
}
def coin_bias(symbol: str, news_items: List[Dict]) -> float:
    base=symbol.split("/")[0].upper()
    keys=COIN_ALIASES.get(base,[base.lower()])
    pat=re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.I)
    filt=[n for n in news_items if pat.search(n["title"])]
    return compute_sentiment_bias(filt)

with tab_predict:
    st.markdown("## Predict")
    st.caption("ARIMA forecast plus a small shift from coin‑specific sentiment. Sentiment shows as 0–100; math uses −1..+1.")

    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, price_key, 40)
    except Exception as e:
        st.error(f"Load markets failed: {e}"); syms=[]
    left,right = st.columns([2,1])
    with left:
        sym = st.selectbox("Symbol", options=syms)
        tf  = st.selectbox("Timeframe", ["15m","1h","4h"], index=1)
    with right:
        st.caption("Tune in sidebar: sentiment impact & bias floor")

    if sym:
        raw = fetch_ohlcv_df(exchange_id, sym, timeframe=tf, limit=600, _ttl_key=price_key).copy()
        hist = raw[["close"]].reset_index().rename(columns={"dt":"ds","close":"y"})
        periods = {"15m":96,"1h":24,"4h":6}[tf]; freq=tf_to_freq(tf)

        base_fc = forecast_series(hist, periods, freq)

        news = latest_news(st.session_state["cp_token"], limit=100, _ttl_key=news_key)
        b_coin = coin_bias(sym, news)
        b_disp = to_0_100(b_coin)
        w = float(st.session_state["sent_w"]); floor = float(st.session_state["bias_floor"])
        adj = 1.0 if abs(b_coin) < floor else (1.0 + float(np.tanh(b_coin))*w)

        fc = base_fc.copy()
        fc["yhat"]=base_fc["yhat"]*adj
        fc["hi"]  =base_fc["hi"]  *adj
        fc["lo"]  =base_fc["lo"]  *adj

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Predicted"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["hi"],   mode="lines", name="Upper", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["lo"],   mode="lines", name="Lower", line=dict(dash="dot"), fill="tonexty"))
        fig.update_layout(title=f"{sym} {tf} — ARIMA + sentiment", xaxis_title="Time", yaxis_title="Price",
                          margin=dict(l=10,r=10,t=36,b=8))
        st.plotly_chart(fig, use_container_width=True)

        last=float(hist["y"].iloc[-1]); end=float(fc["yhat"].iloc[-1])
        change_pct = (end/last - 1.0) * 100.0
        st.metric("Coin sentiment (0–100)", f"{b_disp}")
        st.metric("Predicted change", f"{change_pct:+.2f}%")

        ws = _open_logs_sheet(st.session_state.get("gsheet_id",""))
        due_ts = (hist["ds"].iloc[-1] + pd.to_timedelta(periods*tf_to_minutes(tf), unit="m")).isoformat()
        _append_log(ws, {
            "ts_utc": pd.Timestamp.utcnow().isoformat(),
            "kind": "forecast",
            "symbol": sym, "timeframe": tf, "last_price": last,
            "horizon_steps": periods, "horizon_tf": tf,
            "pred_end_price": end, "pred_change_pct": change_pct,
            "sentiment_bias": b_coin, "risk_bucket": "",
            "entry": "", "tp": "", "sl": "",
            "due_ts_utc": due_ts, "status":"", "actual_price":"", "realized_change_pct":"", "correct":""
        })

        if st.button("Reconcile now", key="recon2"):
            _reconcile_logs(ws, exchange_id)
            st.success("Reconciled (where due).")
