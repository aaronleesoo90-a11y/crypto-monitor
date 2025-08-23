# app.py — Crypto Monitoring System (polished UI + sentiment 0–100 + Predict tab)
# Uses statsmodels ARIMA (no Prophet/pmdarima). Mobile/cloud friendly.

from __future__ import annotations
import warnings, re, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

import ccxt, requests, feedparser, ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ----------------------- Page / Theme -----------------------
st.set_page_config(
    page_title="Crypto Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS polish
st.markdown("""
<style>
/* tighter layout */
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
/* section headers */
h1, h2, h3 { margin-top: .6rem; }
/* card look */
.card {
  border-radius: 16px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,.06);
  border: 1px solid rgba(0,0,0,.05); background: #ffffff;
}
/* chip */
.chip {display:inline-block; padding:4px 10px; border-radius:999px; font-size:.85rem; font-weight:600;}
.chip.low{background:#e9f9ee; color:#127c38;}
.chip.medium{background:#fff7e6; color:#a15b00;}
.chip.high{background:#ffefef; color:#a11a1a;}
/* table zebra + compact */
.dataframe tbody tr:nth-child(odd) { background-color: rgba(0,0,0,.02); }
.dataframe td, .dataframe th { padding: 6px 10px; font-size: 14px; }
.smallnote {color:#6b7280; font-size:.85rem;}
.badge-green{color:#0b7a34; font-weight:700;}
.badge-red{color:#b00020; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ----------------------- Utilities -----------------------
analyzer = SentimentIntensityAnalyzer()

@dataclass
class Bracket:
    entry: float; tp: float; sl: float

def fmt_num(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "-"
        return f"{x:.6f}" if x < 1000 else f"{x:.2f}"
    except: return "-"

def usd(x: float) -> str:
    try:
        if x >= 1e9: return f"${x/1e9:.2f}B"
        if x >= 1e6: return f"${x/1e6:.2f}M"
        if x >= 1e3: return f"${x/1e3:.2f}k"
        return f"${x:.2f}"
    except: return "-"

def tf_to_pandas_freq(tf: str) -> str:
    return {"15m":"15min","1h":"H","4h":"4H"}.get(tf.lower().strip(), tf)

def score_to_color(score: float) -> str:
    # for tables: positive green, negative red
    return "badge-green" if score >= 0 else "badge-red"

def risk_chip(bucket: str) -> str:
    b = bucket.lower()
    cls = "low" if b == "low" else "medium" if b == "medium" else "high"
    label = bucket.title()
    return f'<span class="chip {cls}">{label}</span>'

def to_0_100(signed: float) -> int:
    # Rescale −1..+1 to 0..100 for display (does not change internal math)
    v = int(round((float(signed) + 1.0) * 50))
    return max(0, min(100, v))

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.title("Settings")
    exchange_id = st.text_input("Exchange id (ccxt)", value=st.session_state.get("exchange_id", "coinbase"))
    quote = st.text_input("Quote currency", value=st.session_state.get("quote", "USDC"))
    st.caption("Examples: coinbase or binance. Quote: USDC or USDT.")

    st.subheader("News API (optional)")
    cp_token = st.text_input("CryptoPanic token", value=st.session_state.get("cp_token",""), type="password")

    st.subheader("Refresh")
    price_ttl = st.number_input("Price cache ttl (sec)", 5, value=int(st.session_state.get("price_ttl",30)), step=5)
    news_ttl  = st.number_input("News cache ttl (sec)", 15, value=int(st.session_state.get("news_ttl",120)), step=5)
    if st.button("Refresh now"):
        st.cache_data.clear(); st.success("Refreshed caches")

    st.subheader("Trading plan")
    total_capital = st.number_input("Total capital", 0.0, value=float(st.session_state.get("capital",100.0)), step=10.0)
    fee_pct = st.number_input("Round-trip fee %", 0.0, value=float(st.session_state.get("fee_pct",0.10)), step=0.05)
    alloc_low  = st.slider("Low %", 0,100, int(st.session_state.get("alloc_low",40)))
    alloc_med  = st.slider("Med %", 0,100, int(st.session_state.get("alloc_med",40)))
    alloc_high = st.slider("High %",0,100, int(st.session_state.get("alloc_high",20)))
    if alloc_low+alloc_med+alloc_high != 100:
        st.warning("Allocations should sum to 100%.")

st.session_state.update({
    "exchange_id": exchange_id, "quote": quote,
    "cp_token": cp_token, "price_ttl": price_ttl, "news_ttl": news_ttl,
    "capital": total_capital, "fee_pct": fee_pct,
    "alloc_low": alloc_low, "alloc_med": alloc_med, "alloc_high": alloc_high
})

# ----------------------- Data (cached) -----------------------
@st.cache_data(ttl=120, show_spinner=False)
def make_exchange(ex_id: str) -> ccxt.Exchange:
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    ex.load_markets(); return ex

@st.cache_data(ttl=180, show_spinner=False)
def top_symbols_by_dollar_volume(ex_id: str, quote: str, top_n: int=40) -> List[str]:
    ex = make_exchange(ex_id)
    syms = [s for s,m in ex.markets.items() if m.get("spot") and m.get("active") and m.get("quote","").upper()==quote.upper()]
    if not syms: return []
    tickers = ex.fetch_tickers(syms)
    rows=[]
    for sym,t in tickers.items():
        last = t.get("last") or t.get("close") or 0
        base_vol = t.get("baseVolume") or 0
        rows.append((sym, (last or 0)*(base_vol or 0)))
    rows.sort(key=lambda x:x[1], reverse=True)
    return [s for s,_ in rows[:top_n]]

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv_df(ex_id: str, symbol: str, timeframe: str="1h", limit: int=400) -> pd.DataFrame:
    ex = make_exchange(ex_id)
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_ticker(ex_id: str, symbol: str) -> Dict:
    return make_exchange(ex_id).fetch_ticker(symbol)

@st.cache_data(ttl=120, show_spinner=False)
def latest_news(cp_token: Optional[str], limit: int=100) -> List[Dict]:
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
            for e in fp.entries[:40]:
                title=e.get("title",""); link=e.get("link",""); ts=e.get("published","") or e.get("updated","")
                vs=analyzer.polarity_scores(title)["compound"]
                out.append({"source":src,"title":title,"url":link,"ts":ts,"sentiment":vs})
        except: continue
    return out[:limit]

# ----------------------- Analytics -----------------------
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

def composite_score(dfi: pd.DataFrame, news_bias: float=0.0) -> float:
    if dfi.empty or "sma20" not in dfi.columns: return 0.0
    r=dfi.iloc[-1]; s=0.0
    s += 0.30 if r["sma20"]>r["sma50"] else -0.30
    s += 0.20 if r["close"]>r["sma20"] else -0.05
    s += 0.20*np.tanh(r["volume"]/max(1e-9, dfi["volume"].tail(20).mean()) - 1.0)
    s += 0.10*np.tanh((r.get("rsi14",50)-50)/15)
    s += 0.20*np.tanh(news_bias)
    return float(np.clip(s,-1.0,1.0))

def risk_bucket(d: pd.DataFrame, dv: float) -> str:
    try: atrp=float(d["atr_pct"].iloc[-1])
    except: return "High"
    if atrp<3.5 and dv>5_000_000: return "Low"
    if atrp<7.5 and dv>1_500_000: return "Medium"
    return "High"

def bracket_from_atr(entry: float, atr: float, bucket: str) -> Bracket:
    k={"Low":(1.2,0.8),"Medium":(1.8,1.0)}.get(bucket,(2.5,1.2))
    return Bracket(entry, entry + k[0]*atr, entry - k[1]*atr)

def compute_sentiment_bias(items: List[Dict]) -> float:
    if not items: return 0.0
    num=den=0.0
    for n in items:
        w=max(0.1,abs(n["sentiment"])); num+=w*n["sentiment"]; den+=w
    return num/den if den else 0.0

def size_position(capital: float, alloc_pct: float, price: float, fee_pct: float) -> Dict[str,float]:
    budget = capital * (alloc_pct/100.0)
    qty = 0.0 if price <= 0 else budget/price
    fees = budget * (fee_pct/100.0)
    return {"budget":budget, "qty":qty, "est_fees":fees}

# ---------- Coin-specific sentiment ----------
COIN_ALIASES = {
    "BTC":["btc","bitcoin"],
    "ETH":["eth","ethereum"],
    "SOL":["sol","solana"],
    "XRP":["xrp","ripple"],
    "ADA":["ada","cardano"],
    "DOGE":["doge","dogecoin"],
}

def coin_bias(symbol: str, news_items: List[Dict]) -> float:
    base = symbol.split("/")[0].upper()
    keys = COIN_ALIASES.get(base, [base.lower()])
    pat = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.I)
    filtered = [n for n in news_items if pat.search(n["title"])]
    return compute_sentiment_bias(filtered)

# ----------------------- Forecast helpers (statsmodels) -----------------------
def auto_arima_statsmodels(y: pd.Series, max_p=3, max_d=2, max_q=3) -> Tuple[Tuple[int,int,int], ARIMA]:
    best_aic=np.inf; best=None; best_model=None
    y=pd.Series(y).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    for d in range(max_d+1):
        for p in range(max_p+1):
            for q in range(max_q+1):
                if p==q==0 and d==0: continue
                try:
                    model=ARIMA(y, order=(p,d,q))
                    res=model.fit(method="statespace", disp=False)
                    if res.aic < (best_aic if best_aic is not None else np.inf):
                        best_aic=res.aic; best=(p,d,q); best_model=res
                except: continue
    if best_model is None:
        best=(1,1,1); best_model=ARIMA(y, order=best).fit(method="statespace", disp=False)
    return best, best_model

def forecast_series(df_xy: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    s=df_xy[["ds","y"]].dropna().sort_values("ds")
    y=s["y"].astype(float).values
    _, model = auto_arima_statsmodels(y)
    fc = model.get_forecast(steps=periods)
    conf = fc.conf_int(alpha=0.2)  # 80% band
    idx = pd.date_range(s["ds"].iloc[-1], periods=periods+1, freq=freq)[1:]
    return pd.DataFrame({"ds":idx,"yhat":fc.predicted_mean,"lo":conf.iloc[:,0].values,"hi":conf.iloc[:,1].values})

# ----------------------- Tabs -----------------------
tab_dash, tab_live, tab_news, tab_forecast, tab_predict, tab_help = st.tabs(
    ["Dashboard","Live","News","Forecast","Predict","Help"]
)

# ---------- Dashboard ----------
with tab_dash:
    st.markdown("## Dashboard")
    news = latest_news(cp_token, limit=80)
    global_bias = compute_sentiment_bias(news)
    bias_display = to_0_100(global_bias)  # 0..100 for UI

    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Sentiment (0–100)", f"{bias_display}")
        st.caption(f"Internal bias: {global_bias:+.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Quick tips")
        st.write("• Try coinbase/USDC or binance/USDT")
        st.write("• Pull‑to‑refresh via sidebar")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Headlines")
        for n in news[:6]:
            st.write(f"- [{n['title']}]({n['url']})  ({to_0_100(n['sentiment'])}/100)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Markets
    try: syms = top_symbols_by_dollar_volume(exchange_id, quote, 40)
    except Exception as e: st.error(f"Failed to load markets: {e}"); syms=[]
    default_watch = [s for s in syms if any(x in s for x in ["BTC/","ETH/","SOL/"])][:5]
    watch = st.multiselect(f"Watchlist ({quote.upper()})", options=syms, default=default_watch)

    # Chart
    if watch:
        primary = watch[0]
        ind = compute_indicators(fetch_ohlcv_df(exchange_id, primary, "1h", 400))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind.index, y=ind["close"], mode="lines", name="Price"))
        if ind["sma20"].notna().any(): fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"], mode="lines", name="SMA20"))
        if ind["sma50"].notna().any(): fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], mode="lines", name="SMA50"))
        fig.update_layout(title=f"{primary} 1h", xaxis_title="Time", yaxis_title="Price", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Table
    rows=[]
    for sym in watch:
        try:
            dfi = compute_indicators(fetch_ohlcv_df(exchange_id, sym, "1h", 400))
            t = fetch_ticker(exchange_id, sym)
            last = float(t.get("last") or t.get("close") or dfi["close"].iloc[-1])
            dv = float((t.get("last") or 0) * (t.get("baseVolume") or 0))
            score = composite_score(dfi, global_bias)
            bucket = risk_bucket(dfi, dv)
            atr = float(dfi["atr"].iloc[-1]) if "atr" in dfi else np.nan
            br = bracket_from_atr(last, atr, bucket)
            rows.append({
                "Symbol": sym,
                "Score (−1..+1)": score,
                "Score (0–100)": to_0_100(score),
                "Risk": bucket,
                "Last": last,
                "ATR": atr,
                "ATR %": float(dfi["atr_pct"].iloc[-1]) if "atr_pct" in dfi else np.nan,
                "Entry": br.entry, "TP": br.tp, "SL": br.sl,
                "Dollar Volume": dv
            })
        except Exception as e:
            st.warning(f"{sym}: {e}")

    if rows:
        df = pd.DataFrame(rows).sort_values(["Risk","Score (0–100)"], ascending=[True, False]).reset_index(drop=True)
        # Pretty HTML for risk + score color
        df_display = df.copy()
        df_display["Risk"] = df_display["Risk"].map(risk_chip)
        df_display["Score (−1..+1)"] = df_display["Score (−1..+1)"].map(lambda s: f'<span class="{score_to_color(s)}">{s:+.3f}</span>')
        df_display["Last"] = df_display["Last"].map(fmt_num)
        df_display["ATR"] = df_display["ATR"].map(fmt_num)
        df_display["ATR %"] = df_display["ATR %"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
        df_display["Entry"] = df_display["Entry"].map(fmt_num)
        df_display["TP"] = df_display["TP"].map(fmt_num)
        df_display["SL"] = df_display["SL"].map(fmt_num)
        df_display["Dollar Volume"] = df_display["Dollar Volume"].map(usd)

        st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.caption("Scores: color shows bullish/bearish tilt. Risk chip uses ATR% and dollar volume.")
    else:
        st.info("Pick symbols to analyze.")

# ---------- Live ----------
with tab_live:
    st.markdown("## Live")
    try: syms = top_symbols_by_dollar_volume(exchange_id, quote, 40)
    except Exception as e: st.error(f"Failed to load markets: {e}"); syms=[]
    default_watch = [s for s in syms if any(x in s for x in ["BTC/","ETH/","SOL/"])][:5]
    watch_live = st.multiselect("Watchlist", options=syms, default=default_watch, key="live")
    rows=[]
    for sym in watch_live:
        try:
            t = fetch_ticker(exchange_id, sym)
            rows.append({"Symbol":sym, "Last": float(t.get("last") or t.get("close") or 0), "24h %": t.get("percentage")})
        except Exception as e: st.warning(f"{sym}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df["Last"] = df["Last"].map(fmt_num)
        st.dataframe(df, use_container_width=True)

# ---------- News ----------
with tab_news:
    st.markdown("## News")
    data = latest_news(cp_token, limit=100)
    st.metric("Global Sentiment (0–100)", f"{to_0_100(compute_sentiment_bias(data))}")
    for n in data:
        st.write(f"- [{n['title']}]({n['url']}) — {to_0_100(n['sentiment'])}/100")

# ---------- Forecast (raw) ----------
with tab_forecast:
    st.markdown("## Forecast")
    try: syms = top_symbols_by_dollar_volume(exchange_id, quote, 40)
    except Exception as e: st.error(f"Load markets failed: {e}"); syms=[]
    sym = st.selectbox("Symbol", options=syms)
    tf  = st.selectbox("Timeframe", ["15m","1h","4h"], index=1)
    if sym:
        raw = fetch_ohlcv_df(exchange_id, sym, timeframe=tf, limit=600).copy()
        s = raw[["close"]].reset_index().rename(columns={"dt":"ds","close":"y"}).dropna()
        periods = {"15m":96, "1h":24, "4h":6}[tf]; freq = tf_to_pandas_freq(tf)
        fc = forecast_series(s, periods, freq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["ds"], y=s["y"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["hi"], mode="lines", name="Upper", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["lo"], mode="lines", name="Lower", line=dict(dash="dot"), fill="tonexty"))
        fig.update_layout(title=f"{sym} {tf} ARIMA", xaxis_title="Time", yaxis_title="Price", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- Predict (forecast + sentiment) ----------
with tab_predict:
    st.markdown("## Predict")
    st.caption("ARIMA forecast adjusted by coin‑specific news sentiment. Sentiment shown as 0–100; math uses −1..+1.")
    left, right = st.columns([2,1])
    with left:
        try: syms = top_symbols_by_dollar_volume(exchange_id, quote, 40)
        except Exception as e: st.error(f"Load markets failed: {e}"); syms=[]
        sym = st.selectbox("Symbol", options=syms, key="pred_sym")
        tf  = st.selectbox("Timeframe", ["15m","1h","4h"], index=1, key="pred_tf")
    with right:
        sentiment_weight = st.slider("Sentiment impact", 0.0, 0.05, 0.02, 0.005,
                                     help="Fractional shift for full bias (e.g., 0.02 ≈ ±2%).")
        bias_floor = st.slider("Ignore bias below", 0.0, 0.5, 0.10, 0.05)

    if sym:
        raw = fetch_ohlcv_df(exchange_id, sym, timeframe=tf, limit=600).copy()
        hist = raw[["close"]].reset_index().rename(columns={"dt":"ds","close":"y"}).dropna()

        # Base forecast
        periods={"15m":96,"1h":24,"4h":6}[tf]; freq=tf_to_pandas_freq(tf)
        base_fc = forecast_series(hist, periods, freq)

        # Sentiment
        news = latest_news(cp_token, limit=100)
        bias_coin = coin_bias(sym, news)  # −1..+1
        bias_display = to_0_100(bias_coin)  # for UI
        bias_adj = 0.0 if abs(bias_coin) < bias_floor else float(np.tanh(bias_coin)) * sentiment_weight

        # Adjust path
        adj_factor = (1.0 + bias_adj)
        fc_adj = base_fc.copy()
        fc_adj["yhat"] = base_fc["yhat"] * adj_factor
        fc_adj["hi"]   = base_fc["hi"]   * adj_factor
        fc_adj["lo"]   = base_fc["lo"]   * adj_factor

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=fc_adj["ds"], y=fc_adj["yhat"], mode="lines", name="Predicted"))
        fig.add_trace(go.Scatter(x=fc_adj["ds"], y=fc_adj["hi"], mode="lines", name="Upper", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=fc_adj["ds"], y=fc_adj["lo"], mode="lines", name="Lower", line=dict(dash="dot"), fill="tonexty"))
        fig.update_layout(title=f"{sym} {tf} Prediction (ARIMA + sentiment)", xaxis_title="Time", yaxis_title="Price",
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        last_price = float(hist["y"].iloc[-1])
        end_price  = float(fc_adj["yhat"].iloc[-1])
        updn = (end_price/last_price - 1.0) * 100.0

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Coin sentiment (0–100)", f"{bias_display}")
        with c2: st.metric("Applied shift", f"{bias_adj:+.2%}")
        with c3: st.metric("Predicted change", f"{updn:+.2f}%")

        with st.expander("Coin-related headlines used"):
            base = sym.split("/")[0].upper()
            keys = COIN_ALIASES.get(base, [base.lower()])
            st.write(f"Keywords: {', '.join(keys)}")
            for n in news:
                if any(re.search(rf'\\b{k}\\b', n['title'], re.I) for k in keys):
                    st.write(f"- [{n['title']}]({n['url']}) — {to_0_100(n['sentiment'])}/100")

# ---------- Help ----------
with tab_help:
    st.markdown("## Help")
    st.markdown("""
**Sentiment scale**
- Model outputs −1..+1 internally for calculations.
- UI shows 0–100 for readability: `0=very negative`, `50=neutral`, `100=very positive`.

**Scores**
- Composite score blends trend, momentum, volume surge, band expansion, RSI tilt, and global news bias.

**Risk**
- ATR% and dollar volume drive Low/Medium/High buckets.

**Predict**
- ARIMA forecast + a small multiplicative shift from coin-specific sentiment.
""")    return ex

@st.cache_data(ttl=180, show_spinner=False)
def top_symbols_by_dollar_volume(ex_id: str, quote: str, top_n: int = 40) -> List[str]:
    ex = make_exchange(ex_id)
    syms = [sym for sym, m in ex.markets.items()
            if m.get("spot") and m.get("active") and m.get("quote", "").upper() == quote.upper()]
    if not syms:
        return []
    tickers = ex.fetch_tickers(syms)
    rows = []
    for sym, t in tickers.items():
        last = t.get("last") or t.get("close") or 0
        base_vol = t.get("baseVolume") or 0
        dv = (last or 0) * (base_vol or 0)
        rows.append((sym, dv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv_df(ex_id: str, symbol: str, timeframe: str = "1h", limit: int = 400) -> pd.DataFrame:
    ex = make_exchange(ex_id)
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    return df

@st.cache_data(ttl=15, show_spinner=False)
def fetch_ticker(ex_id: str, symbol: str) -> Dict:
    ex = make_exchange(ex_id)
    return ex.fetch_ticker(symbol)

@st.cache_data(ttl=120, show_spinner=False)
def latest_news(cryptopanic_token: Optional[str], limit: int = 60) -> List[Dict]:
    out = []
    if cryptopanic_token:
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {"auth_token": cryptopanic_token, "public": "true", "kind": "news", "regions": "en", "page": 1}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            for it in r.json().get("results", [])[:limit]:
                title = it.get("title", "")
                link = it.get("url", "")
                ts = it.get("published_at", "")
                vs = analyzer.polarity_scores(title)["compound"]
                out.append({"source": "CryptoPanic", "title": title, "url": link, "ts": ts, "sentiment": vs})
        except Exception:
            pass
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
    ]
    for src in feeds:
        try:
            fp = feedparser.parse(src)
            for e in fp.entries[:20]:
                title = e.get("title", "")
                link = e.get("link", "")
                ts = e.get("published", "") or e.get("updated", "")
                vs = analyzer.polarity_scores(title)["compound"]
                out.append({"source": src, "title": title, "url": link, "ts": ts, "sentiment": vs})
        except Exception:
            continue
    out.sort(key=lambda x: (abs(x["sentiment"]), str(x["ts"])), reverse=True)
    return out[:limit]

# -----------------------
# Analytics
# -----------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if len(d) < 60:
        return d.assign(sma20=np.nan, sma50=np.nan, rsi14=np.nan, bb_high=np.nan, bb_low=np.nan, bb_bw=np.nan, atr=np.nan, atr_pct=np.nan)
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["rsi14"] = ta.momentum.RSIIndicator(d["close"], window=14).rsi()
    bb = ta.volatility.BollingerBands(close=d["close"], window=20, window_dev=2)
    d["bb_high"] = bb.bollinger_hband()
    d["bb_low"] = bb.bollinger_lband()
    d["bb_bw"] = (d["bb_high"] - d["bb_low"]) / d["sma20"]
    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=14)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = 100.0 * d["atr"] / d["close"]
    return d

def composite_score(dfi: pd.DataFrame, sentiment_score: float = 0.0) -> float:
    if dfi.empty or "sma20" not in dfi.columns:
        return 0.0
    row = dfi.iloc[-1]
    score = 0.0
    score += 0.30 if (row["sma20"] > row["sma50"]) else -0.30
    score += 0.20 if (row["close"] > row["sma20"]) else -0.05
    vol_ratio = row["volume"] / max(1e-9, dfi["volume"].tail(20).mean())
    score += 0.20 * np.tanh(vol_ratio - 1.0)
    bb_bw = row.get("bb_bw", np.nan)
    bw_avg = dfi["bb_bw"].tail(50).mean() if "bb_bw" in dfi else np.nan
    if pd.notna(bb_bw) and pd.notna(bw_avg) and bb_bw > bw_avg * 1.2 and row["close"] > row.get("bb_high", row["close"] + 1):
        score += 0.20
    score += 0.10 * np.tanh((row.get("rsi14", 50.0) - 50.0) / 15.0)
    score += 0.20 * np.tanh(sentiment_score)
    return float(np.clip(score, -1.0, 1.0))

def risk_bucket(dfi: pd.DataFrame, dollar_volume: float) -> str:
    try:
        atr_pct = float(dfi["atr_pct"].iloc[-1])
    except Exception:
        return "High"
    if atr_pct < 3.5 and dollar_volume > 5_000_000:
        return "Low"
    if atr_pct < 7.5 and dollar_volume > 1_500_000:
        return "Medium"
    return "High"

def bracket_from_atr(entry: float, atr: float, bucket: str) -> Bracket:
    if bucket == "Low":
        k_tp, k_sl = 1.2, 0.8
    elif bucket == "Medium":
        k_tp, k_sl = 1.8, 1.0
    else:
        k_tp, k_sl = 2.5, 1.2
    return Bracket(entry=entry, tp=entry + k_tp * atr, sl=entry - k_sl * atr)

def compute_sentiment_bias(news_items: List[Dict]) -> float:
    if not news_items:
        return 0.0
    num = den = 0.0
    for n in news_items[:100]:
        w = max(0.1, abs(n["sentiment"]))
        num += w * n["sentiment"]
        den += w
    return num / den if den else 0.0

def size_position(capital: float, alloc_pct: float, price: float, fee_pct: float) -> Dict[str, float]:
    budget = capital * (alloc_pct / 100.0)
    qty = 0.0 if price <= 0 else budget / price
    est_fees = budget * (fee_pct / 100.0)
    return {"budget": budget, "qty": qty, "est_fees": est_fees}

# -----------------------
# Simple ARIMA auto order search (statsmodels)
# -----------------------
def auto_arima_statsmodels(y: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[Tuple[int, int, int], ARIMA]:
    """
    Tiny grid search over (p,d,q) with AIC selection.
    Keeps runtime light for mobile by limiting bounds.
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    for d in range(0, max_d + 1):
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == q == 0 and d == 0:
                    continue
                try:
                    model = ARIMA(y, order=(p, d, q))
                    res = model.fit(method="statespace", disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_model = res
                except Exception:
                    continue
    if best_model is None:
        # Fallback
        best_model = ARIMA(y, order=(1, 1, 1)).fit(method="statespace", disp=False)
        best_order = (1, 1, 1)
    return best_order, best_model

def forecast_statsmodels(df_xy: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    s = df_xy[["ds", "y"]].dropna().sort_values("ds")
    y = s["y"].astype(float).values
    order, model = auto_arima_statsmodels(y)
    fc = model.get_forecast(steps=periods)
    conf = fc.conf_int(alpha=0.2)  # 80 percent band
    idx = pd.date_range(s["ds"].iloc[-1], periods=periods + 1, freq=freq)[1:]
    out = pd.DataFrame({
        "ds": idx,
        "yhat": fc.predicted_mean,
        "yhat_lower": conf.iloc[:, 0].values,
        "yhat_upper": conf.iloc[:, 1].values
    })
    return out

# -----------------------
# Layout: Tabs
# -----------------------
tab_dash, tab_live, tab_news, tab_forecast, tab_help = st.tabs(
    ["Dashboard", "Live Monitor", "News", "Forecast", "How it works"]
)

# ---------- Dashboard ----------
with tab_dash:
    st.title("Crypto Monitoring System")
    st.caption("Market view, signals, and ATR-based bracket guidance.")

    news_data = latest_news(cryptopanic_token if cryptopanic_token else None, limit=60)
    bias = compute_sentiment_bias(news_data)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Sentiment bias", f"{bias:+.3f}")
    with c2:
        st.write("Recent headlines:")
        for n in news_data[:8]:
            st.write(f"- [{n['title']}]({n['url']}) | sentiment {n['sentiment']:+.2f}")

    st.subheader("Market selection")
    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, top_n=40)
    except Exception as e:
        st.error(f"Failed to load markets: {e}")
        syms = []
    default_watch = [s for s in syms if any(x in s for x in ["BTC/", "ETH/", "SOL/"])][:5]
    watch = st.multiselect(f"Watchlist (quote {quote.upper()})", options=syms, default=default_watch)

    # Chart
    if watch:
        primary = watch[0]
        dfp = fetch_ohlcv_df(exchange_id, primary, "1h", 400)
        ind = compute_indicators(dfp)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind.index, y=ind["close"], mode="lines", name="Price"))
        if ind["sma20"].notna().any():
            fig.add_trace(go.Scatter(x=ind.index, y=ind["sma20"], mode="lines", name="SMA20"))
        if ind["sma50"].notna().any():
            fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], mode="lines", name="SMA50"))
        fig.update_layout(title=f"{primary} 1h chart", xaxis_title="Time", yaxis_title="Price", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Table
    rows = []
    for sym in watch:
        try:
            dfi = compute_indicators(fetch_ohlcv_df(exchange_id, sym, "1h", 400))
            score = composite_score(dfi, bias)
            t = fetch_ticker(exchange_id, sym)
            last = float(t.get("last") or t.get("close") or dfi["close"].iloc[-1])
            dv = float((t.get("last") or 0) * (t.get("baseVolume") or 0))
            bucket = risk_bucket(dfi, dollar_volume=dv)
            atr = float(dfi["atr"].iloc[-1])
            b = bracket_from_atr(entry=last, atr=atr, bucket=bucket)
            rows.append({
                "symbol": sym, "score": round(score, 3), "bucket": bucket,
                "price": last, "atr": atr, "atr_pct": float(dfi["atr_pct"].iloc[-1]),
                "entry": b.entry, "tp": b.tp, "sl": b.sl, "dollar_volume": dv
            })
        except Exception as e:
            st.warning(f"{sym}: {e}")

    if rows:
        dfrows = pd.DataFrame(rows).sort_values(by=["bucket", "score"], ascending=[True, False])
        st.dataframe(dfrows, use_container_width=True)
    else:
        st.info("Pick symbols to analyze.")

    # Picks
    st.subheader("Top 3 picks and 24h guide")
    if rows:
        dfrows = pd.DataFrame(rows)
        picks = {}
        for b in ["Low", "Medium", "High"]:
            sub = dfrows[dfrows["bucket"] == b]
            if not sub.empty:
                picks[b] = sub.sort_values("score", ascending=False).iloc[0]
        cols = st.columns(3)
        for i, b in enumerate(["Low", "Medium", "High"]):
            with cols[i]:
                if b in picks:
                    r = picks[b]
                    alloc_map = {"Low": alloc_low, "Medium": alloc_med, "High": alloc_high}
                    pos = size_position(total_capital, alloc_map[b], r["price"], fee_pct)
                    st.markdown(f"**{b} Risk**")
                    st.write(f"{r['symbol']} - score {r['score']:+.3f}")
                    st.write(f"Entry: {format_num(r['entry'])} | TP: {format_num(r['tp'])} | SL: {format_num(r['sl'])}")
                    st.write(f"Position: budget {usd(pos['budget'])}, qty ~ {format_num(pos['qty'])}, est fees {usd(pos['est_fees'])}")
                else:
                    st.write(f"**{b} Risk**")
                    st.caption("No candidate in this bucket right now.")

# ---------- Live Monitor ----------
with tab_live:
    st.title("Live Monitor")
    st.caption("Updates when cache TTL expires or when you press Refresh now in the sidebar.")
    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, top_n=40)
    except Exception as e:
        st.error(f"Failed to load markets: {e}")
        syms = []
    default_watch = [s for s in syms if any(x in s for x in ["BTC/", "ETH/", "SOL/"])][:5]
    watch_live = st.multiselect("Watchlist", options=syms, default=default_watch, key="watch_live")
    rows = []
    for sym in watch_live:
        try:
            t = fetch_ticker(exchange_id, sym)
            rows.append({"symbol": sym, "last": float(t.get("last") or t.get("close") or 0), "24h %": t.get("percentage")})
        except Exception as e:
            st.warning(f"{sym}: {e}")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- News ----------
with tab_news:
    st.title("News")
    data = list(latest_news(cryptopanic_token if cryptopanic_token else None, limit=60))
    bias_here = compute_sentiment_bias(data)
    st.metric("Sentiment bias", f"{bias_here:+.3f}")
    for n in data:
        st.write(f"- [{n['title']}]({n['url']}) - {n['sentiment']:+.2f}")

# ---------- Forecast ----------
with tab_forecast:
    st.title("Forecast")
    st.caption("Editable data, optional outlier cleaning, and ARIMA forecast with uncertainty bands.")

    try:
        syms = top_symbols_by_dollar_volume(exchange_id, quote, top_n=40)
    except Exception as e:
        st.error(f"Load markets failed: {e}")
        syms = []
    sym = st.selectbox("Symbol", options=syms)
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)

    if sym:
        raw = fetch_ohlcv_df(exchange_id, sym, timeframe=tf, limit=600).copy()
        working = raw[["close"]].rename(columns={"close": "price"}).reset_index().rename(columns={"datetime": "ds"})
        working["y"] = working["price"]

        st.subheader("Data editor")
        edited = st.data_editor(working[["ds", "y"]], num_rows="dynamic", use_container_width=True, key="editdf")
        if st.button("Apply edits"):
            st.session_state["edited_series"] = edited.copy()
            st.success("Applied.")
        series = st.session_state.get("edited_series", edited)

        # Simple auto clean
        s = series.sort_values("ds").reset_index(drop=True).copy()
        s["y"] = pd.to_numeric(s["y"], errors="coerce")
        s["ret"] = np.log(s["y"]).diff()
        z = (s["ret"] - s["ret"].mean()) / (s["ret"].std() + 1e-9)
        s.loc[z.abs() > 5, "y"] = np.nan
        s["y"] = s["y"].interpolate(limit_direction="both")
        s = s[["ds", "y"]].dropna()

        # Forecast
        try:
            periods = {"15m": 96, "1h": 24, "4h": 6}[tf]
            freq = tf_to_pandas_freq(tf)
            fc = forecast_statsmodels(s, periods=periods, freq=freq)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["ds"], y=s["y"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast"))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines", name="Upper", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines", name="Lower", line=dict(dash="dot"), fill="tonexty"))
            fig.update_layout(title=f"{sym} forecast (ARIMA)", xaxis_title="Time", yaxis_title="Price", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download cleaned series (CSV)",
                data=s.to_csv(index=False),
                file_name=f"{sym.replace('/', '_')}_{tf}_clean.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ---------- How it works ----------
with tab_help:
    st.title("How it works")
    st.markdown("""
**Inputs**
- Market data: candles via CCXT
- News: CryptoPanic (optional) and RSS feeds with sentiment

**Indicators**
- SMA20 and SMA50, RSI14, Bollinger Bandwidth, ATR and ATR%

**Composite score** (-1..+1)
- Trend, momentum, volume surge, band expansion, RSI tilt, and news bias

**Risk bucket**
- Low, Medium, High from ATR% and dollar volume

**Brackets**
- Entry = last, TP and SL use Entry ± k × ATR (k by bucket)

**Forecast**
- ARIMA from statsmodels with a small AIC grid search over (p,d,q)
""")
