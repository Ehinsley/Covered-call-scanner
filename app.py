"""Phone-first covered call screener; hosted backend protects brokerage API credentials.
Run locally: streamlit run app.py
"""
from __future__ import annotations
from datetime import datetime, date
from zoneinfo import ZoneInfo
import math
import os
import re
import pandas as pd
import streamlit as st
from market_data import MarketDataError, tradier_snapshot, yahoo_snapshot, ET
from trade_math import assess, tail_above

st.set_page_config(page_title="Live Covered Call Scanner", page_icon="📱", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container {max-width:820px; padding:1rem 1rem 4rem}
h1 {font-size:1.75rem!important;line-height:1.18!important;letter-spacing:-.025em}
.stButton button,.stFormSubmitButton button,.stDownloadButton button {min-height:51px;border-radius:13px;font-weight:650}
div[data-testid="stMetric"] {padding:14px;border-radius:13px;border:1px solid rgba(128,140,150,.24)}
div[data-testid="stMetricLabel"] {font-size:.83rem}
small {font-size:.82rem}
@media(max-width:600px){ .block-container {padding:.6rem .85rem 3rem} h1{font-size:1.7rem!important} div[data-testid="stMetricValue"]{font-size:1.45rem!important}}
</style>""", unsafe_allow_html=True)

RISKY = {"HIMZ", "MSTX", "MSTU", "MSTZ", "AMDL", "NVDL", "TSLL", "CONL", "TQQQ", "SOXL", "UPRO", "SPXL", "BITX", "ETHU"}


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.environ.get(name, default)))
    except Exception:
        return str(os.environ.get(name, default))


def money(x):
    return f"${x:,.2f}"


def p100(x):
    return f"{x:.1%}" if x is not None else "N/A"


def quote_age(time_text):
    if not time_text:
        return "Quote timestamp unavailable"
    try:
        dt = datetime.fromisoformat(str(time_text))
        if dt.tzinfo is None:
            return str(time_text)
        age = max(0, (datetime.now(ET) - dt).total_seconds() / 60)
        if age > 1440:
            return f"{str(time_text).replace('T',' ')} · {age/1440:.1f} days old"
        return f"{str(time_text).replace('T',' ')} · {age:.0f} min old"
    except (ValueError, TypeError):
        return str(time_text)


@st.cache_data(ttl=45, show_spinner=False)
def fetch_market(sym: str, lo: int, hi: int, provider: str, token: str):
    if provider == "Tradier live":
        return tradier_snapshot(sym, lo, hi, token, sandbox=False)
    if provider == "Tradier sandbox":
        return tradier_snapshot(sym, lo, hi, token, sandbox=True)
    return yahoo_snapshot(sym, lo, hi)


app_pw = get_secret("APP_PASSWORD")
if app_pw:
    if not st.session_state.get("auth", False):
        st.title("🔐 Covered Call Scanner")
        with st.form("login"):
            password = st.text_input("App passcode", type="password")
            enter = st.form_submit_button("Unlock scanner", use_container_width=True)
        if enter:
            import hmac
            if hmac.compare_digest(password, app_pw):
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect passcode")
        st.stop()

st.title("📱 Covered Call Scanner")
st.caption("Automatic stock + options-chain lookup · Phone-friendly · Read-only")
token = get_secret("TRADIER_TOKEN")
mode = get_secret("TRADIER_MODE", "production").lower()
if token:
    provider = "Tradier sandbox" if mode == "sandbox" else "Tradier live"
else:
    provider = "Yahoo delayed"

if provider == "Tradier live":
    st.success("**Data feed:** Tradier production brokerage API (real-time for eligible accounts).")
elif provider == "Tradier sandbox":
    st.warning("**Data feed:** Tradier sandbox — delayed, NOT live quotes.")
else:
    st.warning("**Data feed:** Yahoo Finance fallback — reported prices may be delayed or stale. Configure a Tradier production token on the server for live brokerage quotes.")

with st.form("scan_form"):
    ticker = st.text_input("Ticker", "HIMZ", max_chars=12, help="US stock/ETF ticker").strip().upper()
    c1,c2 = st.columns(2)
    budget = c1.number_input("Max new share purchase ($)", min_value=100., value=1000., step=100., format="%.0f")
    account = c2.number_input("Account size ($)", min_value=100., value=4300., step=100., format="%.0f")
    c1,c2 = st.columns(2)
    low_dte = c1.number_input("Min days to expiry", min_value=1, max_value=120, value=21)
    high_dte = c2.number_input("Max days to expiry", min_value=1, max_value=180, value=45)
    with st.expander("Filters & model", expanded=False):
        strike_min = st.slider("Minimum strike / stock price", .5, 1.25, .85, step=.05)
        strike_max = st.slider("Maximum strike / stock price", .75, 2.0, 1.20, step=.05)
        show_over = st.checkbox("Show over-budget contracts for research", value=False)
        drift = st.slider("Assumed annual stock drift (not forecast)", -50, 50, 0, step=5) / 100
        fees = st.number_input("Fees per 100-share package ($)", min_value=0., value=0., step=.50, format="%.2f")
    go = st.form_submit_button("🔎 Fetch prices & scan calls", type="primary", use_container_width=True)

# Persist successful scans across UI interactions, but do not auto-poll and hit rate limits.
if go:
    if not re.fullmatch(r"[A-Z][A-Z0-9./-]{0,11}", ticker):
        st.error("Enter a valid ticker.")
    elif low_dte > high_dte or strike_min > strike_max:
        st.error("Check the expiration and strike filters.")
    else:
        with st.spinner("Fetching stock quote, listed expirations, and option bids/asks…"):
            try:
                st.session_state.market = fetch_market(ticker, int(low_dte), int(high_dte), provider, token)
                st.session_state.params = dict(ticker=ticker, low=int(low_dte), high=int(high_dte), budget=budget, account=account,
                    strike_min=strike_min, strike_max=strike_max, show_over=show_over, drift=drift, fees=fees)
            except MarketDataError as exc:
                st.session_state.pop("market", None)
                st.error(str(exc))
            except Exception as exc:
                st.session_state.pop("market", None)
                st.error(f"Data provider unavailable ({type(exc).__name__}). Try again later.")

market = st.session_state.get("market")
params = st.session_state.get("params")
if market and params:
    sym = market["ticker"]
    spot = market["spot"]
    costs = spot * 100
    with st.container(border=True):
        st.caption(f"{sym} · {market['provider']}")
        a,b = st.columns(2)
        a.metric("Share purchase estimate", money(spot))
        b.metric("100 shares", money(costs))
        st.caption("Purchase basis: " + market["stock_source"] + " · Quote: " + quote_age(market.get("stock_quote_time")))
        st.caption("Data retrieved: " + market["fetched_at"].replace("T", " ") + " ET · Press scan again to refresh; cached up to 45 seconds.")
    if costs > params["budget"]:
        st.error(f"**Exceeds your ${params['budget']:,.0f} gross share-purchase budget.** Buying 100 shares costs about {money(costs)}. An option credit does not reduce the upfront cash required to buy those shares.")
    if costs > params["account"] * .30:
        st.warning(f"One 100-share position represents approximately {costs/params['account']:.0%} of your ${params['account']:,.0f} account.")
    leveraged = sym in RISKY
    if leveraged:
        st.warning("**Daily leveraged ETF:** the underlying resets leverage daily. A simple price distribution cannot fully model path-dependent compounding and tail losses.")
    if market["delayed"]:
        st.warning("**Prices are not confirmed live.** Verify current stock ask, option bid, quote age, and deliverable in your brokerage app.")

    qualified=[]
    for op in market["rows"]:
        dte=(date.fromisoformat(op["expiration"]) - datetime.now(ET).date()).days
        if not (params["low"] <= dte <= params["high"]):
            continue
        if not (params["strike_min"] * spot <= op["strike"] <= params["strike_max"] * spot):
            continue
        if costs > params["budget"] and not params["show_over"]:
            continue
        iv=op["iv"]
        iv_source="Option IV (provider)"
        if iv is None:
            iv=market.get("rv")
            iv_source="60-day realized volatility fallback" if iv is not None else "Volatility unavailable"
        # Quote math is always provided when quotes pass basic validation, even if IV is missing.
        call_credit=op["bid"]*100
        maxprofit=(op["strike"]-spot+op["bid"])*100-params["fees"]
        breakeven=spot-op["bid"]+params["fees"]/100
        spread=(op["ask"]-op["bid"])/((op["ask"]+op["bid"])*.5)
        row=dict(**op, dte=dte, credit=call_credit, maxprofit=maxprofit, breakeven=breakeven,
            spread=spread, netcredit=call_credit-params["fees"], sigma=iv, iv_source=iv_source,
            position_cost=costs, model=None, warnings=[])
        if iv is not None and 0.03 <= iv <= 8.0:
            try:
                # Assess models premiums at bid, share purchase at stock ask when available.
                model=assess(spot,op["strike"],op["bid"],op["ask"],dte,iv,params["budget"],op["oi"],
                    drift=params["drift"],leveraged=leveraged)
                if params["fees"] > 0:
                    if maxprofit <= 0 or breakeven >= op["strike"]:
                        model["p_profit"] = 0.0
                    elif breakeven <= 0:
                        model["p_profit"] = 1.0
                    else:
                        model["p_profit"] = tail_above(spot,breakeven,iv,dte,params["drift"])
                row["model"]=model
                row["warnings"].extend(model["warnings"])
            except ValueError:
                pass
        if iv is None:
            row["warnings"].append("No volatility available; probability not calculated")
        if op["oi"] < 25:
            row["warnings"].append("Low open interest")
        if spread > 0.30:
            row["warnings"].append("Wide bid/ask spread")
        if maxprofit <= 0:
            row["warnings"].append("Negative maximum profit if called away")
        qualified.append(row)

    qualified.sort(key=lambda r: (
        r["maxprofit"] > 0,
        (r["model"] or {}).get("verdict") not in {"PASS / EXCLUDE", None},
        (r["model"] or {}).get("score", 0),
        r["oi"],
        ), reverse=True)
    st.subheader("Matching covered calls")
    if not qualified:
        st.info("No contracts match these filters, or usable option bids/asks were unavailable. If the stock is over budget, turn on **Show over-budget contracts for research**, or choose a stock below $10.")
    else:
        st.caption("Call credit uses **option bid**; share cost uses **stock ask** when available. A limit order may fill differently. Scoring is a heuristic, not a true probability of a good investment.")
        for row in qualified[:20]:
            model=row["model"]
            with st.container(border=True):
                st.markdown(f"**{row['expiration']} · ${row['strike']:g} call** · {row['dte']} DTE")
                a,b=st.columns(2)
                a.metric("Sell-call bid",money(row["bid"]))
                b.metric("Premium collected",money(row["credit"]))
                a,b=st.columns(2)
                a.metric("Effective cost/share",money(row["breakeven"]))
                b.metric("Max profit if assigned",money(row["maxprofit"]))
                if model:
                    a,b=st.columns(2)
                    a.metric("Modeled P(profit)", p100(model["p_profit"]))
                    b.metric("Trade score (not odds)",f"{model['score']}/100")
                else:
                    st.caption("Probability not available without a valid volatility estimate.")
                with st.expander("Bid/ask, liquidity & risk details"):
                    st.write(f"**Contract:** `{row['contract']}` · 100-share standard assumed; confirm deliverable in your broker.")
                    st.write(f"**Option bid / ask:** {money(row['bid'])} / {money(row['ask'])} · Spread {p100(row['spread'])}")
                    st.write(f"**Bid time:** {quote_age(row['bid_time'])}")
                    st.write(f"**Ask time:** {quote_age(row['ask_time'])}")
                    st.write(f"**Open interest:** {row['oi']:,} · **Option volume:** {row['volume']:,}")
                    st.write(f"**Volatility:** {p100(row['sigma'])} · {row['iv_source']} (IV may be updated hourly)")
                    if model:
                        st.write(f"**Modeled P(above strike):** {p100(model['p_above'])} · **Modeled P(down 25%):** {p100(model['p_drop25'])}")
                        st.write(f"**Max loss if share price goes to zero:** {money(costs-row['credit']+params['fees'])}")
                    st.write("**Event risks and dividend/early-assignment risk not auto-checked.**")
                    for msg in dict.fromkeys(row["warnings"]):
                        st.warning(msg)

        records=[]
        for x in qualified:
            mod=x["model"] or {}
            records.append({"Ticker":sym,"Expiration":x["expiration"],"DTE":x["dte"],"Strike":x["strike"],"Stock entry":spot,
                "Call bid":x["bid"],"Call ask":x["ask"],"Premium":x["credit"],"Effective cost":x["breakeven"],
                "Max profit if assigned":x["maxprofit"],"P profit model":mod.get("p_profit"),"P ITM model":mod.get("p_above"),
                "Score (NOT probability)":mod.get("score"),"IV":x["sigma"],"IV source":x["iv_source"],
                "Open interest":x["oi"],"Volume":x["volume"],"Option contract":x["contract"],"Bid quote time":x["bid_time"],
                "Provider":market["provider"],"Warnings":"; ".join(dict.fromkeys(x["warnings"]))})
        st.download_button("Download results (CSV)",pd.DataFrame(records).to_csv(index=False).encode("utf-8"),f"{sym}_covered_call_scan.csv",mime="text/csv",use_container_width=True)

st.divider()
with st.expander("How this works · what the probability means"):
    st.write("**Trade payoff at expiration:** 100 × [min(stock at expiration, strike) − stock purchase price + call premium] − fees. Shares can still lose most of their value. Early assignment, dividends, transaction taxes, and path-specific risks are not captured.")
    st.write("**P(profit) model:** Probability expiration stock price exceeds the net break-even under a lognormal stock-price model. Annual stock drift defaults to 0%; volatility is provider option IV or 60-day historical realized volatility. It is not a validated win rate.")
    st.write("**Score:** A 0–100 ranking combining probability estimate, capped return, option liquidity and downside scenario. It is *not* a probability, and should not be used as a buy signal.")
    st.write("**Quote source:** Real-time Tradier requires an eligible production Tradier Brokerage account/token. Tradier sandbox is ~15 min delayed; Yahoo quotes may be older, especially options. Providers can rate-limit or fail. Verify in Webull before trading.")
    st.write("**Credentials:** A Tradier token belongs in server-side Streamlit secrets, never inside public HTML, shared ZIP, GitHub, or chat. This app makes only read-only market-data calls and never places orders.")
st.caption("Research tool only · No brokerage order entry · Don't treat modeled probabilities as guarantees. Confirm earnings, corporate actions, contract multiplier, bid/ask spread, liquidity and quote time before trading.")
