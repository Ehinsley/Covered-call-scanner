"""Phone-first, read-only covered-call and cash-secured-put scanner.
Deploy app.py, market_data.py, trade_math.py and requirements.txt to Streamlit Cloud.
"""
from __future__ import annotations
from datetime import datetime, date
import hmac
import os
import re
import pandas as pd
import streamlit as st
from market_data import ET, MarketDataError, tradier_snapshot, yahoo_snapshot
from trade_math import assess, assess_put, tail_above

st.set_page_config(page_title="Options Income Scanner", page_icon="📱", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container { max-width:810px; padding: .8rem .95rem 4rem; }
h1 { font-size:1.8rem!important; letter-spacing:-.025em; }
.stButton button, .stFormSubmitButton button, .stDownloadButton button {min-height:54px; border-radius:12px; font-weight:650;}
div[data-testid="stMetric"] {padding:12px; border:1px solid rgba(128,140,150,.25); border-radius:13px;}
@media(max-width:600px){div[data-testid="stMetricValue"]{font-size:1.36rem!important}.block-container{padding:.55rem .75rem 3rem}}
</style>""", unsafe_allow_html=True)

RISKY = {"HIMZ", "MSTX", "MSTU", "MSTZ", "AMDL", "NVDL", "TSLL", "CONL", "TQQQ", "SOXL", "UPRO", "SPXL", "BITX", "ETHU"}


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.environ.get(name, default)))
    except Exception:
        return str(os.environ.get(name, default))


def money(v: float) -> str:
    return f"${v:,.2f}"


def percent(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "N/A"


def age_text(raw: str | None) -> str:
    if not raw:
        return "Timestamp unavailable"
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            return str(raw)
        mins = max(0., (datetime.now(ET)-dt).total_seconds()/60)
        return f"{dt.strftime('%b %d, %I:%M %p ET')} · {mins:.0f} min old" if mins < 1440 else f"{dt.strftime('%b %d, %Y %I:%M %p ET')} · {mins/1440:.1f} days old"
    except (ValueError, TypeError):
        return str(raw)


@st.cache_data(ttl=45, show_spinner=False)
def fetch(symbol: str, low: int, high: int, kind: str, provider: str, token: str):
    if provider == "Tradier live":
        return tradier_snapshot(symbol, low, high, token, sandbox=False, kind=kind)
    if provider == "Tradier sandbox":
        return tradier_snapshot(symbol, low, high, token, sandbox=True, kind=kind)
    return yahoo_snapshot(symbol, low, high, kind=kind)


password = secret("APP_PASSWORD")
if password and not st.session_state.get("logged_in", False):
    st.title("🔐 Options Income Scanner")
    with st.form("login"):
        supplied = st.text_input("Private app passcode", type="password")
        submit = st.form_submit_button("Unlock", use_container_width=True)
    if submit:
        if hmac.compare_digest(supplied, password):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Incorrect passcode")
    st.stop()

st.title("📱 Options Income Scanner")
st.caption("Covered calls + cash-secured puts · Mobile-friendly · Quotes, no trade execution")

kind_label = st.radio("Choose a strategy", ["Cash-secured put", "Covered call"], horizontal=True, index=0)
kind = "put" if kind_label == "Cash-secured put" else "call"
api_token = secret("TRADIER_TOKEN")
provider = "Tradier sandbox" if api_token and secret("TRADIER_MODE", "production").lower() == "sandbox" else (
    "Tradier live" if api_token else "Yahoo delayed")
if provider == "Tradier live":
    st.success("**Feed:** Tradier production (live only with proper brokerage market-data entitlement).")
elif provider == "Tradier sandbox":
    st.warning("**Feed:** Tradier sandbox (delayed).")
else:
    st.warning("**Feed:** Yahoo Finance automatic fallback. Stock/option quotes may be delayed, stale or missing. Live premiums require an eligible Tradier production feed.")

with st.form("scan"):
    ticker = st.text_input("Ticker symbol", value="MARA", max_chars=12).strip().upper()
    a,b = st.columns(2)
    budget = a.number_input("Max collateral / purchase ($)", min_value=100., value=1000., step=100., format="%.0f")
    account = b.number_input("Total account ($)", min_value=100., value=4300., step=100., format="%.0f")
    a,b = st.columns(2)
    low_dte = a.number_input("Minimum days to expiration", min_value=1, max_value=180, value=21)
    high_dte = b.number_input("Maximum days to expiration", min_value=1, max_value=365, value=45)
    with st.expander("Optional strike and probability settings"):
        low_strike = st.slider("Minimum strike ÷ stock price", .25, 1.5, .65 if kind == "put" else .85, step=.05)
        high_strike = st.slider("Maximum strike ÷ stock price", .5, 2.0, 1.10 if kind == "put" else 1.20, step=.05)
        show_over = st.checkbox("Show over-budget trades for research", value=False)
        drift = st.slider("Assumed annual stock drift (NOT a forecast)", -50, 50, 0, step=5)/100
        fees = st.number_input("Estimated fees / 1 contract ($)", min_value=0., value=0., step=.50)
    submitted = st.form_submit_button("🔎 Fetch prices & scan " + ("puts" if kind == "put" else "calls"), type="primary", use_container_width=True)

if submitted:
    if not re.fullmatch(r"[A-Z][A-Z0-9./-]{0,11}", ticker):
        st.error("Enter a valid ticker, such as MARA, BB, or HIMZ.")
    elif low_dte > high_dte or low_strike > high_strike:
        st.error("Minimum days/strike cannot exceed maximum.")
    else:
        with st.spinner("Looking up stock prices, expirations, option bids and asks…"):
            try:
                snapshot = fetch(ticker, int(low_dte), int(high_dte), kind, provider, api_token)
                st.session_state.snapshot = snapshot
                st.session_state.filter_params = dict(kind=kind, low=int(low_dte), high=int(high_dte),
                    budget=budget, account=account, low_strike=low_strike, high_strike=high_strike,
                    show_over=show_over, drift=drift, fees=fees)
            except MarketDataError as exc:
                st.session_state.pop("snapshot", None)
                st.error(str(exc))
            except ModuleNotFoundError as exc:
                st.session_state.pop("snapshot", None)
                st.error(f"Missing Python dependency: {exc.name}. Verify `requirements.txt` is in your GitHub repository root and Streamlit has redeployed.")
            except Exception as exc:
                st.session_state.pop("snapshot", None)
                st.error(f"Market-data service failed ({type(exc).__name__}). Refresh later or verify the data source.")

market = st.session_state.get("snapshot")
params = st.session_state.get("filter_params")
if market and params and params["kind"] != kind:
    st.info("Choose **Fetch prices & scan** to load the selected strategy. The earlier scan is not shown.")
if market and params and params["kind"] == kind:
    sym = market["ticker"]
    spot = market["spot"]
    st.markdown(f"### {sym} · {'Puts' if kind == 'put' else 'Calls'}")
    with st.container(border=True):
        a,b = st.columns(2)
        a.metric("Reported stock price" if kind == "put" else "Est. share purchase", money(spot))
        b.metric("Your trade budget", money(params["budget"]))
        st.caption(f"Provider: {market['provider']} · Stock source: {market['stock_source']}")
        st.caption("Stock quote: " + age_text(market.get("stock_quote_time")))
        st.caption("Fetched: " + market["fetched_at"].replace("T", " ") + " ET · Refresh by scanning again (45-sec cache).")
    if market["delayed"]:
        st.warning("**Not verified live:** option bid/ask may be stale or missing. Confirm fresh quotes and the actual contract deliverable in Webull before any order.")
    if sym in RISKY:
        st.warning("**Leveraged daily-reset ETF:** deep losses and long-run volatility effects aren't reliably captured by the simplified probability model.")
    if kind == "put":
        st.info("Cash-secured collateral = **strike × 100** for a standard contract. Selling the put can obligate you to buy 100 shares at that price. Premium received is not guaranteed profit.")
    else:
        estimate = spot*100
        if estimate > params["budget"]:
            st.warning(f"Buying 100 shares requires around **{money(estimate)}**, above your new-purchase limit. The option credit doesn't eliminate the initial share purchase.")

    eligible = []
    for option in market["rows"]:
        dte = (date.fromisoformat(option["expiration"]) - datetime.now(ET).date()).days
        strike = option["strike"]
        if not (params["low"] <= dte <= params["high"]):
            continue
        if not (params["low_strike"]*spot <= strike <= params["high_strike"]*spot):
            continue
        collateral = strike*100 if kind == "put" else spot*100
        if collateral > params["budget"] and not params["show_over"]:
            continue
        bid, ask = option["bid"], option["ask"]
        credit = bid*100
        fees = params["fees"]
        if kind == "put":
            maxprofit = credit-fees
            breakeven = strike-bid+fees/100
            maxloss = max(0, collateral-credit+fees)
        else:
            maxprofit = (strike-spot+bid)*100-fees
            breakeven = spot-bid+fees/100
            maxloss = max(0, collateral-credit+fees)
        spread = (ask-bid)/((ask+bid)*.5)
        vol = option["iv"]
        vol_source = "Provider option IV"
        if vol is None:
            vol = market.get("rv")
            vol_source = "Historical volatility fallback" if vol is not None else "Unavailable"
        warnings = []
        model = None
        if vol is not None and .03 <= vol <= 8.0:
            try:
                if kind == "put":
                    model = assess_put(spot, strike, bid, ask, dte, vol, params["budget"], option["oi"],
                        drift=params["drift"], leveraged=sym in RISKY, fees=fees)
                else:
                    model = assess(spot, strike, bid, ask, dte, vol, params["budget"], option["oi"],
                        drift=params["drift"], leveraged=sym in RISKY)
                    if fees:
                        model["p_profit"] = 0.0 if maxprofit <= 0 or breakeven >= strike else (1.0 if breakeven <= 0 else tail_above(spot, breakeven, vol, dte, params["drift"]))
                warnings.extend(model["warnings"])
            except ValueError:
                pass
        if not model:
            warnings.append("No usable volatility; cannot estimate probability")
        if option["oi"] < 25:
            warnings.append("Low open interest")
        if spread > .30:
            warnings.append("Wide bid/ask spread")
        if collateral > params["account"]*.30:
            warnings.append("One position uses over 30% of total account")
        if kind == "call" and maxprofit <= 0:
            warnings.append("No maximum profit at call-away price")
        eligible.append(dict(**option, dte=dte, collateral=collateral, credit=credit,
            maxprofit=maxprofit, maxloss=maxloss, breakeven=breakeven, spread=spread,
            vol=vol, vol_source=vol_source, model=model, warnings=list(dict.fromkeys(warnings)),
            roi=maxprofit/collateral))

    eligible.sort(key=lambda x: (
        x["model"] is not None and x["model"]["verdict"] not in {"PASS / EXCLUDE"},
        (x["model"] or {}).get("score", 0), x["oi"], x["roi"]), reverse=True)
    st.subheader("Matching cash-secured puts" if kind == "put" else "Matching covered calls")
    if not eligible:
        st.info("No standard 100-share contracts passed these filters with usable bids/asks. Try another ticker, widen the strike range, or enable research mode. This does not mean no listed options exist.")
    else:
        st.caption("Premium calculated using the *bid*. Probability estimates use a one-period lognormal volatility model, **not observed win rates**. Confirm actual execution price and market-data freshness.")
        for op in eligible[:20]:
            model = op["model"]
            with st.container(border=True):
                st.markdown(f"**{op['expiration']} · ${op['strike']:g} {'PUT' if kind == 'put' else 'CALL'} · {op['dte']} days**")
                if op["collateral"] > params["budget"]:
                    st.error("Over your per-trade budget — research only")
                a,b = st.columns(2)
                a.metric("Collateral" if kind == "put" else "100-share cost", money(op["collateral"]))
                b.metric("Premium (bid × 100)", money(op["credit"]))
                a,b = st.columns(2)
                a.metric("Break-even / share", money(op["breakeven"]))
                b.metric("Max profit", money(op["maxprofit"]))
                a,b = st.columns(2)
                a.metric("Maximum return / collateral" if kind == "put" else "Max return / share cost", percent(op["roi"]))
                b.metric("Modeled P(profit)", percent(model["p_profit"]) if model else "N/A")
                if model:
                    a,b = st.columns(2)
                    a.metric("P(finish ITM / possible assignment)" if kind == "put" else "P(finish above strike)",
                        percent(model["p_itm"] if kind == "put" else model["p_above"]))
                    b.metric("Trade score (NOT probability)", f"{model['score']}/100")
                    st.caption("Screening verdict: " + model["verdict"] + " · Simplified assumptions; not a buy signal")
                with st.expander("Contract, liquidity and downside details"):
                    st.write(f"**Standard contract:** `{op['contract']}` · Verify 100-share deliverable in Webull")
                    st.write(f"**Option bid/ask:** {money(op['bid'])} / {money(op['ask'])} · spread {percent(op['spread'])}")
                    st.write(f"**Bid time:** {age_text(op['bid_time'])} · **Ask time:** {age_text(op['ask_time'])}")
                    st.write(f"**Open interest:** {op['oi']:,} · **Daily option volume:** {op['volume']:,}")
                    st.write(f"**Annualized volatility:** {percent(op['vol'])} ({op['vol_source']})")
                    st.write(f"**Worst-case loss if stock goes to $0:** {money(op['maxloss'])}")
                    if model:
                        st.write(f"**P(stock down ≥25% by expiry):** {percent(model['p_drop25'])}")
                    st.write("Early assignment, earnings, ex-dividend dates, option exercise, tax and slippage events are not automatically checked.")
                    for warning in op["warnings"]:
                        st.warning(warning)
        rows = []
        for op in eligible:
            m = op["model"] or {}
            rows.append(dict(Ticker=sym,Type=kind.upper(),Expiration=op["expiration"],DTE=op["dte"],Strike=op["strike"],
                Stock_reference=spot,Collateral_or_100_shares=op["collateral"],Option_bid=op["bid"],Option_ask=op["ask"],
                Premium=op["credit"],Break_even=op["breakeven"],Max_profit=op["maxprofit"],Max_loss=op["maxloss"],
                Max_return=op["roi"],Modeled_P_profit=m.get("p_profit"),Modeled_P_ITM=m.get("p_itm",m.get("p_above")),
                Score_NOT_probability=m.get("score"),IV=op["vol"],IV_source=op["vol_source"],
                Open_interest=op["oi"],Volume=op["volume"],Contract=op["contract"],Bid_timestamp=op["bid_time"],
                Provider=market["provider"],Warnings="; ".join(op["warnings"])))
        st.download_button("Download results (CSV)",pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            f"{sym}_{kind}_scan.csv",mime="text/csv",use_container_width=True)

with st.expander("Manual calculator if the quote provider is down", expanded=False):
    st.caption("Enter your Webull quote. These inputs are NOT fetched live; all example amounts start blank or illustrative.")
    with st.form("manual_calculator"):
        m_spot = st.number_input("Current stock price ($)", min_value=.01, value=9.00, step=.25)
        a,b = st.columns(2)
        m_strike = a.number_input("Strike ($)", min_value=.01, value=9.0, step=.5)
        m_bid = b.number_input("Option bid per share ($)", min_value=.01, value=.50, step=.05)
        a,b = st.columns(2)
        m_ask = a.number_input("Option ask per share ($)", min_value=.01, value=.60, step=.05)
        m_dte = b.number_input("Days to expiry", min_value=1, max_value=730, value=30)
        m_iv = st.number_input("Annual implied volatility (%)", min_value=1., max_value=800., value=60., step=5.)
        m_fee = st.number_input("Total fee ($)", min_value=0., value=0., step=.50)
        compute = st.form_submit_button("Calculate from my Webull quote", use_container_width=True)
    if compute:
        try:
            if kind == "put":
                x = assess_put(m_spot,m_strike,m_bid,m_ask,m_dte,m_iv/100,budget,100,
                               drift=drift,fees=m_fee)
                st.metric("Put cash collateral",money(x["cost"]))
                st.metric("Max profit",money(x["max_profit"]))
                st.metric("Break-even / share",money(x["breakeven"]))
                st.metric("Modeled P(profit)",percent(x["p_profit"]))
                st.metric("Modeled P(finish ITM)",percent(x["p_itm"]))
            else:
                x = assess(m_spot,m_strike,m_bid,m_ask,m_dte,m_iv/100,budget,100,drift=drift)
                st.metric("100-share cost",money(x["cost"]))
                st.metric("Max profit before fees",money(x["max_profit"]))
                st.metric("Break-even before fees",money(x["breakeven"]))
                st.metric("Modeled P(profit) before fees",percent(x["p_profit"]))
                if m_fee:
                    st.caption("Call results above exclude entered fees. Fee-adjusted maximum profit = "+money(x["max_profit"]-m_fee))
            st.caption("Model estimate only. Use a genuine bid, verify the 100-share deliverable, and remember that early assignment and sharp losses can occur.")
        except ValueError as exc:
            st.error(f"Check the entered values: {exc}")

st.divider()
with st.expander("Trade math and probability limitations"):
    st.write("**Cash-secured put:** Collateral = 100×strike; premium = 100×put bid; maximum profit = premium − fees; break-even/share = strike − bid + fees/100; maximum possible loss = 100×strike − premium + fees. When assigned, you purchase 100 shares at the strike regardless of a lower market price.")
    st.write("**Covered call:** Buy 100 shares; write 1 standard call; maximum profit at expiration = 100×(strike − paid share price + bid) − fees. A call does not remove the risk of losses on the stock.")
    st.write("**Estimated P(profit):** Chance the stock finishes above the net break-even under a single-date lognormal model with 0% annual drift by default and option IV (or historical realized volatility). Simplifying assumptions make these uncalibrated numbers inappropriate for predicting individual outcomes. An estimated 80% is not an 80% historical success rate.")
    st.write("**P(finish ITM):** For puts, chance the stock finishes below strike. This is not the probability of early assignment. American-style contracts can be exercised early.")
    st.write("**Trade score:** Heuristic combining modeled probability, maximum return, liquidity and risk. Not a probability, guarantee, or substitute for fundamental analysis. Earnings and pending corporate actions are **not automatically screened**.")
    st.write("**Quotes:** Tradier production requires eligible brokerage entitlements; sandbox is delayed; Yahoo is delayed/unverified. The app reads prices only; it never places orders. Tokens belong in Streamlit Secrets, never GitHub.")
st.caption("Research, not personalized trade execution advice · Check real quotes and contract details with your brokerage before entering orders.")
