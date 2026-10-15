"""Phone-friendly options scanner. Single-file application: no custom Python modules required.

Quotes: unofficial Yahoo Finance/yfinance source; may be delayed, stale, rate-limited, or absent.
The manual Webull quote calculator remains available even if Yahoo fails.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re
from statistics import NormalDist, stdev
from zoneinfo import ZoneInfo

import streamlit as st

ET = ZoneInfo("America/New_York")
N = NormalDist()
LEVERAGED = {"HIMZ", "MSTX", "MSTU", "MSTZ", "AMDL", "NVDL", "TSLL", "CONL", "TQQQ", "SOXL", "UPRO", "SPXL", "BITX", "ETHU"}

st.set_page_config(page_title="Options Income Scanner", page_icon="📈", layout="centered")
st.markdown("""<style>
.block-container {max-width: 790px; padding: .8rem .9rem 4rem;}
h1 {font-size: 1.8rem !important;}
.stButton button,.stFormSubmitButton button {min-height: 52px; border-radius: 12px; font-weight:650;}
div[data-testid="stMetric"] {padding: 12px; border: 1px solid #57616b40; border-radius: 12px;}
@media(max-width:650px){.block-container{padding:.5rem .65rem 3rem} div[data-testid="stMetricValue"]{font-size:1.35rem!important;}}
</style>""", unsafe_allow_html=True)


def dollar(x):
    return f"${x:,.2f}"


def pc(x):
    return f"{x:.1%}" if x is not None and math.isfinite(x) else "Not available"


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def tail_above(spot, threshold, sigma, dte):
    """Illustrative probability under zero-drift lognormal terminal stock-price distribution."""
    if min(spot, threshold, sigma) <= 0 or dte <= 0:
        return None
    t = dte / 365.0
    z = (math.log(threshold / spot) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    return max(0.0, min(1.0, 1.0 - N.cdf(z)))


def analyze(kind, spot, strike, bid, ask, dte, iv, oi, budget, fee=0.0):
    """A single standard 100-share option, sold at its quoted bid."""
    if any(not math.isfinite(x) for x in (spot, strike, bid, ask, fee)):
        raise ValueError("Price or fee is missing.")
    if spot <= 0 or strike <= 0 or bid < 0 or ask < bid or dte <= 0 or fee < 0:
        raise ValueError("Check stock price, strike, bid, ask, expiration, and fees.")
    share_cost = spot * 100
    collateral = strike * 100 if kind == "Cash-secured put" else share_cost
    premium = bid * 100
    net_credit = premium - fee
    if kind == "Cash-secured put":
        breakeven = strike - bid + fee / 100
        max_profit = net_credit
        loss_at_zero = max(0.0, collateral - premium + fee)
        itm = 1 - tail_above(spot, strike, iv, dte) if iv else None
    else:
        breakeven = spot - bid + fee / 100
        max_profit = (strike - spot + bid) * 100 - fee
        loss_at_zero = max(0.0, collateral - premium + fee)
        itm = tail_above(spot, strike, iv, dte) if iv else None

    pprofit = None
    pdrop = None
    if iv is not None and 0.03 <= iv <= 8.0:
        if max_profit <= 0:
            pprofit = 0.0
        elif breakeven <= 0:
            pprofit = 1.0
        else:
            pprofit = tail_above(spot, breakeven, iv, dte)
        pdrop = 1 - tail_above(spot, spot * 0.75, iv, dte)
    denominator = (ask + bid) / 2
    spread = (ask - bid) / denominator if denominator > 0 else None
    roi = max_profit / collateral
    # Screening-quality score: NOT a probability of winning.
    liq_score = max(0, 1 - (spread or 1) / .40) * min(1, (oi or 0) / 100)
    yield_score = min(1, max(0, roi) / max(.001, .12 * dte / 365))
    score = (0.5 * (pprofit if pprofit is not None else 0.3)
             + 0.2 * yield_score + 0.2 * liq_score
             + 0.1 * (1 - (pdrop if pdrop is not None else .5)))
    score = round(100 * score)
    flags = []
    if collateral > budget:
        flags.append("Over your per-trade budget")
    if max_profit <= 0:
        flags.append("No positive maximum profit at quoted bid")
    if bid <= 0:
        flags.append("No positive bid quote")
    if spread is None or spread > .35:
        flags.append("Wide or missing bid/ask spread")
    if (oi or 0) < 25:
        flags.append("Low/missing open interest")
    if iv is None:
        flags.append("Volatility unavailable: probabilities not estimated")
    return dict(collateral=collateral, premium=premium, max_profit=max_profit,
                roi=roi, breakeven=breakeven, loss_at_zero=loss_at_zero,
                itm=itm, pprofit=pprofit, pdrop=pdrop, spread=spread,
                score=score, flags=flags)


@st.cache_data(ttl=120, show_spinner=False)
def yahoo_scan(ticker, minimum, maximum, side):
    """All imports/data errors handled by caller; no external app modules needed."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    today = datetime.now(ET).date()
    hist = tk.history(period="3mo", auto_adjust=False)
    closes = []
    if hist is not None and not hist.empty and "Close" in hist:
        closes = [float(x) for x in hist["Close"].dropna().tolist() if float(x) > 0]
    if not closes:
        raise RuntimeError("Yahoo did not return a usable stock price.")
    spot = closes[-1]
    rv = None
    if len(closes) >= 22:
        changes = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
        if len(changes) >= 20:
            rv = stdev(changes[-60:]) * math.sqrt(252)
    expirations = []
    for item in tk.options:
        try:
            d = date.fromisoformat(str(item))
        except ValueError:
            continue
        if minimum <= (d - today).days <= maximum:
            expirations.append(str(item))
    rows = []
    failures = []
    for expiration in expirations[:10]:
        try:
            chain = tk.option_chain(expiration)
            frame = chain.puts if side == "Cash-secured put" else chain.calls
            dte = (date.fromisoformat(expiration) - today).days
            for _, row in frame.iterrows():
                strike = num(row.get("strike"))
                bid = num(row.get("bid"))
                ask = num(row.get("ask"))
                if strike is None or strike <= 0 or bid is None or bid <= 0 or ask is None or ask < bid:
                    continue
                contract = str(row.get("contractSymbol", ""))
                # Adjusted roots often indicate nonstandard deliverables. Exclude these.
                clean = ticker.replace(".", "").replace("-", "").upper()
                m = re.match(r"^([A-Z]+\d?)(\d{6})([CP])(\d{8})$", contract)
                if m and m.group(1) != clean:
                    continue
                iv = num(row.get("impliedVolatility"))
                if iv is not None and not (.03 <= iv <= 8):
                    iv = None
                rows.append(dict(expiration=expiration, dte=dte, strike=strike,
                                 bid=bid, ask=ask, iv=iv,
                                 volatility_source="Option implied volatility" if iv else "Historical fallback",
                                 oi=int(num(row.get("openInterest")) or 0),
                                 volume=int(num(row.get("volume")) or 0),
                                 contract=contract))
        except Exception as exc:
            failures.append(f"{expiration}: {type(exc).__name__}")
    return dict(spot=spot, volatility=rv, rows=rows, expirations=expirations,
                failures=failures, fetched=datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET"))


st.title("📱 Options Income Scanner")
st.caption("Covered calls + cash-secured puts · 100-share contracts · No broker login or trade execution")
st.info("Automatic prices come from Yahoo Finance, **not Webull**. They can be delayed, stale, or unavailable. Always check the actual bid, ask, and contract deliverable in Webull.")

strategy = st.radio("Strategy", ["Cash-secured put", "Covered call"], horizontal=True)
mode = st.radio("Market-data mode", ["Automatic Yahoo scan", "Manual Webull quote"], horizontal=True,
                help="Manual mode works even if the Yahoo feed is blocked or down.")

if mode == "Automatic Yahoo scan":
    with st.form("scan"):
        ticker = st.text_input("Ticker", value="MARA", max_chars=12).strip().upper()
        a, b = st.columns(2)
        budget = a.number_input("Trade limit ($)", value=1000., min_value=100., step=100.)
        account = b.number_input("Account equity ($)", value=4300., min_value=100., step=100.)
        a, b = st.columns(2)
        minimum = a.number_input("Minimum days", min_value=1, max_value=365, value=21)
        maximum = b.number_input("Maximum days", min_value=1, max_value=365, value=45)
        over_budget = st.checkbox("Show over-budget trades for comparison", value=False)
        run = st.form_submit_button("🔎 Fetch prices & scan", type="primary", use_container_width=True)
    if run:
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,11}", ticker):
            st.error("Enter a ticker such as MARA or HIMZ.")
        elif minimum > maximum:
            st.error("Minimum days must not exceed maximum days.")
        else:
            with st.spinner("Requesting reported prices and option chains…"):
                try:
                    snapshot = yahoo_scan(ticker, int(minimum), int(maximum), strategy)
                    st.session_state["scanner_data"] = (strategy, ticker, budget, account, over_budget, snapshot)
                    st.session_state.pop("scanner_error", None)
                except Exception as exc:
                    st.session_state.pop("scanner_data", None)
                    st.session_state["scanner_error"] = (type(exc).__name__, str(exc))
    if st.session_state.get("scanner_error"):
        etype, message = st.session_state["scanner_error"]
        st.error("Yahoo Finance could not supply data right now. Your app is still working; switch to **Manual Webull quote** to calculate a trade.")
        with st.expander("Error details for troubleshooting"):
            st.code(f"{etype}: {message[:700]}")
    data = st.session_state.get("scanner_data")
    if data and data[0] == strategy:
        _, sym, budget, account, over_budget, snapshot = data
        st.subheader(f"{sym} · {strategy}")
        st.metric("Reported share price (last daily close)", dollar(snapshot["spot"]))
        st.caption(f"Fetched: {snapshot['fetched']} · Source: Yahoo daily history/option chain · Not a live execution quote")
        if sym in LEVERAGED:
            st.warning("This ticker is a daily leveraged ETF. A high option premium does not eliminate substantial loss risk.")
        if strategy == "Covered call" and snapshot["spot"] * 100 > budget:
            st.warning(f"Buying 100 shares would cost about {dollar(snapshot['spot'] * 100)}, above your {dollar(budget)} budget.")
        if not snapshot["expirations"]:
            st.warning("No expiration dates were returned in the selected 21–45-day window. Try a wider range.")
        if snapshot["failures"]:
            st.caption(f"Some expiration chains failed to load ({len(snapshot['failures'])}).")
        picks = []
        for option in snapshot["rows"]:
            iv = option["iv"] if option["iv"] else snapshot["volatility"]
            try:
                outcome = analyze(strategy, snapshot["spot"], option["strike"], option["bid"],
                                  option["ask"], option["dte"], iv, option["oi"], budget)
            except ValueError:
                continue
            if not over_budget and outcome["collateral"] > budget:
                continue
            picks.append((outcome["score"], option, outcome, iv))
        picks.sort(key=lambda x: x[0], reverse=True)
        if not picks:
            st.warning("No usable quoted contracts fit your settings. This may mean the trade exceeds your budget, has no quoted bid, or Yahoo returned incomplete data. Turn on **Show over-budget trades** or use manual mode.")
        else:
            st.caption(f"{len(picks)} quoted contracts matched. Ranked by an illustrative trade-quality score, **not** a win probability.")
            for _, opt, outcome, iv in picks[:15]:
                label = (f"{opt['expiration']} · ${opt['strike']:g} strike · "
                         f"${opt['bid']:.2f} bid / ${opt['ask']:.2f} ask")
                with st.expander(label, expanded=False):
                    a, b = st.columns(2)
                    a.metric("Premium at bid", dollar(outcome["premium"]))
                    b.metric("Cash needed", dollar(outcome["collateral"]))
                    a, b = st.columns(2)
                    a.metric("Break-even at expiration", dollar(outcome["breakeven"]))
                    b.metric("Max profit", dollar(outcome["max_profit"]))
                    a, b = st.columns(2)
                    a.metric("Est. profit probability", pc(outcome["pprofit"]))
                    b.metric("Est. finishes ITM", pc(outcome["itm"]))
                    st.write(f"Max return on committed capital: **{pc(outcome['roi'])}**  ·  Quality score: **{outcome['score']}/100**")
                    st.write(f"Volume: {opt['volume']} · Open interest: {opt['oi']} · IV used: {pc(iv)}")
                    st.caption(f"Expiration: {opt['dte']} days · Max theoretical loss if stock goes to zero: {dollar(outcome['loss_at_zero'])}")
                    for flag in outcome["flags"]:
                        st.caption("⚠️ " + flag)
else:
    with st.form("manual"):
        st.caption("Enter the prices you see in Webull. Uses the **bid** as the conservative sell-to-open premium estimate.")
        a, b = st.columns(2)
        spot = a.number_input("Stock price ($)", min_value=.01, value=9.00, step=.10, format="%.2f")
        strike = b.number_input("Option strike ($)", min_value=.01, value=9.00, step=.50, format="%.2f")
        a, b = st.columns(2)
        bid = a.number_input("Option bid ($/share)", min_value=0., value=.65, step=.05, format="%.2f")
        ask = b.number_input("Option ask ($/share)", min_value=0., value=.80, step=.05, format="%.2f")
        a, b = st.columns(2)
        expiry = a.date_input("Expiration date", value=datetime.now(ET).date() + timedelta(days=30))
        iv_pct = b.number_input("Annual volatility (%)", min_value=0., max_value=800., value=80., step=5.)
        a, b = st.columns(2)
        budget = a.number_input("Trade limit ($)", min_value=100., value=1000., step=100.)
        fee = b.number_input("Estimated fees ($)", min_value=0., value=0., step=.5)
        oi = st.number_input("Open interest (0 if unknown)", min_value=0, value=100)
        calculate = st.form_submit_button("🧮 Calculate trade", type="primary", use_container_width=True)
    if calculate:
        dte = (expiry - datetime.now(ET).date()).days
        try:
            result = analyze(strategy, spot, strike, bid, ask, dte, iv_pct / 100 if iv_pct else None, oi, budget, fee)
            st.subheader("Trade estimate")
            a, b = st.columns(2)
            a.metric("Premium received", dollar(result["premium"]))
            b.metric("Required capital", dollar(result["collateral"]))
            a, b = st.columns(2)
            a.metric("Break-even at expiration", dollar(result["breakeven"]))
            b.metric("Maximum profit", dollar(result["max_profit"]))
            a, b = st.columns(2)
            a.metric("Modeled profit probability", pc(result["pprofit"]))
            b.metric("Modeled finishes ITM", pc(result["itm"]))
            st.write("Max return on capital:", pc(result["roi"]))
            st.write("Max theoretical loss (stock goes to zero):", dollar(result["loss_at_zero"]))
            for item in result["flags"]:
                st.warning(item)
        except ValueError as exc:
            st.error(str(exc))

st.divider()
with st.expander("How the probability estimate works"):
    st.write("Estimates assume the share price at expiration follows a lognormal distribution with zero expected annual price drift and the entered/implied/historical annual volatility. This is an illustration, not a calibrated forecast or guarantee. A leveraged ETF may diverge greatly from these assumptions. American-style options may be assigned before expiration. The score is a subjective screening heuristic, not a probability.")
    st.write("One standard contract controls 100 shares. Put collateral = 100 × strike; covered-call purchase = 100 × stock price. Premium is not free profit: the underlying shares can fall sharply.")
