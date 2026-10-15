"""Personal options opportunity report for Streamlit Community Cloud.

On-demand watchlist scanner using MarketData.app. This is not a whole-market scan,
not a broker execution client, and never silently labels quotes as real time.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

API_ROOT = "https://api.marketdata.app/v1"
DEFAULT_WATCHLIST = "BB, SNAP, SOUN, NOK, UAMY, RIG, NIO, PLUG, BBAI, PTON, F, SOFI, MARA, AAL, LCID"
LEVERAGED = {"HIMZ", "MSTX", "BITX", "TQQQ", "SQQQ", "SOXL", "SOXS", "TSLL", "NVDL", "CONL"}
NY = ZoneInfo("America/New_York")


def numeric(value, fallback=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_above(spot, threshold, sigma, days):
    """Simplified lognormal P(spot_at_expiry > threshold), zero price drift.
    NOT an observed back-tested win rate. No jumps or early exercise.
    """
    if spot <= 0 or threshold <= 0 or sigma is None or sigma <= 0 or days <= 0:
        return None
    z = (math.log(spot / threshold) - 0.5 * sigma * sigma * days / 365.0) / (sigma * math.sqrt(days / 365.0))
    return max(0.0, min(1.0, norm_cdf(z)))


def quality_score(pop, roi, oi, volume, spread_pct, buffer_pct, iv, leveraged):
    """Heuristic 0--100 rank, not the probability of a 'good trade'."""
    oi_score = min(max(oi, 0) / 500.0, 1.0)
    vol_score = min(max(volume, 0) / 40.0, 1.0)
    spread_quality = max(0.0, 1.0 - min(max(spread_pct, 0.0) / 0.45, 1.0))
    liquidity = 0.45 * oi_score + 0.20 * vol_score + 0.35 * spread_quality
    probability = pop if pop is not None else 0.40
    return_component = min(max(roi, 0.0) / 0.12, 1.0)
    downside_component = min(max(buffer_pct, 0.0) / 0.20, 1.0)
    risk_component = 1.0 if iv is not None and iv <= 0.60 else 0.65 if iv is not None and iv <= 1.0 else 0.25
    if iv is None:
        risk_component = 0.25
    score = (25 * probability + 20 * return_component + 25 * liquidity
             + 20 * downside_component + 10 * risk_component)
    if leveraged:
        score -= 25
    return round(max(0.0, min(100.0, score)), 1)


def rows_from_columnar(payload):
    """MarketData API returns arrays grouped by field, not row dictionaries."""
    if isinstance(payload.get("data"), list):
        return payload["data"]
    array_columns = {k: v for k, v in payload.items() if isinstance(v, list)}
    length = max((len(v) for v in array_columns.values()), default=0)
    return [{k: (v[i] if i < len(v) else None) for k, v in array_columns.items()} for i in range(length)]


def human_time(unix):
    value = numeric(unix)
    if value is None:
        return "Unknown"
    return datetime.fromtimestamp(value, NY).strftime("%Y-%m-%d %I:%M %p ET")


def calculate_candidate(symbol, side, record, today, capital_limit, min_oi=50,
                        max_spread=0.45, allow_assumed_iv=False,
                        assumed_iv=0.70, max_age_days=4):
    """Returns (candidate, reason). Reason is used to avoid silent omissions."""
    strike = numeric(record.get("strike"))
    bid = numeric(record.get("bid"))
    ask = numeric(record.get("ask"))
    spot = numeric(record.get("underlyingPrice"))
    oi = int(numeric(record.get("openInterest"), 0) or 0)
    vol = int(numeric(record.get("volume"), 0) or 0)
    raw_iv = numeric(record.get("iv"))
    # MarketData.app documents IV in decimals. Do not reinterpret unusually high
    # provider IV as a percentage; reject impossible values instead.
    iv = raw_iv if raw_iv is not None and 0 < raw_iv <= 5 else None
    if strike is None or strike <= 0 or bid is None or bid <= 0 or ask is None or ask < bid or spot is None or spot <= 0:
        return None, "Missing/invalid spot, strike, or two-sided quote"
    raw_dte = numeric(record.get("dte"))
    exp = record.get("expiration")
    if exp is not None and numeric(exp) is not None:
        expiration = datetime.fromtimestamp(float(exp), NY).date()
    elif isinstance(exp, str):
        try:
            expiration = date.fromisoformat(exp[:10])
        except ValueError:
            return None, "Unparseable expiration"
    else:
        return None, "Missing expiration"
    days = (expiration - today).days
    if days <= 0:
        return None, "Expired contract"
    if days < 21 or days > 45:
        return None, "Outside 21-45-day expiration window"
    updated = numeric(record.get("updated"))
    age_days = None
    if updated is not None:
        age_days = (datetime.now(timezone.utc).timestamp() - updated) / 86400.0
        if age_days > max_age_days:
            return None, "Quote older than age limit"
    if oi < min_oi:
        return None, "Open interest too low"
    midpoint = (bid + ask) / 2.0
    spread_pct = (ask - bid) / midpoint if midpoint > 0 else 1.0
    if spread_pct > max_spread:
        return None, "Bid/ask spread too wide"
    side = side.lower()
    if side == "put":
        capital = strike * 100.0
        if capital > capital_limit:
            return None, "Put collateral exceeds limit"
        # Avoid distant ITM strikes and ultra-low delta strikes for an income approach.
        if not (0.60 <= strike / spot <= 1.06):
            return None, "Put strike outside preferred spot ratio"
        breakeven = strike - bid
        max_profit = 100.0 * bid
        max_loss = max(0.0, 100.0 * (strike - bid))
        roi = max_profit / capital
        prob = prob_above(spot, breakeven, iv if iv is not None else assumed_iv if allow_assumed_iv else None, days)
        assign_prob = None if prob is None else 1.0 - prob_above(spot, strike, iv if iv is not None else assumed_iv if allow_assumed_iv else None, days)
    elif side == "call":
        capital = spot * 100.0
        if capital > capital_limit:
            return None, "Covered-call stock purchase exceeds limit"
        if not (0.90 <= strike / spot <= 1.30):
            return None, "Call strike outside preferred spot ratio"
        breakeven = spot - bid
        max_profit = 100.0 * (strike - spot + bid)
        if max_profit <= 0:
            return None, "Cannot profit on covered-call cap"
        max_loss = max(0.0, capital - bid * 100.0)
        roi = max_profit / capital
        if iv is not None or allow_assumed_iv:
            use_iv = iv if iv is not None else assumed_iv
            prob = prob_above(spot, breakeven, use_iv, days)
            assign_prob = prob_above(spot, strike, use_iv, days)
        else:
            prob = assign_prob = None
    else:
        return None, "Unknown option side"
    if breakeven <= 0:
        return None, "Nonsensical break-even"
    sigma = iv if iv is not None else (assumed_iv if allow_assumed_iv else None)
    buffer_pct = max(0, (spot - breakeven) / spot)
    warning = "Leveraged ETF — amplified risk" if symbol.upper() in LEVERAGED else ""
    score = quality_score(prob, roi, oi, vol, spread_pct, buffer_pct, sigma, bool(warning))
    return {
        "Score /100": score,
        "Ticker": symbol.upper(),
        "Strategy": "Cash-secured put" if side == "put" else "Covered call",
        "Share price $": round(spot, 2),
        "Expiration": expiration.isoformat(),
        "DTE": days,
        "Strike $": round(strike, 2),
        "Bid $": round(bid, 2),
        "Ask $": round(ask, 2),
        "Premium $": round(bid * 100, 2),
        "Capital required $": round(capital, 2),
        "Break-even $": round(breakeven, 2),
        "Max profit $": round(max_profit, 2),
        "Max loss $": round(max_loss, 2),
        "Max return %": round(roi * 100, 2),
        "Model POP %": None if prob is None else round(prob * 100, 1),
        "Model assignment %": None if assign_prob is None else round(assign_prob * 100, 1),
        "IV %": None if sigma is None else round(sigma * 100, 1),
        "IV source": "Option quote" if iv is not None else "Assumed by user" if allow_assumed_iv else "Unavailable",
        "Open interest": oi,
        "Volume": vol,
        "Spread %": round(spread_pct * 100, 1),
        "Quote updated ET": human_time(updated),
        "Quote age (days)": None if age_days is None else round(max(0, age_days), 2),
        "Risk flag": warning,
        "Source": "MarketData.app options chain; historical or delayed per plan",
        "Contract": record.get("optionSymbol", ""),
    }, "OK"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chain(symbol, side, target_days, api_token):
    url = f"{API_ROOT}/options/chain/{symbol}/"
    params = {"dte": int(target_days), "side": side, "strikeLimit": 5, "minBid": 0.05}
    try:
        response = requests.get(url, params=params,
                                headers={"Authorization": f"Bearer {api_token}", "Accept": "application/json"},
                                timeout=22)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network issue fetching {symbol} {side}: {type(exc).__name__}") from exc
    if response.status_code == 401:
        raise RuntimeError("API token rejected (401). Check your Streamlit secret.")
    if response.status_code == 403:
        raise RuntimeError("Provider denied access (403). Check plan, IP limits, and access rules.")
    if response.status_code == 429:
        raise RuntimeError("Daily API quota exceeded (429). Wait for the credit reset or scan fewer symbols.")
    if response.status_code not in (200, 203):
        raise RuntimeError(f"Market-data provider returned HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Provider returned invalid JSON.") from exc
    if payload.get("s") == "error":
        raise RuntimeError(f"Provider error: {str(payload.get('errmsg', 'unknown'))[:160]}")
    if payload.get("s") == "no_data":
        return []
    return rows_from_columnar(payload)


def parse_symbols(text, limit=8):
    symbols = []
    for candidate in re.split(r"[\s,;]+", text.upper()):
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", candidate) and candidate not in symbols:
            symbols.append(candidate)
    return symbols[:limit]


def get_token():
    token = os.environ.get("MARKETDATA_TOKEN", "")
    if token:
        return token
    try:
        return st.secrets.get("MARKETDATA_TOKEN", "")
    except (FileNotFoundError, KeyError, AttributeError):
        return ""


def main():
    st.set_page_config(page_title="Options Opportunity Report", page_icon="📊", layout="wide")
    st.title("📊 Options Opportunity Report")
    st.caption("Find potential cash-secured puts and covered calls across multiple stocks — personal research, not trade execution.")

    with st.expander("Account settings and scan filters", expanded=True):
        a, b = st.columns(2)
        with a:
            account = st.number_input("Trading account ($)", min_value=100.0, max_value=1000000.0, value=4300.0, step=100.0)
            trade_limit = st.number_input("Max capital per trade ($)", min_value=100.0, max_value=float(account), value=min(1000.0, float(account)), step=100.0)
            strategy = st.selectbox("Strategies", ["Both", "Cash-secured put", "Covered call"])
        with b:
            target_dte = st.slider("Preferred expiry (days; nearest listed)", 21, 45, 30)
            min_oi = st.slider("Minimum option open interest", 0, 500, 50, 25)
            max_spread_pct = st.slider("Maximum bid/ask spread (%)", 10, 100, 45, 5)
        st.caption("Default scan checks the *one listed expiry closest* to the selected day count. This reduces free-plan API usage; it does not exhaustively scan all 21–45-day expirations.")
        symbols_text = st.text_area("Watchlist: potential tickers (up to 8 per scan)", value=DEFAULT_WATCHLIST,
                                    height=95, help="The report covers a watchlist, not the entire market. Choose batch 2 to scan symbols 9–16.")
        batch_number = st.number_input("Watchlist batch (1 = tickers 1–8, 2 = tickers 9–16)", min_value=1, max_value=10, value=1, step=1)
        include_leveraged = st.checkbox("Allow known leveraged ETFs (high risk)", value=False)
        missing_iv = st.checkbox("Estimate probabilities when IV is missing (assumed annual volatility)", value=False)
        assumed_iv = st.slider("Assumed volatility (%) when enabled", 20, 200, 70, 5) / 100.0 if missing_iv else 0.70
        st.caption("Maximum share purchase or put collateral must be ≤ your position limit. Cash received as premium is not subtracted from a put's cash-collateral requirement.")

    st.info("**Data:** MarketData.app free plan is at least 24 hours delayed. Results reflect the quote timestamp, not the price you can execute now. Verify bids in Webull.")
    token = get_token()
    if not token:
        st.warning("Set up your free MarketData.app API token once to enable scans. Go to Streamlit **Manage app → Settings → Secrets** and add `MARKETDATA_TOKEN = \"your_token\"`. Never put the token on GitHub.")
        st.markdown("[Create a free MarketData.app account](https://www.marketdata.app/) · [Provider documentation](https://www.marketdata.app/docs/api/options/chain/)")
    if st.button("🔎 Pull opportunity report", type="primary", use_container_width=True, disabled=not bool(token)):
        all_symbols = parse_symbols(symbols_text, limit=80)
        first = (int(batch_number) - 1) * 8
        watchlist = all_symbols[first:first + 8]
        if not include_leveraged:
            watchlist = [s for s in watchlist if s not in LEVERAGED]
        if not watchlist:
            st.error("No valid tickers selected.")
            return
        sides = ["put", "call"] if strategy == "Both" else ["put"] if strategy == "Cash-secured put" else ["call"]
        all_candidates = []
        notes = []
        progress = st.progress(0, text="Retrieving option chains...")
        total = len(watchlist) * len(sides)
        count = 0
        for ticker in watchlist:
            for side in sides:
                count += 1
                try:
                    contracts = fetch_chain(ticker, side, target_dte, token)
                    reasons = {}
                    matched = 0
                    for opt in contracts:
                        entry, reason = calculate_candidate(ticker, side, opt, datetime.now(NY).date(), float(trade_limit),
                                                            min_oi=int(min_oi), max_spread=float(max_spread_pct)/100,
                                                            allow_assumed_iv=missing_iv, assumed_iv=assumed_iv)
                        if entry:
                            all_candidates.append(entry)
                            matched += 1
                        else:
                            reasons[reason] = reasons.get(reason, 0) + 1
                    if not contracts:
                        notes.append({"Ticker": ticker, "Side": side, "Contracts returned": 0, "Matches": 0,
                                      "Reason": "Data provider returned no contracts for the nearest listed expiry"})
                    elif not matched:
                        top_reason = max(reasons, key=reasons.get) if reasons else "No matching contracts"
                        notes.append({"Ticker": ticker, "Side": side, "Contracts returned": len(contracts), "Matches": 0, "Reason": top_reason})
                    else:
                        notes.append({"Ticker": ticker, "Side": side, "Contracts returned": len(contracts), "Matches": matched, "Reason": "OK"})
                except RuntimeError as exc:
                    notes.append({"Ticker": ticker, "Side": side, "Contracts returned": 0, "Matches": 0, "Reason": str(exc)})
                progress.progress(count / total, text=f"Scanned {count}/{total} symbol-strategy combinations")
        progress.empty()
        st.session_state["scan_rows"] = all_candidates
        st.session_state["scan_diagnostics"] = notes
        st.session_state["scan_time"] = datetime.now(NY).strftime("%Y-%m-%d %I:%M %p ET")

    if "scan_rows" in st.session_state:
        rows = st.session_state["scan_rows"]
        st.subheader("Report results")
        st.caption(f"Scan requested: {st.session_state.get('scan_time','')} · Quotes may be older; see individual timestamps. Latest report remains visible until you scan again.")
        if rows:
            frame = pd.DataFrame(rows).sort_values(["Score /100", "Model POP %", "Max return %"], ascending=False, na_position="last")
            one_per = frame.drop_duplicates("Ticker", keep="first")
            st.metric("Qualifying contracts", len(frame))
            tab1, tab2, tab3 = st.tabs(["🏆 Best per ticker", "📋 Every matching trade", "⚠️ Data and exclusions"])
            cols = ["Score /100", "Ticker", "Strategy", "Share price $", "Strike $", "Expiration", "DTE", "Bid $", "Ask $", "Premium $", "Capital required $", "Break-even $", "Max profit $", "Max return %", "Model POP %", "Open interest", "Spread %", "Quote updated ET"]
            with tab1:
                st.dataframe(one_per[cols].head(12), hide_index=True, use_container_width=True)
                st.caption("One highest-ranked contract per ticker. Rank is a heuristic, **not** a success probability or guarantee.")
            with tab2:
                st.dataframe(frame[cols + ["IV source", "Model assignment %", "Max loss $", "Risk flag"]], hide_index=True, use_container_width=True)
                csv_data = frame.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ Download full report (CSV / Excel compatible)", data=csv_data,
                                   file_name=f"options_opportunities_{datetime.now(NY).date().isoformat()}.csv", mime="text/csv", use_container_width=True)
            with tab3:
                st.dataframe(pd.DataFrame(st.session_state.get("scan_diagnostics", [])), hide_index=True, use_container_width=True)
                st.caption("Quotes older than 4 days, weak bids, illiquid options, and over-budget trades are excluded. Dates may be missing because the provider has no data; that does not mean the stock has no listed options.")
        else:
            st.warning("No contracts met your filters. Review the diagnostic table; a provider error, missing quote, or capital limit may be the reason.")
            st.dataframe(pd.DataFrame(st.session_state.get("scan_diagnostics", [])), hide_index=True, use_container_width=True)

    with st.expander("How the probabilities and scores work"):
        st.markdown("""**Model probability of profit** uses a zero-drift lognormal stock-price model: at expiry, the trade is profitable if the stock finishes above the option's break-even. IV from the option quote is used where provided. If missing, probability is blank unless you explicitly enable an assumed volatility. This does not predict earnings gaps, dividends, volatility changes, or early exercise.

**Trade quality score (0–100)** uses modeled profit probability, return if assigned or put expires, bid/ask spread, volume, open interest, downside buffer, and a volatility penalty. **It is not the chance a trade will be good.** Options premium received is not free profit: the stock can lose more than the premium. Assumes standard 100-share options; nonstandard/adjusted contracts need manual verification. Earnings and corporate-action dates are not screened.

**Data provider:** [MarketData.app](https://www.marketdata.app/). Free-tier data is ≥24h delayed and limited to 100 daily API credits. Each contract with a quote may use a credit, so repeated broad scans can exhaust your allowance. **This is a personal watchlist scanner, not an entire-US-market scan.** For actionable decisions confirm real Webull prices, earnings, and corporate actions before trading.
""")
    st.caption("Personal research tool. No order execution. For personal use only; do not redistribute licensed market data through a public app.")


if __name__ == "__main__":
    main()
