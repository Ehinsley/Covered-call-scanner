"""Automatic US option-income opportunity reporter for Streamlit.

Personal research only. No trade execution; quote source/delay disclosed.
Run: streamlit run app.py
"""
from __future__ import annotations

import math
import os
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

NY = ZoneInfo("America/New_York")
TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"
NASDAQ_SCREEN = "https://api.nasdaq.com/api/screener/stocks"
CBOE_CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
# Backup universe, not a claim that these are currently good trades.
STARTER_UNIVERSE = (
    "F BB SNAP RIG NOK NIO PLUG SOUN BBAI AAL LCID PTON UAA WBD"
    " HOOD SOFI MARA HIMS UAMY RKT GPRO TLRY MVIS CLF KOS ACHR"
    " JOBY IONQ OPEN CHPT BTG GOLD KGC CDE AG GME AMC PARA"
    " HBI SAVE SIRI VALE NNDM VZ KHC PBR NU BBD ET WULF"
).split()
LEVERAGED_FUNDS = {"HIMZ", "MSTX", "MSTU", "BITX", "TSLL", "NVDL", "TQQQ", "SQQQ", "SOXL", "SOXS", "CONL", "AMDL"}


def finite(value, default=None):
    try:
        x = float(str(value).replace(",", "").replace("$", "").replace("%", ""))
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def prob_above(spot, target, iv, days):
    """Terminal P(S_T > target) under zero-drift lognormal assumption."""
    if not all([spot and spot > 0, target and target > 0, iv and 0 < iv < 5, days and days > 0]):
        return None
    t = days / 365.0
    z = (math.log(spot / target) - 0.5 * iv ** 2 * t) / (iv * math.sqrt(t))
    return min(1.0, max(0.0, norm_cdf(z)))


def parse_iv(x):
    v = finite(x)
    return v if v is not None and 0.01 <= v <= 4 else None


def normalize_feed_time(value):
    """Parse an explicitly labeled FEED snapshot time (not last-trade time)."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=NY) if not any(c in value for c in ("Z", "+")) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return value

def timestamp_to_et(value):
    if value is None:
        return "Unknown (bid/ask may be stale)"
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromtimestamp(float(value), timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return "Unknown (bid/ask may be stale)"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY).strftime("%Y-%m-%d %I:%M %p ET")


def age_hours(value):
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromtimestamp(float(value), timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
    except (ValueError, TypeError, OverflowError, OSError):
        return None



def daily_direction(current_price, previous_close, reference_source="Provider previous close"):
    """Daily move compared with PRIOR SESSION close, not option P/L.

    An unavailable prior close produces Unknown, never a guessed direction.
    """
    last, previous = finite(current_price), finite(previous_close)
    empty = {"previous_close": None, "change_dollars": None,
             "change_percent": None, "direction": "Unknown", "change_source": reference_source}
    if last is None or previous is None or last <= 0 or previous <= 0:
        return empty
    move = last - previous
    pct = move / previous * 100
    direction = "Up" if pct > 0.005 else "Down" if pct < -0.005 else "Flat"
    return {"previous_close": round(previous, 4), "change_dollars": round(move, 4),
            "change_percent": round(pct, 3), "direction": direction,
            "change_source": reference_source}


def choose_previous_close(record, spot=None):
    """Use only unambiguous previous-session fields, NOT today's 'close'."""
    for key in ("prevclose", "prev_close", "previous_close", "previousClose",
                "prior_close", "priorClose", "yesterday_close"):
        previous = finite(record.get(key))
        if previous is not None and previous > 0:
            return previous
    # A quoted absolute daily price change can also establish prior close.
    for key in ("change", "price_change", "net_change"):
        change = finite(record.get(key))
        if change is not None and spot is not None and spot - change > 0:
            return spot - change
    return None


@st.cache_data(ttl=600, show_spinner=False)
def yahoo_previous_close(ticker):
    """Independent best-effort fallback for a Cboe quote without a prior close.

    FastInfo.previous_close is preferred; daily history is a fallback only.
    Yahoo may be delayed or unavailable, so caller labels this cross-source.
    """
    import yfinance as yf
    inst = yf.Ticker(ticker)
    try:
        ref = finite(inst.fast_info.get("previous_close"))
        if ref is not None and ref > 0:
            return ref
    except Exception:
        pass
    history = inst.history(period="8d", interval="1d", auto_adjust=False)
    if history is None or history.empty:
        return None
    closes = history["Close"].dropna()
    if closes.empty:
        return None
    day = closes.index[-1].date()
    # If today's row is already in Yahoo history, it is often intraday,
    # so yesterday's row is the prior session close.
    if day == datetime.now(NY).date():
        return finite(closes.iloc[-2]) if len(closes) >= 2 else None
    return finite(closes.iloc[-1])


@st.cache_data(ttl=900, show_spinner=False)
def yahoo_session_history(ticker):
    """Best-effort DAILY close history for trend comparisons (not live quotes).

    Four months normally covers at least 30 US trading sessions. The return
    is a Pandas dataframe and can be independently cached per ticker.
    """
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="4mo", interval="1d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError("Yahoo did not return historical closing prices")
    return hist


def trend_direction(pct):
    """Observed return direction, not predictive trend strength."""
    if pct is None or not math.isfinite(pct):
        return "Unknown"
    if pct > 0.05:
        return "Up"
    if pct < -0.05:
        return "Down"
    return "Flat"


def trend_pattern(day_direction, five_direction, thirty_direction):
    """Identify a pullback versus a broader climb and the opposite pattern."""
    if thirty_direction == "Up" and day_direction == "Down":
        return "Red today, up over 30 sessions"
    if thirty_direction == "Down" and day_direction == "Up":
        return "Green today, down over 30 sessions"
    if five_direction == "Down" and thirty_direction == "Up":
        return "5-session pullback in 30-session rise"
    if five_direction == "Up" and thirty_direction == "Down":
        return "5-session rebound in 30-session decline"
    if five_direction == "Up" and thirty_direction == "Up":
        return "Up over 5 and 30 sessions"
    if five_direction == "Down" and thirty_direction == "Down":
        return "Down over 5 and 30 sessions"
    if five_direction == "Unknown" or thirty_direction == "Unknown":
        return "Trend data incomplete"
    return "Mixed / mostly flat"


def reference_anchor(quote_ts, history_dates, today=None):
    """Determine observation session date, avoiding future Yahoo candles.

    If the feed does not give a usable stock quote timestamp, prefer today's
    session on weekdays but the last historical session on weekends/holidays.
    Comparisons are tagged cross-source in the report, not claimed as live.
    """
    now = today or datetime.now(NY).date()
    anchor = now
    try:
        if isinstance(quote_ts, datetime):
            anchor = (quote_ts if quote_ts.tzinfo else quote_ts.replace(tzinfo=timezone.utc)).astimezone(NY).date()
        elif quote_ts is not None:
            anchor = datetime.fromtimestamp(float(quote_ts), timezone.utc).astimezone(NY).date()
    except (ValueError, TypeError, OverflowError, OSError):
        pass
    if quote_ts is None and now.weekday() >= 5 and history_dates:
        anchor = min(now, history_dates[-1])
    return min(anchor, now)


def compute_historical_trends(spot, history, quote_ts=None, today=None, day_direction="Unknown"):
    """Compare observed price to closes 5/30 PRIOR trading sessions back.

    Never substitute missing price history with assumed returns. Spot is
    taken from the selected option quote provider, while history is Yahoo.
    A source mismatch or delayed spot can distort these comparisons.
    """
    result = {
        "5d change %": None, "30d change %": None,
        "5d direction": "Unknown", "30d direction": "Unknown",
        "5d reference date": None, "30d reference date": None,
        "Trend pattern": "Trend data incomplete",
        "Trend source": "Yahoo historical closes; current quote from selected provider (may be cross-source)",
    }
    px = finite(spot)
    if px is None or px <= 0 or history is None or history.empty or "Close" not in history.columns:
        return result
    series = history["Close"].dropna()
    dates_and_prices = []
    for stamp, close in series.items():
        try:
            day = stamp.date() if hasattr(stamp, "date") else date.fromisoformat(str(stamp)[:10])
            price = finite(close)
            if price is not None and price > 0:
                dates_and_prices.append((day, price))
        except (ValueError, TypeError):
            pass
    if not dates_and_prices:
        return result
    # Collapse duplicate dates and order chronologically.
    daily = sorted(dict(dates_and_prices).items())
    anchor = reference_anchor(quote_ts, [d for d, _ in daily], today=today)
    earlier = [(d, p) for d, p in daily if d < anchor]
    # A feed with weeks-old historical closes is not reliable for a fresh scan.
    if not earlier or (anchor - earlier[-1][0]).days > 8:
        return result
    for window in (5, 30):
        if len(earlier) < window:
            continue
        ref_day, ref_close = earlier[-window]
        pct = 100.0 * (px / ref_close - 1)
        result[f"{window}d change %"] = round(pct, 2)
        result[f"{window}d direction"] = trend_direction(pct)
        result[f"{window}d reference date"] = ref_day.isoformat()
    result["Trend pattern"] = trend_pattern(day_direction, result["5d direction"], result["30d direction"])
    return result


def match_trend_filter(frame, choice):
    """Dataframe trend filter, independent of any model risk score."""
    if choice == "All trends" or frame.empty:
        return frame
    if choice == "5d up":
        return frame[frame["5d direction"] == "Up"]
    if choice == "5d down":
        return frame[frame["5d direction"] == "Down"]
    if choice == "30d up":
        return frame[frame["30d direction"] == "Up"]
    if choice == "30d down":
        return frame[frame["30d direction"] == "Down"]
    if choice == "5d and 30d up":
        return frame[(frame["5d direction"] == "Up") & (frame["30d direction"] == "Up")]
    if choice == "5d and 30d down":
        return frame[(frame["5d direction"] == "Down") & (frame["30d direction"] == "Down")]
    if choice == "5d pullback, 30d up":
        return frame[(frame["5d direction"] == "Down") & (frame["30d direction"] == "Up")]
    if choice == "5d bounce, 30d down":
        return frame[(frame["5d direction"] == "Up") & (frame["30d direction"] == "Down")]
    return frame

def safe_json_get(url, params=None, headers=None):
    r = requests.get(url, params=params, headers=headers or REQUEST_HEADERS, timeout=13)
    if r.status_code == 401:
        raise RuntimeError("API token rejected (401); check Streamlit Secrets")
    if r.status_code == 403:
        raise RuntimeError("Provider denied request (403); permissions or throttling")
    if r.status_code == 429:
        raise RuntimeError("Rate limited (429); try a smaller scan later")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def web_discover_symbols():
    """Best-effort PUBLIC Nasdaq stock screen. Never described as exhaustive."""
    data = safe_json_get(NASDAQ_SCREEN, params={"tableonly": "true", "limit": "5000", "offset": "0", "download": "true"})
    body = data.get("data") or {}
    rows = (body.get("table") or {}).get("rows") or body.get("rows") or []
    if not rows:
        raise RuntimeError("Nasdaq public screener returned no rows")
    picks = []
    for r in rows:
        ticker = str(r.get("symbol") or "").upper().strip()
        price = finite(r.get("lastsale") or r.get("lastSale") or r.get("last"))
        volume = finite(r.get("volume"), 0)
        market_cap = finite(r.get("marketCap"), 0)
        if not re.fullmatch(r"[A-Z]{1,5}", ticker):
            continue
        if price is None or not 3 <= price <= 18 or volume < 500_000 or market_cap < 200_000_000:
            continue
        if ticker in LEVERAGED_FUNDS:
            continue
        picks.append((ticker, volume, price))
    picks.sort(key=lambda p: p[1], reverse=True)
    return picks


def get_secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except (FileNotFoundError, KeyError, AttributeError, RuntimeError):
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def cboe_data(ticker):
    """Cboe publicly displayed delayed options quotes; availability is best effort.

    Only use for personal research subject to the provider's terms. This
    endpoint is not a guaranteed/contracted market-data API.
    """
    if not re.fullmatch(r"[A-Z]{1,5}", ticker):
        raise ValueError("Invalid stock symbol")
    payload = safe_json_get(CBOE_CHAIN_URL.format(symbol=ticker), headers={
        "User-Agent": REQUEST_HEADERS["User-Agent"], "Accept": "application/json",
    })
    data = payload.get("data") or {}
    if not data.get("options") or not isinstance(data.get("options"), list):
        raise RuntimeError("Cboe did not return an option chain for this symbol")
    data["_feed_time"] = normalize_feed_time(payload.get("timestamp"))
    return data


class CboeDelayed:
    name = "Cboe public delayed quotes · typically at least 15-minute delay"
    def __init__(self):
        self.cache = {}

    def item(self, ticker):
        if ticker not in self.cache:
            self.cache[ticker] = cboe_data(ticker)
        return self.cache[ticker]

    def spot(self, ticker):
        data = self.item(ticker)
        spot = finite(data.get("current_price")) or finite(data.get("close"))
        if not spot or spot <= 0:
            raise RuntimeError("Cboe underlying quote is missing")
        # Only use Cboe snapshot metadata, not option last-trade timestamps.
        return spot, data.get("_feed_time")

    def daily_move(self, ticker, spot):
        data = self.item(ticker)
        previous = choose_previous_close(data, spot)
        return daily_direction(spot, previous, "Cboe stock snapshot")

    def expiries(self, ticker):
        data = self.item(ticker)
        found = set()
        for option in data["options"]:
            match = re.fullmatch(r"([A-Z]{1,6})(\d{6})([CP])(\d{8})", str(option.get("option", "")))
            if not match or match.group(1) != ticker:
                continue  # Ignore adjusted / mismatched options roots.
            yy, mm, dd = match.group(2)[:2], match.group(2)[2:4], match.group(2)[4:6]
            try:
                found.add(date(2000 + int(yy), int(mm), int(dd)).isoformat())
            except ValueError:
                pass
        return sorted(found)

    def chain(self, ticker, expiry):
        records = []
        for x in self.item(ticker)["options"]:
            sym = str(x.get("option") or "")
            match = re.fullmatch(r"([A-Z]{1,6})(\d{6})([CP])(\d{8})", sym)
            if not match or match.group(1) != ticker:
                continue
            try:
                option_date = date(2000 + int(match.group(2)[:2]),
                                   int(match.group(2)[2:4]), int(match.group(2)[4:6])).isoformat()
            except ValueError:
                continue
            if option_date != expiry:
                continue
            records.append({
                "side": "call" if match.group(3) == "C" else "put",
                "strike": int(match.group(4)) / 1000,
                "bid": x.get("bid"), "ask": x.get("ask"),
                "oi": x.get("open_interest"), "volume": x.get("volume"),
                "iv": x.get("iv"), "delta": x.get("delta"),
                # Cboe 'last_trade_time' is a last-trade date, not the bid time.
                "quote_ts": self.item(ticker).get("_feed_time"), "contract_size": 100,
                "root": match.group(1), "contract": sym,
            })
        return records


class TradierSandbox:
    name = "Tradier sandbox · quotes delayed about 15 minutes"
    def __init__(self, token):
        self.token = token

    def get(self, endpoint, params):
        return safe_json_get(
            TRADIER_SANDBOX + endpoint, params=params,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
        )

    def spot(self, ticker):
        data = self.get("/markets/quotes", {"symbols": ticker})
        q = (data.get("quotes") or {}).get("quote") or {}
        if isinstance(q, list):
            q = q[0] if q else {}
        px = finite(q.get("last")) or finite(q.get("close"))
        if not px or px <= 0:
            raise RuntimeError("Missing underlying stock quote")
        return px, q.get("trade_date")

    def daily_move(self, ticker, spot):
        data = self.get("/markets/quotes", {"symbols": ticker})
        q = (data.get("quotes") or {}).get("quote") or {}
        if isinstance(q, list):
            q = q[0] if q else {}
        return daily_direction(spot, choose_previous_close(q, spot), "Tradier stock quote")

    def expiries(self, ticker):
        data = self.get("/markets/options/expirations", {"symbol": ticker, "includeAllRoots": "false"})
        exp = (data.get("expirations") or {}).get("date") or []
        return [str(v) for v in (exp if isinstance(exp, list) else [exp])]

    def chain(self, ticker, expiry):
        data = self.get("/markets/options/chains", {"symbol": ticker, "expiration": expiry, "greeks": "true"})
        opts = (data.get("options") or {}).get("option") or []
        if isinstance(opts, dict):
            opts = [opts]
        records = []
        for x in opts:
            greeks = x.get("greeks") or {}
            records.append({
                "side": str(x.get("option_type") or "").lower(),
                "strike": x.get("strike"), "bid": x.get("bid"), "ask": x.get("ask"),
                "oi": x.get("open_interest"), "volume": x.get("volume"),
                "iv": greeks.get("mid_iv"), "delta": greeks.get("delta"),
                "quote_ts": x.get("bid_date") or x.get("ask_date"),
                "contract_size": x.get("contract_size", 100),
                "root": x.get("root_symbol") or ticker,
                "contract": x.get("symbol", ""),
            })
        return records


class YahooBestEffort:
    name = "Yahoo Finance · free best-effort, delay/availability not guaranteed"
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self.tickers = {}

    def item(self, ticker):
        if ticker not in self.tickers:
            self.tickers[ticker] = self.yf.Ticker(ticker)
        return self.tickers[ticker]

    def spot(self, ticker):
        t = self.item(ticker)
        px = finite(t.fast_info.get("last_price"))
        if not px or px <= 0:
            hist = t.history(period="5d")
            px = finite(hist["Close"].iloc[-1]) if not hist.empty else None
        if not px or px <= 0:
            raise RuntimeError("Yahoo did not provide a usable stock quote")
        return px, None  # Stock quote timestamp not reliably known.

    def daily_move(self, ticker, spot):
        previous = yahoo_previous_close(ticker)
        return daily_direction(spot, previous, "Yahoo prior session close")

    def expiries(self, ticker):
        return list(self.item(ticker).options or [])

    def chain(self, ticker, expiry):
        data = self.item(ticker).option_chain(expiry)
        result = []
        for side, df in [("call", data.calls), ("put", data.puts)]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                # lastTradeDate is LAST TRADE, NOT bid/ask timestamp; don't mislabel.
                result.append({
                    "side": side, "strike": row.get("strike"), "bid": row.get("bid"),
                    "ask": row.get("ask"), "oi": row.get("openInterest"),
                    "volume": row.get("volume"), "iv": row.get("impliedVolatility"),
                    "delta": None, "quote_ts": None, "contract_size": 100,
                    "root": ticker, "contract": row.get("contractSymbol", ""),
                })
        return result


def score_trade(pop, roi, oi, volume, spread_ratio, buffer_pct, iv, share_price):
    """Heuristic score, NOT a calibrated success probability."""
    pop_score = 0.35 if pop is None else min(pop, 0.88) / 0.88
    income = min(max(roi, 0) / 0.10, 1)
    liquidity = 0.50 * min(oi / 1500, 1) + 0.15 * min(volume / 100, 1) + 0.35 * max(0, 1 - spread_ratio / 0.25)
    cushion = min(buffer_pct / 0.18, 1)
    volatility = 0.15 if iv is None else 1 if iv <= .5 else .7 if iv <= .85 else .35
    price_penalty = 10 if share_price < 4 else 0
    return round(max(0, min(100, 33 * pop_score + 25 * income + 20 * liquidity + 12 * cushion + 10 * volatility - price_penalty)), 1)


def build_trade(ticker, spot, expiry, quote, budget, min_premium, min_oi, spread_limit,
                min_dte, max_dte, allow_iv_assumption=False, assumed_iv=.65, max_quote_hours=120):
    side = str(quote.get("side") or "").lower()
    if side not in ("put", "call"):
        return None, "Not a put/call"
    try:
        dte = (date.fromisoformat(expiry) - datetime.now(NY).date()).days
    except (TypeError, ValueError):
        return None, "Invalid expiry"
    if not min_dte <= dte <= max_dte:
        return None, "Outside expiration window"
    strike, bid, ask = finite(quote.get("strike")), finite(quote.get("bid")), finite(quote.get("ask"))
    if not (strike and strike > 0 and bid and bid > 0 and ask and ask >= bid and spot > 0):
        return None, "Missing usable two-sided bid/ask"
    contract_size = finite(quote.get("contract_size"), 100)
    if contract_size != 100 or str(quote.get("root", ticker)).upper() != ticker.upper():
        return None, "Adjusted or nonstandard option skipped"
    premium = 100 * bid  # Model selling at BID, not midpoint or last trade.
    if premium < min_premium:
        return None, "Premium below minimum"
    oi = max(0, int(finite(quote.get("oi"), 0)))
    volume = max(0, int(finite(quote.get("volume"), 0)))
    if oi < min_oi:
        return None, "Open interest too low"
    spread = (ask - bid) / ((ask + bid) / 2)
    if spread > spread_limit:
        return None, "Bid/ask spread too wide"
    age = age_hours(quote.get("quote_ts"))
    if age is not None and age > max_quote_hours:
        return None, "Stale option quote"
    if side == "put":
        capital = strike * 100
        if capital > budget:
            return None, "Put collateral exceeds per-trade limit"
        if not 0.65 <= strike / spot <= 1.03:
            return None, "Put strike too far ITM/OTM"
        breakeven = strike - bid
        cap_profit = premium
        max_loss = max(0, capital - premium)
        roi = cap_profit / capital
    else:
        capital = spot * 100
        if capital > budget:
            return None, "Covered-call share cost exceeds limit"
        if not 0.95 <= strike / spot <= 1.25:
            return None, "Call strike too far ITM/OTM"
        breakeven = spot - bid
        cap_profit = 100 * (strike - spot + bid)
        if cap_profit <= 0:
            return None, "Negative capped return"
        max_loss = max(0, capital - premium)
        roi = cap_profit / capital
    if breakeven <= 0:
        return None, "Invalid break-even"
    iv = parse_iv(quote.get("iv"))
    volatility = iv if iv is not None else assumed_iv if allow_iv_assumption else None
    pop = prob_above(spot, breakeven, volatility, dte)
    p_above_strike = prob_above(spot, strike, volatility, dte)
    p_assign = None if p_above_strike is None else (1 - p_above_strike if side == "put" else p_above_strike)
    buffer = max(0, spot - breakeven) / spot
    score = score_trade(pop, roi, oi, volume, spread, buffer, volatility, spot)
    return {
        "Score / 100": score, "Ticker": ticker,
        "Strategy": "Cash-secured put" if side == "put" else "Covered call",
        "Stock $": round(spot, 2), "Expiration": expiry, "DTE": dte,
        "Strike $": round(strike, 2), "Bid $": round(bid, 2), "Ask $": round(ask, 2),
        "Premium $": round(premium, 2), "Cash needed $": round(capital, 2),
        "Break-even $": round(breakeven, 2), "Max profit $": round(cap_profit, 2),
        "Max loss $": round(max_loss, 2), "Max return %": round(100 * roi, 2),
        "Model POP %": None if pop is None else round(100 * pop, 1),
        "Model assign %": None if p_assign is None else round(100 * p_assign, 1),
        "IV %": None if volatility is None else round(100 * volatility, 1),
        "IV basis": "Quote IV" if iv is not None else "User assumption" if allow_iv_assumption else "Not available",
        "Open interest": oi, "Volume": volume,
        "Spread %": round(spread * 100, 1), "Quote time ET": timestamp_to_et(quote.get("quote_ts")),
        "Quote age hours": None if age is None else round(age, 1),
        "Contract": quote.get("contract", ""),
        "Caution": "Earnings/corporate events not verified; downside loss possible",
    }, "OK"


def selection_from_discovery(web_items, universe_limit):
    """Combine curated liquid watchlist + diversified public screen, not whole market."""
    ticker_list = []
    for sym in STARTER_UNIVERSE:
        if sym not in ticker_list and sym not in LEVERAGED_FUNDS:
            ticker_list.append(sym)
    # Blend in online screener results so report scans stocks not known ahead of time.
    if web_items:
        diversified = []
        for low, high in [(3, 6), (6, 10), (10, 18)]:
            diversified += [sym for sym, _, px in web_items if low <= px < high][:max(3, universe_limit // 3)]
        result = []
        for pair in zip(ticker_list, diversified + ticker_list):
            for sym in pair:
                if sym not in result:
                    result.append(sym)
        for sym in diversified:
            if sym not in result:
                result.append(sym)
        return result[:universe_limit]
    return ticker_list[:universe_limit]


def main():
    st.set_page_config(page_title="Automatic Options Opportunity Report", page_icon="📈", layout="wide")
    st.title("📈 Automatic Options Opportunity Report")
    st.caption("One tap searches candidate stocks, checks puts and calls, and ranks trade opportunities. Orders stay in Webull.")

    sandbox_token = get_secret("TRADIER_SANDBOX_TOKEN")
    providers = ["Cboe delayed option chains (no account; best effort)", "Yahoo Finance (no account; less reliable)"]
    if sandbox_token:
        providers.insert(0, "Tradier sandbox (about 15-minute delayed quotes)")
    provider_choice = st.selectbox("Market-data source", providers, index=0)
    st.warning("**Data freshness:** Cboe public quotes and Tradier sandbox are delayed (commonly at least 15 minutes). Yahoo Finance quote timing and availability vary. Free feeds can fail or be stale. Verify actual bid/ask and stock price in Webull before ordering.")
    if not sandbox_token:
        with st.expander("Want a more reliable free delayed feed?"):
            st.markdown("If you have a Tradier *sandbox* token, add `TRADIER_SANDBOX_TOKEN = \"...\"` to **Streamlit Community Cloud → your app → Settings → Secrets**. Keep it out of GitHub. Otherwise the app uses free Yahoo Finance best effort. Tradier account/API access is required; no Webull API connection is needed.")

    with st.expander("Account, risk, and scan settings", expanded=True):
        left, right = st.columns(2)
        with left:
            account = st.number_input("Account equity ($)", 100, 500000, 4300, 100)
            budget = st.number_input("Max per trade ($)", 100, int(account), min(1000, int(account)), 100)
            strategy = st.selectbox("Trades to rank", ["Both", "Cash-secured puts", "Covered calls"])
            max_stocks = st.slider("Number of stocks to investigate", 6, 40, 16, 2,
                                   help="More tickers use more provider API calls and take longer. This is not a whole-market scan.")
        with right:
            min_dte, max_dte = st.slider("Expiration range (days)", 7, 90, (21, 45))
            expiration_count = st.select_slider("Expirations checked per ticker", options=[1, 2, 3], value=2)
            min_premium = st.number_input("Minimum premium per contract ($)", 0, 500, 30, 5)
            min_oi = st.number_input("Minimum open interest", 0, 10000, 100, 50)
            max_spread = st.slider("Maximum bid/ask spread (%)", 5, 100, 35, 5) / 100
        use_web = st.checkbox("Automatically discover additional low-priced stocks from Nasdaq's public screener", True)
        yahoo_move_fallback = st.checkbox(
            "Try Yahoo previous-close data when Cboe does not provide it", True,
            help="May add time to scans. When stock price and previous close are from different sources, the report labels the estimate and may be out of sync.")
        history_enabled = st.checkbox(
            "Compare 5-session and 30-session price moves", True,
            help="Fetches daily closing prices from Yahoo for each scanned ticker. More requests can slow down the report; missing history is displayed as unavailable."
        )
        assumption = st.checkbox("Show probability estimates if option IV is missing (uses assumed IV)", False)
        assumed_iv = st.slider("Assumed annual IV (%)", 15, 200, 65, 5) / 100 if assumption else .65
        extra_text = st.text_input("Optional extra tickers (not required)", "", placeholder="e.g. SOFI, BB")
        st.caption("Default scans a diversified subset of active stocks; it does NOT search every listed US option. Earnings and other major events are not yet automatically checked.")

    if st.button("🔍 Discover stocks & generate ranked report", type="primary", use_container_width=True):
        today = datetime.now(NY).date()
        source = (TradierSandbox(sandbox_token) if provider_choice.startswith("Tradier") and sandbox_token
                  else CboeDelayed() if provider_choice.startswith("Cboe") else YahooBestEffort())
        diagnostics = []
        online_symbols = []
        if use_web:
            try:
                online_symbols = web_discover_symbols()
                st.info(f"Nasdaq web discovery returned {len(online_symbols)} eligible symbols before batching.")
            except Exception as exc:
                st.warning(f"Nasdaq stock discovery unavailable ({type(exc).__name__}). Scanning the built-in liquid-stock universe instead.")
        symbols = selection_from_discovery(online_symbols, max_stocks)
        extras = [s for s in re.split(r"[\s,;]+", extra_text.upper()) if re.fullmatch(r"[A-Z]{1,5}", s)]
        for extra in reversed(extras):
            if extra not in LEVERAGED_FUNDS:
                symbols.insert(0, extra)
        symbols = list(dict.fromkeys(symbols))[:max_stocks]
        st.caption("Candidates queued: " + ", ".join(symbols) + " · Source: " + source.name)
        results = []
        stock_moves = []
        progress = st.progress(0, text="Looking up stock prices and options...")
        for idx, ticker in enumerate(symbols, 1):
            scanned = 0
            excluded = {}
            note = ""
            try:
                spot, underlying_ts = source.spot(ticker)
                if spot < 2 or spot > 18:
                    raise ValueError("Stock outside $2–$18 discovery range")
                try:
                    movement = source.daily_move(ticker, spot)
                except Exception:
                    movement = daily_direction(spot, None, "Daily previous-close lookup unavailable")
                if (movement["direction"] == "Unknown" and yahoo_move_fallback
                        and not isinstance(source, YahooBestEffort)):
                    try:
                        previous = yahoo_previous_close(ticker)
                        movement = daily_direction(spot, previous, "Yahoo prior close + provider spot (cross-source)")
                    except Exception:
                        pass  # Do not hide valid options because a direction lookup failed.
                historical = compute_historical_trends(spot, None, underlying_ts, day_direction=movement["direction"])
                if history_enabled:
                    try:
                        historical = compute_historical_trends(
                            spot, yahoo_session_history(ticker), underlying_ts,
                            day_direction=movement["direction"])
                    except Exception:
                        # Historical lookups must not block an otherwise valid option scan.
                        pass
                stock_moves.append({
                    "Ticker": ticker, "Stock $": round(spot, 2),
                    "Previous close $": movement["previous_close"],
                    "Day change $": None if movement["change_dollars"] is None else round(movement["change_dollars"], 2),
                    "Day change %": None if movement["change_percent"] is None else round(movement["change_percent"], 2),
                    "Direction": movement["direction"], "Change reference": movement["change_source"],
                    "Stock quote time ET": timestamp_to_et(underlying_ts),
                    **historical,
                })
                exps = []
                for exp in source.expiries(ticker):
                    try:
                        d = (date.fromisoformat(str(exp)[:10]) - today).days
                        if min_dte <= d <= max_dte:
                            exps.append((abs(d - (min_dte + max_dte) / 2), str(exp)[:10]))
                    except ValueError:
                        pass
                if not exps:
                    raise ValueError("No usable expirations returned in selected window (may be feed issue)")
                exps.sort()
                for _, exp in exps[:expiration_count]:
                    for item in source.chain(ticker, exp):
                        side = item.get("side")
                        if strategy == "Cash-secured puts" and side != "put":
                            continue
                        if strategy == "Covered calls" and side != "call":
                            continue
                        trade, reason = build_trade(
                            ticker, spot, exp, item, float(budget), float(min_premium),
                            int(min_oi), float(max_spread), min_dte, max_dte,
                            assumption, assumed_iv)
                        if trade:
                            trade["Data source"] = source.name
                            trade["Underlying time ET"] = timestamp_to_et(underlying_ts)
                            trade["Previous close $"] = movement["previous_close"]
                            trade["Day change $"] = (None if movement["change_dollars"] is None else round(movement["change_dollars"], 2))
                            trade["Day change %"] = (None if movement["change_percent"] is None else round(movement["change_percent"], 2))
                            trade["Day direction"] = movement["direction"]
                            trade["Change reference"] = movement["change_source"]
                            trade.update(historical)
                            if movement["change_percent"] is not None and movement["change_percent"] < -5:
                                trade["Caution"] += "; stock down >5% versus previous close"
                            results.append(trade)
                            scanned += 1
                        else:
                            excluded[reason] = excluded.get(reason, 0) + 1
                if scanned:
                    note = f"{scanned} qualifying contracts"
                elif excluded:
                    top_reason = max(excluded, key=excluded.get)
                    note = f"0 matches — most common exclusion: {top_reason} ({excluded[top_reason]})"
                else:
                    note = "Option chain empty for selected dates"
            except Exception as exc:
                note = f"{type(exc).__name__}: {str(exc)[:160]}"
            diagnostics.append({"Ticker": ticker, "Result": note})
            progress.progress(idx / len(symbols), text=f"Checked {idx}/{len(symbols)} stocks: {ticker}")
        progress.empty()
        st.session_state.report = results
        st.session_state.stock_moves = stock_moves
        st.session_state.diagnostics = diagnostics
        st.session_state.report_source = source.name
        st.session_state.report_discovery = "Nasdaq public screener plus starter universe" if online_symbols else "Starter universe only (online discovery failed or disabled)"
        st.session_state.report_time = datetime.now(NY).strftime("%Y-%m-%d %I:%M %p ET")

    if "report" in st.session_state:
        st.subheader("🏆 Report: best potential trades")
        st.caption(f"Run: {st.session_state.report_time} | Discovery: {st.session_state.report_discovery} | Option quotes: {st.session_state.report_source}")
        all_moves = st.session_state.get("stock_moves", [])
        if all_moves:
            direction_counts = {d: sum(x["Direction"] == d for x in all_moves)
                                for d in ("Up", "Down", "Flat", "Unknown")}
            a, b, c, d = st.columns(4)
            a.metric("🟢 Up", direction_counts["Up"])
            b.metric("🔴 Down", direction_counts["Down"])
            c.metric("➖ Flat", direction_counts["Flat"])
            d.metric("❔ Unknown", direction_counts["Unknown"])
        view = st.selectbox("Show trades for stocks that are", ["All directions", "Up", "Down", "Flat", "Unknown"],
                            help="Filters the report results without changing the risk score or re-running market requests.")
        trend_view = st.selectbox(
            "5-day / 30-day trend filter",
            ["All trends", "5d up", "5d down", "30d up", "30d down",
             "5d and 30d up", "5d and 30d down", "5d pullback, 30d up", "5d bounce, 30d down"],
            help="Filter by past price movement; this does not mean a trade has a higher probability of profit."
        )
        rows = st.session_state.report
        if rows:
            df = pd.DataFrame(rows).sort_values(["Score / 100", "Max return %", "Model POP %"], ascending=False, na_position="last")
            if view != "All directions":
                df = df[df["Day direction"] == view]
            df = match_trend_filter(df, trend_view)
            top = df.drop_duplicates("Ticker").head(10)
            st.metric("Matching contracts (daily + multi-day filters)", len(df))
            st.metric("Matching tickers", int(df.Ticker.nunique()))
            tabs = st.tabs(["🏆 Top ten", "📑 All qualifying", "📈 Stock movement", "🔎 Scan diagnostics"])
            key_cols = ["Score / 100", "Ticker", "Strategy", "Stock $", "Day direction", "Day change %", "5d change %", "30d change %", "Trend pattern", "Expiration", "Strike $", "Bid $", "Ask $", "Premium $", "Cash needed $", "Break-even $", "Max return %", "Model POP %", "Model assign %", "Open interest", "Spread %", "Quote time ET"]
            with tabs[0]:
                if top.empty:
                    st.info("No qualified trades match this Up/Down filter. Choose 'All directions' or run a wider scan.")
                for rank, (_, candidate) in enumerate(top.iterrows(), 1):
                    headline = (f"#{rank} · {candidate['Ticker']} · {candidate['Strategy']} · "
                                f"Score {candidate['Score / 100']:.0f}/100")
                    with st.expander(headline, expanded=rank <= 3):
                        trend = candidate["Day direction"]
                        trend_emoji = {"Up": "🟢⬆️", "Down": "🔴⬇️", "Flat": "➖", "Unknown": "❔"}.get(trend, "❔")
                        change = candidate["Day change %"]
                        trend_text = (f"{float(change):+.2f}%" if pd.notna(change) else "unavailable")
                        st.write(f"**{candidate['Expiration']} · ${candidate['Strike $']:.2f} strike** "
                                 f"· stock ${candidate['Stock $']:.2f}")
                        st.write(f"{trend_emoji} **Stock {trend.lower()} versus previous close: {trend_text}**")
                        st.caption("Price-change reference: " + str(candidate["Change reference"]))
                        five = candidate.get("5d change %")
                        thirty = candidate.get("30d change %")
                        five_text = f"{float(five):+.2f}%" if pd.notna(five) else "N/A"
                        thirty_text = f"{float(thirty):+.2f}%" if pd.notna(thirty) else "N/A"
                        st.write(f"**5 sessions:** {five_text} · **30 sessions:** {thirty_text}")
                        st.caption(f"Price context: {candidate.get('Trend pattern', 'Unavailable')}"
                                   " · Historical comparison from Yahoo, may differ in timestamp from option quote")
                        a, b = st.columns(2)
                        a.metric("Premium at bid", f"${candidate['Premium $']:,.0f}")
                        b.metric("Cash required", f"${candidate['Cash needed $']:,.0f}")
                        a.metric("Max return if capped", f"{candidate['Max return %']:.1f}%")
                        b.metric("Break-even", f"${candidate['Break-even $']:.2f}")
                        prob = candidate['Model POP %']
                        st.caption(f"Model profit probability: {prob:.1f}%" if pd.notna(prob) else
                                   "Model profit probability: unavailable (option IV missing)")
                        st.caption(f"Option bid/ask: ${candidate['Bid $']:.2f} / ${candidate['Ask $']:.2f} · "
                                   f"OI: {int(candidate['Open interest'])} · Quote: {candidate['Quote time ET']}")
                st.caption("The 0–100 rating is a heuristic, **not** a probability that a stock will rise or a trade will succeed.")
            with tabs[1]:
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download ranked report (CSV / Excel)", df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"ranked_options_report_{datetime.now(NY).date().isoformat()}.csv", mime="text/csv", use_container_width=True)
            with tabs[2]:
                movement_df = pd.DataFrame(all_moves)
                if view != "All directions":
                    movement_df = movement_df[movement_df["Direction"] == view]
                movement_df = match_trend_filter(movement_df, trend_view)
                st.dataframe(movement_df, hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download stock up/down report (CSV)",
                                   movement_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"stock_movement_{datetime.now(NY).date().isoformat()}.csv",
                                   mime="text/csv", use_container_width=True)
                st.caption("Daily change is versus the prior session close. 5d/30d compare the selected source's "
                           "stock quote against Yahoo closes from 5 and 30 prior trading sessions. "
                           "Different data sources and stale stock quotes can distort the comparisons; check Webull.")
            with tabs[3]:
                st.dataframe(pd.DataFrame(st.session_state.diagnostics), hide_index=True, use_container_width=True)
        else:
            st.warning("No qualifying quotes came back. Review the diagnostics below. If a free data feed reports missing dates, try another listed provider; do not assume the stock has no options.")
            if all_moves:
                st.subheader("📈 Stock movement (even though no options qualified)")
                movement_df = pd.DataFrame(all_moves)
                if view != "All directions":
                    movement_df = movement_df[movement_df["Direction"] == view]
                movement_df = match_trend_filter(movement_df, trend_view)
                st.dataframe(movement_df, hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download stock trend report (CSV)",
                                   movement_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"stock_trends_{datetime.now(NY).date().isoformat()}.csv",
                                   mime="text/csv", use_container_width=True)
            st.dataframe(pd.DataFrame(st.session_state.diagnostics), hide_index=True, use_container_width=True)

    with st.expander("Important: how probabilities, scores, and risks work"):
        st.markdown("""**Model POP** = probability the stock finishes above the trade's break-even at expiration under a zero-drift lognormal distribution using the option IV. It is *not backtested* and doesn't model earnings jumps, changing volatility or early assignment. If IV is missing, the probability stays blank unless you explicitly opt into a user-assumed IV. **Assignment** refers to ending ITM, not a guarantee of actual exercise behavior.\n\n**Ranking /100** combines modeled POP (33 points), capped return on required capital (25), option liquidity (20), break-even cushion (12), and an IV-risk term (10), with a low-share-price penalty. It is **not** a percentage chance of a good trade. Bids are used for premium and standard 100-share contracts only. Buying shares for a call requires 100 × stock price; cash-secured puts reserve 100 × strike even after premium. Stock losses can far exceed premium received.\n\n**Direction indicators:** green/up and red/down compare reported stock price against the previous trading close. They are **not predictions** and do not change trade-quality scores. For Cboe where previous-close data is missing, a Yahoo previous close may be used with a **cross-source** label. Timing differences can distort daily changes. If neither feed provides a usable prior close, the direction is Unknown rather than guessed.

**5-session / 30-session comparisons:** the app compares the selected provider's stock price with Yahoo Finance closing prices from 5 and 30 *prior trading sessions*, not 5 and 30 calendar days. They are observed price changes, not forecasts. They do not alter modeled profit probability or trade-quality ratings. If history is insufficient or unavailable the corresponding change displays N/A. When sources have different quote times, the percentages may be inaccurate. A five-session dip during a 30-session increase can be labeled a pullback, not automatically a buy signal.

**Watch the limitations:** no complete market coverage; public stock discovery can fail; free Cboe or Yahoo chains may be stale, blocked, or absent; Tradier sandbox is roughly 15 minutes delayed and requires a sandbox token. There is no automatic earnings check. Verify price, option bid/ask, corporate actions, earnings and contract deliverables in Webull before placing any order. This app does not place trades.""")
    st.caption("For personal research use. Do not publicly republish licensed option quotes or API tokens.")


if __name__ == "__main__":
    main()
