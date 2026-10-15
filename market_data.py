"""Stock and options quotes; read-only, no order-routing endpoints.

Production Tradier quotes are real-time to eligible Tradier brokerage users.
Yahoo fallback is convenient but unverified/delayed. This app never fabricates quotes.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo
import math
import re
import requests

ET = ZoneInfo("America/New_York")
VALID_SYMBOL = re.compile(r"^[A-Z][A-Z0-9./-]{0,11}$")


class MarketDataError(RuntimeError):
    pass


def number(x: Any, default: float | None = None) -> float | None:
    try:
        n = float(x)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(x: Any, default: int = 0) -> int:
    n = number(x)
    return max(0, int(n)) if n is not None else default


def validated_symbol(symbol: str) -> str:
    sym = symbol.strip().upper()
    if not VALID_SYMBOL.fullmatch(sym):
        raise MarketDataError("Enter a valid US stock ticker, e.g., BB, HIMZ, or MARA.")
    return sym


def iso_time(raw: Any) -> str | None:
    """Convert provider Unix timestamps into unambiguous ET timestamps."""
    v = number(raw)
    if v is None or v <= 0:
        return None
    try:
        if v > 1e12:
            v /= 1000.0
        return datetime.fromtimestamp(v, timezone.utc).astimezone(ET).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError):
        return None


def call_rows(exp: str, options: list[dict], symbol: str) -> list[dict]:
    """Strictly exclude nonstandard / adjusted deliverables and unusable option quotes."""
    res: list[dict] = []
    for op in options:
        if str(op.get("option_type", "")).lower() != "call":
            continue
        underlying = str(op.get("underlying") or symbol).upper().replace(".", "/")
        root = str(op.get("root_symbol") or symbol).upper().replace(".", "/")
        if underlying != symbol.replace(".", "/") or root != symbol.replace(".", "/"):
            continue  # adjusted series can have a non-standard underlying/deliverable
        qty = integer(op.get("contract_size"), 100)
        if qty != 100:
            continue
        strike = number(op.get("strike"))
        bid = number(op.get("bid"))
        ask = number(op.get("ask"))
        if strike is None or bid is None or ask is None:
            continue
        if strike <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            continue
        greeks = op.get("greeks") or {}
        iv = None
        if isinstance(greeks, dict):
            for k in ("smv_vol", "mid_iv", "bid_iv", "ask_iv"):
                temp = number(greeks.get(k))
                if temp is not None and 0.03 <= temp <= 8.0:
                    iv = temp
                    break
        res.append({
            "expiration": exp,
            "strike": strike, "bid": bid, "ask": ask,
            "volume": integer(op.get("volume")), "oi": integer(op.get("open_interest")),
            "contract": str(op.get("symbol") or ""), "iv": iv,
            "bid_time": iso_time(op.get("bid_date")),
            "ask_time": iso_time(op.get("ask_date")),
            "trade_time": iso_time(op.get("trade_date")),
            "multiplier": qty,
        })
    return res


def tradier_get(token: str, endpoint: str, params: dict, *, sandbox: bool = False, session=None) -> dict:
    if not token or not token.strip():
        raise MarketDataError("Tradier API token is missing. Configure server secrets or choose Yahoo fallback.")
    base = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
    getter = session or requests
    try:
        response = getter.get(
            base + endpoint,
            params=params,
            headers={"Accept": "application/json", "Authorization": "Bearer " + token.strip()},
            timeout=12,
        )
        if response.status_code in (401, 403):
            raise MarketDataError("Tradier authorization failed; verify the API token and account permissions.")
        if response.status_code == 429:
            raise MarketDataError("Tradier rate limit reached. Wait briefly before refreshing.")
        response.raise_for_status()
        payload = response.json()
    except MarketDataError:
        raise
    except (requests.RequestException, ValueError) as exc:
        raise MarketDataError(f"Tradier quote request failed ({type(exc).__name__}).") from exc
    if not isinstance(payload, dict):
        raise MarketDataError("Unexpected Tradier data format.")
    if payload.get("errors") or payload.get("fault"):
        raise MarketDataError("Tradier returned an error. Check your symbol and API permissions.")
    return payload


def tradier_snapshot(ticker: str, min_dte: int, max_dte: int, token: str, sandbox: bool = False) -> dict:
    sym = validated_symbol(ticker)
    q = tradier_get(token, "/markets/quotes", {"symbols": sym}, sandbox=sandbox)
    item = (q.get("quotes") or {}).get("quote") if isinstance(q.get("quotes"), dict) else None
    if isinstance(item, list):
        item = item[0] if item else None
    if not isinstance(item, dict):
        raise MarketDataError("Stock quote not returned for this symbol.")
    ask = number(item.get("ask"))
    last = number(item.get("last"))
    stock_bid = number(item.get("bid"))
    # When there is no valid ask, fall back to last and make it explicit.
    spot = ask if ask is not None and ask > 0 else last
    stock_source = "stock ask" if spot == ask and ask and ask > 0 else "stock last (ask unavailable)"
    if spot is None or spot <= 0:
        raise MarketDataError("Stock price is missing. No purchase estimate can be produced.")

    e = tradier_get(token, "/markets/options/expirations", {"symbol": sym, "includeAllRoots": "false"}, sandbox=sandbox)
    exps = (e.get("expirations") or {}).get("date") if isinstance(e.get("expirations"), dict) else None
    if isinstance(exps, str):
        exps = [exps]
    if not isinstance(exps, list):
        exps = []
    today = datetime.now(ET).date()
    rv = None
    try:
        history = tradier_get(token, "/markets/history", {
            "symbol": sym, "interval": "daily",
            "start": (today - timedelta(days=155)).isoformat(),
            "end": today.isoformat(),
        }, sandbox=sandbox)
        hist_days = (history.get("history") or {}).get("day") if isinstance(history.get("history"), dict) else None
        if isinstance(hist_days, dict):
            hist_days = [hist_days]
        closes = [number(d.get("close")) for d in (hist_days or []) if isinstance(d, dict)]
        closes = [v for v in closes if v is not None and v > 0]
        logs = [math.log(b/a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
        if len(logs) >= 20:
            rv = __import__("statistics").stdev(logs[-60:]) * math.sqrt(252)
    except (MarketDataError, ValueError):
        pass
    selected = []
    for ex in exps:
        try:
            d = date.fromisoformat(str(ex))
        except ValueError:
            continue
        if min_dte <= (d - today).days <= max_dte:
            selected.append(str(ex))
    rows = []
    failures = 0
    for ex in selected[:12]:
        try:
            chain = tradier_get(token, "/markets/options/chains", {"symbol": sym, "expiration": ex, "greeks": "true"}, sandbox=sandbox)
            options = (chain.get("options") or {}).get("option") if isinstance(chain.get("options"), dict) else None
            if isinstance(options, dict):
                options = [options]
            if isinstance(options, list):
                rows.extend(call_rows(ex, options, sym))
        except MarketDataError:
            failures += 1
    if selected and failures == len(selected[:12]):
        raise MarketDataError("All Tradier options-chain requests failed. Try refreshing later.")
    return {
        "ticker": sym, "spot": spot, "stock_last": last, "stock_ask": ask, "stock_bid": stock_bid,
        "stock_source": stock_source,
        "stock_quote_time": iso_time(item.get("ask_date") or item.get("trade_date")),
        "provider": "Tradier sandbox · delayed" if sandbox else "Tradier brokerage · real-time entitlement required",
        "delayed": bool(sandbox), "rows": rows, "rv": rv,
        "fetched_at": datetime.now(ET).isoformat(timespec="seconds"),
        "error_count": failures,
    }


def yahoo_snapshot(ticker: str, min_dte: int, max_dte: int) -> dict:
    """Free fallback. Yahoo option bids/asks can be stale or zero. Never label them live."""
    import yfinance as yf
    import pandas as pd
    sym = validated_symbol(ticker)
    try:
        asset = yf.Ticker(sym)
        hist = asset.history(period="6mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise MarketDataError("Yahoo did not return stock price history.")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise MarketDataError("Yahoo returned an empty closing-price series.")
        last_close = float(closes.iloc[-1])
        quote = None
        try:
            quote = number(asset.fast_info.get("last_price"))
        except Exception:
            pass
        spot = quote if quote is not None and quote > 0 else last_close
        lr = (closes / closes.shift(1)).apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan")).dropna()
        rv = number(lr.tail(60).std() * math.sqrt(252)) if len(lr) >= 15 else None
        expirations = list(asset.options)
        today = datetime.now(ET).date()
        rows: list[dict] = []
        for exp in expirations:
            try:
                dte = (date.fromisoformat(exp) - today).days
            except ValueError:
                continue
            if not min_dte <= dte <= max_dte:
                continue
            try:
                options = asset.option_chain(exp).calls
            except Exception:
                continue
            for _, item in options.iterrows():
                size = str(item.get("contractSize", "REGULAR")).upper()
                if size not in {"REGULAR", "100"}:
                    continue
                strike = number(item.get("strike"))
                bid = number(item.get("bid"))
                ask = number(item.get("ask"))
                if strike is None or bid is None or ask is None or strike <= 0 or bid <= 0 or ask < bid:
                    continue
                opt_sym = str(item.get("contractSymbol", ""))
                # Avoid adjusted deliverables (e.g. HIMZ1) by requiring OCC root exact match.
                if not opt_sym.startswith(sym):
                    continue
                suffix = opt_sym[len(sym):]
                if not re.fullmatch(r"\d{6}C\d{8}", suffix):
                    continue
                iv = number(item.get("impliedVolatility"))
                if iv is None or not 0.03 <= iv <= 8:
                    iv = None
                last_trade = item.get("lastTradeDate")
                trade_time = None
                if hasattr(last_trade, "to_pydatetime"):
                    try:
                        trade_time = last_trade.to_pydatetime().astimezone(ET).isoformat(timespec="seconds")
                    except Exception:
                        pass
                rows.append({
                    "expiration": exp, "strike": strike, "bid": bid, "ask": ask,
                    "oi": integer(item.get("openInterest")), "volume": integer(item.get("volume")),
                    "contract": opt_sym, "iv": iv, "bid_time": None,
                    "ask_time": None, "trade_time": trade_time, "multiplier": 100,
                })
        return {
            "ticker": sym, "spot": spot, "stock_last": spot, "stock_ask": None, "stock_bid": None,
            "stock_source": "Yahoo reported price (not verified ask)",
            "stock_quote_time": str(closes.index[-1].date()) + " (latest daily bar; intraday freshness unknown)",
            "provider": "Yahoo Finance · delayed/unverified",
            "delayed": True, "rows": rows, "rv": rv,
            "fetched_at": datetime.now(ET).isoformat(timespec="seconds"),
            "error_count": 0,
        }
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Yahoo data is temporarily unavailable ({type(exc).__name__}).") from exc
