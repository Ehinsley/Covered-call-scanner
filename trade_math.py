"""Transparent probability model for a 100-share covered call.

Estimated probabilities use a single lognormal distribution at expiration.
This is not a predictive or calibrated recommendation engine.
"""
import math
from statistics import NormalDist

N = NormalDist()


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def tail_above(spot: float, threshold: float, sigma: float, days: int, drift: float=0.0) -> float:
    """P(S_at_expiry > threshold) assuming lognormal dynamics."""
    if spot <= 0 or threshold <= 0 or days < 1 or sigma <= 0:
        raise ValueError("Spot, threshold, volatility and time must be positive.")
    t = days / 365.0
    z = (math.log(threshold / spot) - (drift - sigma * sigma / 2.0)*t) / (sigma * math.sqrt(t))
    return clamp(1.0 - N.cdf(z))


def assess(spot: float, strike: float, bid: float, ask: float, days: int,
           sigma: float, budget: float, open_interest: int=0,
           drift: float=0.0, earnings: bool=False, leveraged: bool=False,
           contract_size: int=100) -> dict:
    if min(spot, strike, bid, sigma) <= 0 or days < 1 or ask < bid or contract_size != 100:
        raise ValueError("Check stock price, strike, call bid/ask, days, volatility and 100-share deliverable.")
    if not (-1.0 <= drift <= 1.0):
        raise ValueError("Unreasonable drift assumption")
    share_cost = spot * contract_size
    credit = bid * contract_size
    breakeven = spot - bid
    max_profit = (strike - spot + bid)*contract_size
    max_return = max_profit / share_cost
    # Payoff at expiration: n * (min(S_T, K) - S_entry + premium)
    if max_profit <= 0 or breakeven >= strike:
        p_profit = 0.0
    elif breakeven <= 0:
        p_profit = 1.0
    else:
        p_profit = tail_above(spot, breakeven, sigma, days, drift)
    p_above = tail_above(spot, strike, sigma, days, drift)
    p_drop_25 = 1.0 - tail_above(spot, spot*0.75, sigma, days, drift)
    spread_pct = (ask - bid)/((ask + bid)*0.5) if ask+bid else 1.0
    # Non-probabilistic, explicitly labeled ranking heuristic
    ret_quality = clamp(max_return / (0.20 * days / 365.0))
    liquidity_quality = clamp(1.0 - spread_pct/0.40) * clamp(open_interest/100.0)
    score = round(clamp(0.50*p_profit + 0.20*ret_quality + 0.15*liquidity_quality + 0.15*(1-p_drop_25))*100
                  - (10 if earnings else 0) - (10 if leveraged else 0))
    score = min(100,max(0,score))
    passes_budget = share_cost <= budget
    warnings = []
    if not passes_budget: warnings.append("Costs more than share-purchase budget")
    if max_profit <= 0: warnings.append("Maximum payoff does not make a profit")
    if spread_pct > 0.35: warnings.append("Wide bid/ask spread")
    if open_interest < 25: warnings.append("Low open interest")
    if earnings: warnings.append("Earnings during the contract")
    if leveraged: warnings.append("Leveraged daily-reset ETF: amplified/path-dependent risk")
    if days > 90: warnings.append("Long holding period")
    if not passes_budget or max_profit<=0 or spread_pct>0.35:
        verdict = "PASS / EXCLUDE"
    elif score >= 75:
        verdict = "WATCHLIST / VERIFY"
    elif score >= 60:
        verdict = "REVIEW"
    else:
        verdict = "LOW PRIORITY"
    return dict(cost=share_cost, credit=credit, breakeven=breakeven,
                max_profit=max_profit, max_return=max_return,
                p_profit=p_profit, p_above=p_above, p_drop25=p_drop_25,
                spread_pct=spread_pct, score=score, verdict=verdict,
                max_loss=max(0, share_cost-credit), warnings=warnings)
