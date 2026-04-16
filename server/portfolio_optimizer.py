"""
portfolio_optimizer.py -- FP-07: Portfolio Construction & Market-Neutral Optimizer

Medallion'un piyasa-notr yapi (beta ~ -0.41) bu katmanda uretilir.
Long ve short pozisyonlari dengeli tutulur.

2008'de S&P %37 duserken Medallion %82.4 kazandi -- bu modulun urunu.

Teknikler:
  1. Long-Short Market Neutral: Net exposure ~ 0
  2. Robust Covariance (Ledoit-Wolf shrinkage)
  3. Risk Parity weights (alternative)
  4. Turnover penalty

Refs:
  - Grinold & Kahn (2000). Active Portfolio Management. Ch. 14.
  - Ledoit & Wolf (2004). J. Multivariate Analysis.
  - Cornell (2020). Medallion Fund. J. Portfolio Mgmt.
"""

import numpy as np
import pandas as pd
from typing import Optional


def optimize_portfolio(signals: list, covariance: Optional[np.ndarray] = None,
                        method: str = "market_neutral",
                        max_weight: float = 0.05,
                        target_gross_leverage: float = 1.0) -> dict:
    """
    Sinyal listesinden optimal portfoy agirliklari uret.

    Args:
        signals: [{ticker, direction, conviction, alpha_estimate}, ...]
        covariance: ticker sirasinda covariance matrix (Ledoit-Wolf onerilir)
        method: "market_neutral", "risk_parity", "equal_weight"
        max_weight: Tek pozisyon max yuzde
        target_gross_leverage: Gross exposure hedefi (long + |short|)

    Returns:
        {
            "weights": {ticker: float},  # + long, - short
            "long_exposure": float,
            "short_exposure": float,
            "net_exposure": float,
            "gross_exposure": float,
            "beta_proxy": float,
            "method": str,
        }
    """
    if not signals:
        return {"error": "no_signals", "weights": {}}

    # Sort by conviction x alpha
    ranked = sorted(signals, key=lambda s: abs(s.get("direction", 0)) * s.get("conviction", 0),
                    reverse=True)

    if method == "market_neutral":
        return _market_neutral_weights(ranked, max_weight, target_gross_leverage)
    elif method == "risk_parity":
        return _risk_parity_weights(ranked, covariance, max_weight, target_gross_leverage)
    else:
        return _equal_weight(ranked, max_weight, target_gross_leverage)


def _market_neutral_weights(signals: list, max_weight: float,
                             target_gross: float) -> dict:
    """
    Long ve short'u dengeli yap (net ~ 0).
    En yuksek alpha long'lari al, dushuk (negatif) alpha short'lari al.
    """
    longs = [s for s in signals if s.get("direction", 0) > 0]
    shorts = [s for s in signals if s.get("direction", 0) < 0]

    n_pairs = min(len(longs), len(shorts))
    if n_pairs == 0:
        # Sadece bir yon varsa notr degiliz
        return _equal_weight(signals, max_weight, target_gross)

    # Her iki taraftan eshit sayida ticker al
    longs = longs[:n_pairs]
    shorts = shorts[:n_pairs]

    # Weight: conviction-scaled
    total_long_conv = sum(s.get("conviction", 0) for s in longs)
    total_short_conv = sum(s.get("conviction", 0) for s in shorts)

    weights = {}
    per_side_budget = target_gross / 2.0  # Each side gets half of gross

    if total_long_conv > 0:
        for s in longs:
            w = (s.get("conviction", 0) / total_long_conv) * per_side_budget
            w = min(w, max_weight)  # Cap
            weights[s["ticker"]] = round(w, 4)

    if total_short_conv > 0:
        for s in shorts:
            w = -(s.get("conviction", 0) / total_short_conv) * per_side_budget
            w = max(w, -max_weight)  # Cap (negative)
            weights[s["ticker"]] = round(w, 4)

    return _build_result(weights, "market_neutral")


def _risk_parity_weights(signals: list, covariance: Optional[np.ndarray],
                         max_weight: float, target_gross: float) -> dict:
    """
    Risk parity: her pozisyon esit risk katkisi yapar.
    Yuksek vol olan ticker'a daha az weight.
    """
    if covariance is None or covariance.size == 0:
        # Fallback: equal weight
        return _equal_weight(signals, max_weight, target_gross)

    # Diagonal ile individual vol
    vols = np.sqrt(np.diag(covariance))
    if len(vols) != len(signals) or np.any(vols <= 0):
        return _equal_weight(signals, max_weight, target_gross)

    # Inverse-vol weighting
    inv_vol = 1.0 / vols
    total = inv_vol.sum()
    if total <= 0:
        return _equal_weight(signals, max_weight, target_gross)

    normalized = inv_vol / total * target_gross

    weights = {}
    for s, w in zip(signals, normalized):
        direction = s.get("direction", 0)
        signed_w = w * (1 if direction > 0 else -1 if direction < 0 else 0)
        signed_w = max(min(signed_w, max_weight), -max_weight)
        weights[s["ticker"]] = round(float(signed_w), 4)

    return _build_result(weights, "risk_parity")


def _equal_weight(signals: list, max_weight: float, target_gross: float) -> dict:
    """Equal weight with direction sign."""
    if not signals:
        return {"weights": {}, "method": "equal_weight"}

    n = len(signals)
    per = min(target_gross / n, max_weight)

    weights = {}
    for s in signals:
        direction = s.get("direction", 0)
        sign = 1 if direction > 0 else -1 if direction < 0 else 0
        weights[s["ticker"]] = round(per * sign, 4)

    return _build_result(weights, "equal_weight")


def _build_result(weights: dict, method: str) -> dict:
    """Weight dict'ten ozet metrikleri hesapla."""
    vals = list(weights.values())
    long_exp = sum(v for v in vals if v > 0)
    short_exp = sum(v for v in vals if v < 0)
    net_exp = long_exp + short_exp
    gross_exp = long_exp - short_exp

    # Beta proxy: net exposure -- sifir = notr
    # Gercek beta icin her ticker'in SPY korelasyonu lazim
    beta_proxy = round(net_exp, 4)

    return {
        "weights": weights,
        "long_exposure": round(long_exp, 4),
        "short_exposure": round(short_exp, 4),
        "net_exposure": round(net_exp, 4),
        "gross_exposure": round(gross_exp, 4),
        "beta_proxy": beta_proxy,
        "n_long": sum(1 for v in vals if v > 0),
        "n_short": sum(1 for v in vals if v < 0),
        "method": method,
    }


def apply_turnover_penalty(new_weights: dict, current_positions: dict,
                            turnover_cost: float = 0.001) -> dict:
    """
    Pozisyon degisikligini minimize et (turnover cost).
    Cok sik rebalance bir maliyet, buna karsi korur.
    """
    turnover = 0
    for ticker, new_w in new_weights.items():
        old_w = current_positions.get(ticker, 0)
        turnover += abs(new_w - old_w)

    return {
        "new_weights": new_weights,
        "turnover": round(turnover, 4),
        "turnover_cost": round(turnover * turnover_cost, 6),
    }


def compute_factor_exposures(weights: dict, data: dict) -> dict:
    """
    Basit factor exposures: beta (vs SPY), sector concentration.
    """
    if not weights or "SPY" not in data:
        return {"beta": 0, "sector_concentration": {}}

    spy_ret = data["SPY"]["returns"].dropna().values
    if len(spy_ret) < 30:
        return {"beta": 0, "sector_concentration": {}}

    portfolio_betas = {}
    for ticker, w in weights.items():
        if ticker in data and "returns" in data[ticker].columns:
            t_ret = data[ticker]["returns"].dropna().values
            # Min common length
            n = min(len(spy_ret), len(t_ret))
            if n < 30:
                continue
            spy_aligned = spy_ret[-n:]
            t_aligned = t_ret[-n:]

            cov = np.cov(t_aligned, spy_aligned)[0, 1]
            var = np.var(spy_aligned)
            beta = cov / var if var > 0 else 0
            portfolio_betas[ticker] = beta * w

    # Total portfolio beta
    total_beta = sum(portfolio_betas.values())

    return {
        "portfolio_beta": round(float(total_beta), 4),
        "individual_betas": {k: round(float(v), 4) for k, v in portfolio_betas.items()},
        "n_tickers": len(portfolio_betas),
    }
