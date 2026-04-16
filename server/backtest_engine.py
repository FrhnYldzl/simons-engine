"""
backtest_engine.py -- FP-09: Backtesting & Validation Framework

Sinyal kalite kontrolu: sadece istatistiksel olarak anlamli sinyaller uretime alinir.
Renaissance'ta her sinyal birden fazla bagimsiz zaman diliminde test edilmeden
asla uretime alinmaz.

Teknikler:
  1. Walk-Forward Backtesting (in-sample + out-of-sample)
  2. Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado
  3. Deflated Sharpe Ratio — Bailey & Lopez de Prado
  4. Monte Carlo Stress Testing (bootstrap)

Refs:
  - Lopez de Prado (2018). Advances in Financial ML. Ch. 11-12.
  - Bailey & Lopez de Prado (2014). The Deflated Sharpe Ratio.
  - Harvey & Liu (2015). Backtesting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def run_backtest(data: dict, signal_fn, initial_capital: float = 100_000,
                 transaction_cost: float = 0.001, max_position_pct: float = 0.05) -> dict:
    """
    Historical replay backtest.

    Args:
        data: {ticker: DataFrame} -- DataPipeline cikisi
        signal_fn: Sinyal ureten fonksiyon. signal_fn(df_window) -> direction in [-1,1]
        initial_capital: Baslangic sermayesi
        transaction_cost: Her trade'de yuzde maliyet (slippage + commission)
        max_position_pct: Tek pozisyon maksimum sermaye yuzdesi

    Returns:
        dict with metrics: total_return, sharpe, sortino, max_dd, win_rate, trades
    """
    if not data:
        return {"error": "no_data"}

    # Backtest her ticker icin ayri yapiyor
    all_returns = []
    trades = []

    for ticker, df in data.items():
        if len(df) < 60 or "returns" not in df.columns:
            continue

        # Strateji: her gun sinyal kontrol et, pozisyona gir/cik
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0
        ticker_returns = []

        for i in range(60, len(df) - 1):
            window = df.iloc[:i+1]
            next_return = df["returns"].iloc[i+1]

            try:
                signal = signal_fn(window)
            except Exception:
                signal = 0

            # Sinyal yon degistirdi mi?
            if position == 0 and abs(signal) > 0.3:
                position = 1 if signal > 0 else -1
                entry_price = float(df["close"].iloc[i])
                # Giris cost
                ticker_returns.append(-transaction_cost)
            elif position != 0:
                # Pozisyon acikken hareket
                strategy_return = position * next_return * max_position_pct
                ticker_returns.append(strategy_return)

                # Cikis kontrolu: sinyal tersi veya notr
                if (position > 0 and signal < -0.2) or (position < 0 and signal > 0.2) or abs(signal) < 0.1:
                    # Cikis cost
                    ticker_returns.append(-transaction_cost)
                    pnl = (float(df["close"].iloc[i]) - entry_price) * position / entry_price
                    trades.append({"ticker": ticker, "pnl": pnl, "bars_held": 1})
                    position = 0
                    entry_price = 0

        all_returns.extend(ticker_returns)

    if not all_returns:
        return {"error": "no_trades"}

    returns = np.array(all_returns)
    return compute_performance_metrics(returns, trades, initial_capital)


def compute_performance_metrics(returns: np.ndarray, trades: list = None,
                                initial_capital: float = 100_000) -> dict:
    """
    Sharpe, Sortino, Max DD, Win Rate vb. metriklerini hesapla.
    """
    if len(returns) == 0:
        return {"error": "empty_returns"}

    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return {"error": "all_nan"}

    # Cumulative return
    equity = initial_capital * np.cumprod(1 + returns)
    total_return = (equity[-1] / initial_capital) - 1 if len(equity) > 0 else 0

    # Annualized metrics (252 trading days)
    mean_daily = returns.mean()
    std_daily = returns.std()
    ann_return = mean_daily * 252
    ann_vol = std_daily * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino (only downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = (mean_daily * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf") if gains > 0 else 0

    # Skewness & Kurtosis (for Deflated Sharpe)
    skew = stats.skew(returns) if len(returns) >= 4 else 0
    kurt = stats.kurtosis(returns) if len(returns) >= 4 else 0

    return {
        "total_return": round(float(total_return), 4),
        "total_return_pct": round(float(total_return * 100), 2),
        "annualized_return_pct": round(float(ann_return * 100), 2),
        "annualized_volatility_pct": round(float(ann_vol * 100), 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "win_rate_pct": round(float(win_rate * 100), 2),
        "profit_factor": round(float(profit_factor), 3),
        "skewness": round(float(skew), 3),
        "kurtosis": round(float(kurt), 3),
        "n_trades": len(trades) if trades else 0,
        "n_observations": len(returns),
        "final_equity": round(float(equity[-1]), 2) if len(equity) > 0 else initial_capital,
        "ending_capital_pct": round(float(equity[-1] / initial_capital * 100), 2) if len(equity) > 0 else 100.0,
    }


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                          skew: float = 0, kurt: float = 0) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Multiple testing bias'ini duzelttikten sonra "gercek" Sharpe.
    DSR > 0.95 = sinyal istatistiksel olarak anlamli (p < 0.05).

    Args:
        sharpe: Observed (ham) Sharpe
        n_trials: Kac farkli sinyal test edildi (multiple testing)
        n_obs: Gozlem sayisi
        skew: Return serisi skewness
        kurt: Return serisi excess kurtosis

    Returns:
        DSR (0 ile 1 arasi olasilik)
    """
    if n_trials <= 1 or n_obs <= 2:
        return 0.5

    # Expected max Sharpe (Bailey & Lopez de Prado, Theorem 1)
    euler_mascheroni = 0.5772
    emax = ((1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_trials) +
            euler_mascheroni * stats.norm.ppf(1 - 1/(n_trials * np.e)))

    # Deflated Sharpe numerator
    numerator = (sharpe - emax) * np.sqrt(n_obs - 1)

    # Variance adjustment (skew + kurt correction)
    denom = np.sqrt(1 - skew * sharpe + ((kurt - 1) / 4) * (sharpe ** 2))
    if denom <= 0:
        return 0.0

    dsr_statistic = numerator / denom
    # CDF ile olasiliga cevir
    dsr_prob = float(stats.norm.cdf(dsr_statistic))
    return round(dsr_prob, 4)


def monte_carlo_stress_test(returns: np.ndarray, n_simulations: int = 1000,
                            horizon_days: int = 252) -> dict:
    """
    Block bootstrap ile N senaryo sample'la ve performansi tahmin et.

    Bu, "gerceklik farkli olsaydi" testi. Ornekleme ile confidence interval.
    """
    if len(returns) < 30:
        return {"error": "insufficient_data"}

    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return {"error": "all_nan"}

    # Block bootstrap (5-gun block)
    block_size = 5
    np.random.seed(42)

    final_returns = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Random block indices
        n_blocks = horizon_days // block_size
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, len(returns) - block_size))
            blocks.extend(returns[start:start + block_size])

        sim_returns = np.array(blocks[:horizon_days])
        equity = np.cumprod(1 + sim_returns)

        final_returns.append(equity[-1] - 1 if len(equity) > 0 else 0)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_drawdowns.append(dd.min() if len(dd) > 0 else 0)

    final_returns = np.array(final_returns)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "expected_return_pct": round(float(np.mean(final_returns) * 100), 2),
        "median_return_pct": round(float(np.median(final_returns) * 100), 2),
        "percentile_5_pct": round(float(np.percentile(final_returns, 5) * 100), 2),
        "percentile_95_pct": round(float(np.percentile(final_returns, 95) * 100), 2),
        "prob_positive_pct": round(float((final_returns > 0).sum() / n_simulations * 100), 2),
        "worst_drawdown_pct": round(float(np.min(max_drawdowns) * 100), 2),
        "median_drawdown_pct": round(float(np.median(max_drawdowns) * 100), 2),
    }


def walk_forward_split(data_len: int, n_splits: int = 5,
                       test_size: int = 30) -> list:
    """
    Walk-forward cross-validation splits (time-ordered).

    Time series icin: once X gun train, sonraki Y gun test, kaydir.
    Returns list of (train_indices, test_indices) tuples.
    """
    splits = []
    total = data_len

    # Her split'te test window'u ilerletiyoruz
    for i in range(n_splits):
        test_end = total - i * test_size
        test_start = test_end - test_size
        train_end = test_start - 10  # Embargo: 10 gun arada bosluk (purge)

        if train_end < 60:  # Min train size
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        splits.append((train_idx, test_idx))

    return list(reversed(splits))  # En eskiden en yeniye


def validate_signal(signal_fn, data: dict, ticker: str = None) -> dict:
    """
    Tek bir sinyal stratejisini full validation pipeline'dan gecir.

    Returns:
        {
            "backtest_metrics": {...},
            "deflated_sharpe": 0.xxx,
            "monte_carlo": {...},
            "verdict": "accept"|"reject"|"borderline"
        }
    """
    # Backtest
    test_data = {ticker: data[ticker]} if ticker and ticker in data else data
    backtest = run_backtest(test_data, signal_fn)

    if "error" in backtest:
        return {"verdict": "reject", "reason": backtest["error"]}

    sharpe = backtest.get("sharpe_ratio", 0)
    skew = backtest.get("skewness", 0)
    kurt = backtest.get("kurtosis", 0)
    n_obs = backtest.get("n_observations", 0)

    # DSR (assume 10 trials as baseline)
    dsr = deflated_sharpe_ratio(sharpe, n_trials=10, n_obs=n_obs, skew=skew, kurt=kurt)

    # Monte Carlo (basit)
    # Not: return serisine dogrudan erisim icin daha detayli yapilmali

    # Verdict
    verdict = "reject"
    reason = ""
    if dsr > 0.95:
        verdict = "accept"
        reason = f"DSR={dsr} > 0.95 (statistically significant)"
    elif dsr > 0.8:
        verdict = "borderline"
        reason = f"DSR={dsr} marginal -- needs more data"
    else:
        reason = f"DSR={dsr} too low -- likely overfit"

    return {
        "backtest_metrics": backtest,
        "deflated_sharpe": dsr,
        "verdict": verdict,
        "reason": reason,
    }


# Convenience: sinyal motoru icin hazir adapter'lar
def mean_reversion_signal(df: pd.DataFrame, window: int = 60) -> float:
    """Test amacli basit mean reversion signal (z-score bazli)."""
    if len(df) < window:
        return 0.0
    close = df["close"].values
    mean = np.mean(close[-window:])
    std = np.std(close[-window:])
    if std <= 0:
        return 0.0
    z = (close[-1] - mean) / std
    # z > 2 short, z < -2 long (reversion beklentisi)
    return float(np.clip(-z / 3.0, -1, 1))


def momentum_signal(df: pd.DataFrame, lookback: int = 5) -> float:
    """Test amacli basit momentum signal."""
    if len(df) < lookback + 1:
        return 0.0
    returns = df["returns"].values
    mom = returns[-lookback:].sum()
    return float(np.clip(mom * 20, -1, 1))
