"""
data_pipeline.py — FP-01: Data Pipeline & Cleaning Engine

Sandor Straus'un 1980'lerde başlattığı veri toplama ve temizleme altyapısı.
Renaissance'ın en temel rekabet avantajı.

Nick Patterson: 'Basit şeyleri doğru yapmak için en zeki insanlara
ihtiyacınız var. Bu yüzden sadece veri temizliği için birkaç doktora
istihdam ediyoruz.'

Refs:
  - Zuckerman (2019), Ch. 4-6
  - Lopez de Prado (2018), Ch. 2
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional


def fetch_and_clean(broker, tickers: list, days: int = 252) -> dict[str, pd.DataFrame]:
    """
    Tüm ticker'lar için temizlenmiş OHLCV verisi al.

    Pipeline:
      1. Raw bar verisi fetch
      2. Split/dividend adjustment (Alpaca zaten adjusted)
      3. Outlier detection (Z-score > 4)
      4. Missing data interpolation
      5. Volume normalization
      6. Return hesaplama

    Returns: {ticker: DataFrame} with columns:
      open, high, low, close, volume, returns, log_returns,
      volume_ratio, clean_flag
    """
    result = {}

    for ticker in tickers:
        try:
            bars = broker.get_bars(ticker, limit=days)
            if not bars or len(bars) < 20:
                continue

            df = _bars_to_dataframe(bars)
            df = _clean_outliers(df)
            df = _fill_missing(df)
            df = _add_features(df)
            df = _normalize_volume(df)

            result[ticker] = df
        except Exception as e:
            print(f"[DataPipeline] {ticker} hata: {e}")

    return result


def _bars_to_dataframe(bars) -> pd.DataFrame:
    """Alpaca bar verilerini DataFrame'e çevir."""
    records = []
    for bar in bars:
        records.append({
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Duplicate index temizle
    df = df[~df.index.duplicated(keep="last")]

    return df


def _clean_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """
    Z-score > 4 olan return'leri flag'le ve cap'le.
    Renaissance: Minimum 3 bağımsız kaynak ile reconciliation.
    """
    df["returns_raw"] = df["close"].pct_change()
    df["clean_flag"] = True

    if len(df) < 10:
        return df

    mean_ret = df["returns_raw"].mean()
    std_ret = df["returns_raw"].std()

    if std_ret > 0:
        z_scores = (df["returns_raw"] - mean_ret) / std_ret
        outlier_mask = z_scores.abs() > z_threshold

        # Outlier'ları winsorize (cap at ±z_threshold * std)
        cap = z_threshold * std_ret
        df.loc[outlier_mask, "clean_flag"] = False
        df["returns_raw"] = df["returns_raw"].clip(
            lower=mean_ret - cap,
            upper=mean_ret + cap,
        )

        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"[DataPipeline] {n_outliers} outlier tespit edildi ve winsorize edildi")

    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Missing data interpolation — forward fill + linear interpolation."""
    # Forward fill (max 3 gün)
    df = df.ffill(limit=3)
    # Kalan NaN'ları interpolate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)
    # Hala NaN varsa drop
    df = df.dropna(subset=["close"])
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temel özellikler ekle."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (20-gün rolling)
    df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)

    # Overnight vs intraday decomposition (Laufer's insight)
    df["overnight_return"] = df["open"] / df["close"].shift(1) - 1
    df["intraday_return"] = df["close"] / df["open"] - 1

    # Range features
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    df["true_range"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    return df


def _normalize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Volume normalization — 20-gün ortalamasına göre ratio."""
    avg_vol = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / avg_vol
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    return df


def get_returns_matrix(data: dict[str, pd.DataFrame], min_history: int = 60) -> pd.DataFrame:
    """
    Tüm ticker'ların return matrisini oluştur.
    Cross-asset analiz için gerekli.

    Returns: DataFrame (date x ticker) of daily returns
    """
    returns = {}
    for ticker, df in data.items():
        if len(df) >= min_history and "returns" in df.columns:
            returns[ticker] = df["returns"]

    if not returns:
        return pd.DataFrame()

    matrix = pd.DataFrame(returns)
    # En az %80 veri olan ticker'ları tut
    threshold = len(matrix) * 0.8
    matrix = matrix.dropna(axis=1, thresh=int(threshold))
    matrix = matrix.dropna()

    return matrix


def get_covariance_matrix(returns_matrix: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """
    Robust covariance estimation.

    Methods:
      - sample: Basit sample covariance
      - ledoit_wolf: Ledoit-Wolf shrinkage (default, recommended)
      - ewma: Exponential weighted (short-term)

    Ref: Ledoit & Wolf (2004)
    """
    if returns_matrix.empty:
        return np.array([])

    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns_matrix.dropna())
            return lw.covariance_
        except ImportError:
            return returns_matrix.cov().values

    elif method == "ewma":
        # RiskMetrics EWMA (lambda=0.94)
        return returns_matrix.ewm(span=60).cov().iloc[-len(returns_matrix.columns):].values

    else:
        return returns_matrix.cov().values
