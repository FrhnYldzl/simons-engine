"""
signal_engine.py — FP-03: Statistical Arbitrage Signal Engine

Renaissance'ın sinyal üretim çekirdeği. Her sinyal p < 0.01 eşiğini
geçmeli; %99'dan fazlası eleniyor.

Mercer: 'Zamanın %50.75'inde haklıyız, ama zamanın %100'ünde %50.75
oranında haklıyız. Bu şekilde milyarlarca dolar kazanabilirsiniz.'

Signals:
  1. Mean Reversion — Ornstein-Uhlenbeck process
  2. Momentum — Laufer's 24-hour effect + multi-period
  3. Volume Anomaly — Unusual volume with directional bias
  4. Overnight Gap — Gap exploitation

Refs:
  - Avellaneda & Lee (2010). Statistical Arbitrage in US Equities.
  - Cont (2001). Empirical Properties of Asset Returns.
  - Lopez de Prado (2018), Ch. 11-13: CPCV, Deflated Sharpe.
  - Zuckerman (2019), Ch. 10-14 — Laufer's discoveries.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[Signal] statsmodels yuklu degil — ADF test devre disi")


class Signal:
    """Tek bir sinyal nesnesi."""

    def __init__(self, ticker: str, name: str, direction: float,
                 conviction: float, p_value: float, holding_period: int,
                 alpha_estimate: float, metadata: dict = None):
        self.ticker = ticker
        self.name = name
        self.direction = direction      # -1 to +1 (short to long)
        self.conviction = conviction    # 0 to 1
        self.p_value = p_value          # İstatistiksel anlamlılık
        self.holding_period = holding_period  # Gün
        self.alpha_estimate = alpha_estimate  # Beklenen günlük alpha
        self.metadata = metadata or {}

    def is_valid(self, p_threshold: float = 0.01) -> bool:
        """Sinyal p < threshold mı?"""
        return self.p_value < p_threshold

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "signal_name": self.name,
            "direction": round(self.direction, 4),
            "side": "long" if self.direction > 0 else "short" if self.direction < 0 else "neutral",
            "conviction": round(self.conviction, 4),
            "p_value": round(self.p_value, 6),
            "holding_period": self.holding_period,
            "alpha_estimate": round(self.alpha_estimate, 6),
            "valid": self.is_valid(),
            "metadata": self.metadata,
        }


class SignalEngine:
    """
    İstatistiksel arbitraj sinyal üretici.

    Her ticker için birden fazla sinyal üretir, sonra combine eder.
    Sadece p < 0.01 olanlar geçer.
    """

    def __init__(self, p_threshold: float = 0.05):
        """
        p_threshold: Sinyal kabul eşiği.
        Medallion p < 0.01 kullanır, biz başlangıçta p < 0.05 ile gevşetiyoruz.
        """
        self.p_threshold = p_threshold

    def generate_signals(self, data: dict[str, pd.DataFrame], regime: str = "normal") -> list[Signal]:
        """
        Tüm ticker'lar için sinyal üret.

        Returns: valid sinyaller listesi (p < threshold)
        """
        all_signals = []

        for ticker, df in data.items():
            if len(df) < 60:
                continue

            signals = []

            # 1. Mean Reversion
            mr = self._mean_reversion(ticker, df)
            if mr:
                signals.append(mr)

            # 2. Momentum (multi-period)
            mom = self._momentum(ticker, df)
            if mom:
                signals.append(mom)

            # 3. Volume Anomaly
            vol = self._volume_anomaly(ticker, df)
            if vol:
                signals.append(vol)

            # 4. Overnight Gap
            gap = self._overnight_gap(ticker, df)
            if gap:
                signals.append(gap)

            # Combined signal (ağırlıklı ortalama)
            valid_signals = [s for s in signals if s.is_valid(self.p_threshold)]

            if valid_signals:
                combined = self._combine_signals(ticker, valid_signals)
                all_signals.append(combined)

        # Conviction'a göre sırala
        all_signals.sort(key=lambda s: abs(s.conviction), reverse=True)

        return all_signals

    def _mean_reversion(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Ornstein-Uhlenbeck mean reversion sinyali.

        z-score > 2: short (ortalamaya dönecek)
        z-score < -2: long (ortalamaya dönecek)

        Half-life: HL = -ln(2) / ln(beta)
        """
        try:
            close = df["close"].values
            if len(close) < 60:
                return None

            # 60 günlük rolling mean
            window = 60
            rolling_mean = pd.Series(close).rolling(window).mean().values
            rolling_std = pd.Series(close).rolling(window).std().values

            # Son değer
            current = close[-1]
            mean = rolling_mean[-1]
            std = rolling_std[-1]

            if std <= 0 or np.isnan(mean):
                return None

            z_score = (current - mean) / std

            # ADF test — mean reversion var mı?
            if STATSMODELS_AVAILABLE:
                adf_result = adfuller(close[-window:], maxlag=10)
                p_value = adf_result[1]
            else:
                # Fallback: z-score bazlı p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Half-life hesaplama
            spread = close[-window:] - rolling_mean[-window:]
            spread = spread[~np.isnan(spread)]
            if len(spread) < 20:
                return None

            spread_lag = spread[:-1]
            spread_now = spread[1:]
            if len(spread_lag) > 0 and np.std(spread_lag) > 0:
                beta = np.polyfit(spread_lag, spread_now, 1)[0]
                half_life = max(1, int(-np.log(2) / np.log(max(abs(beta), 0.01))))
            else:
                half_life = 10

            # Direction: z-score'un tersi (mean reversion)
            direction = np.clip(-z_score / 3.0, -1, 1)
            conviction = min(abs(z_score) / 3.0, 1.0)

            # Alpha estimate (günlük)
            alpha = abs(z_score) * std / current / half_life

            return Signal(
                ticker=ticker,
                name="mean_reversion",
                direction=float(direction),
                conviction=float(conviction),
                p_value=float(p_value),
                holding_period=min(half_life, 20),
                alpha_estimate=float(alpha),
                metadata={
                    "z_score": round(float(z_score), 3),
                    "half_life": half_life,
                    "adf_statistic": round(float(adf_result[0]), 3),
                },
            )
        except Exception:
            return None

    def _momentum(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Multi-period momentum sinyali.

        Laufer's discovery: 24-saat etkisi.
        Kısa vadeli momentum (1-5 gün) istatistiksel olarak anlamlı.
        """
        try:
            returns = df["returns"].dropna().values
            if len(returns) < 60:
                return None

            # Multi-period momentum
            mom_1d = returns[-1]
            mom_3d = returns[-3:].sum()
            mom_5d = returns[-5:].sum()
            mom_10d = returns[-10:].sum()

            # Overnight vs intraday decomposition
            overnight = df["overnight_return"].dropna().values
            intraday = df["intraday_return"].dropna().values

            # Momentum composite
            composite = 0.1 * mom_1d + 0.2 * mom_3d + 0.4 * mom_5d + 0.3 * mom_10d

            # T-test: son 20 gün return'ü > 0 mı?
            recent_returns = returns[-20:]
            t_stat, p_value = stats.ttest_1samp(recent_returns, 0)

            # Volume confirmation
            vol_ratio = df["volume_ratio"].values[-1] if "volume_ratio" in df.columns else 1.0

            # Direction
            direction = np.clip(composite * 20, -1, 1)  # Scale to [-1, 1]
            conviction = min(abs(t_stat) / 3.0, 1.0) * min(vol_ratio, 2.0) / 2.0

            # Alpha
            alpha = abs(composite) / 5  # 5-day holding

            return Signal(
                ticker=ticker,
                name="momentum",
                direction=float(direction),
                conviction=float(conviction),
                p_value=float(p_value),
                holding_period=5,
                alpha_estimate=float(alpha),
                metadata={
                    "mom_1d": round(float(mom_1d), 4),
                    "mom_5d": round(float(mom_5d), 4),
                    "mom_10d": round(float(mom_10d), 4),
                    "t_statistic": round(float(t_stat), 3),
                    "volume_ratio": round(float(vol_ratio), 2),
                },
            )
        except Exception:
            return None

    def _volume_anomaly(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Unusual volume + directional bias sinyali.

        Yüksek hacim + pozitif return = institutional buying
        Yüksek hacim + negatif return = institutional selling
        """
        try:
            if "volume_ratio" not in df.columns or len(df) < 20:
                return None

            vol_ratio = df["volume_ratio"].values[-1]
            ret_today = df["returns"].values[-1]
            ret_5d = df["returns"].values[-5:].sum()

            # Volume > 2x ortalama mı?
            if vol_ratio < 1.5:
                return None

            # Direction: return yönünde
            direction = np.clip(ret_today * 10, -1, 1)
            conviction = min((vol_ratio - 1.0) / 3.0, 1.0) * min(abs(ret_today) * 50, 1.0)

            # P-value: volume'ün bu kadar yüksek olma olasılığı
            vol_history = df["volume_ratio"].dropna().values[-60:]
            if len(vol_history) < 20:
                return None
            z_vol = (vol_ratio - np.mean(vol_history)) / max(np.std(vol_history), 0.01)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_vol)))

            return Signal(
                ticker=ticker,
                name="volume_anomaly",
                direction=float(direction),
                conviction=float(conviction),
                p_value=float(p_value),
                holding_period=3,
                alpha_estimate=float(abs(ret_today) / 3),
                metadata={
                    "volume_ratio": round(float(vol_ratio), 2),
                    "z_volume": round(float(z_vol), 2),
                    "return_today": round(float(ret_today), 4),
                },
            )
        except Exception:
            return None

    def _overnight_gap(self, ticker: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Overnight gap exploitation.

        Büyük gap'ler genelde mean-revert eder (gap fill).
        Küçük gap'ler genelde devam eder (continuation).
        """
        try:
            if "overnight_return" not in df.columns or len(df) < 30:
                return None

            gap_today = df["overnight_return"].values[-1]
            avg_gap = df["overnight_return"].dropna().values[-30:].mean()
            std_gap = df["overnight_return"].dropna().values[-30:].std()

            if std_gap <= 0:
                return None

            z_gap = (gap_today - avg_gap) / std_gap

            # Büyük gap → mean reversion (ters yön)
            # Küçük gap → continuation (aynı yön)
            if abs(z_gap) > 2:
                direction = np.clip(-z_gap / 4.0, -1, 1)  # Reversion
                name_suffix = "reversal"
            elif abs(z_gap) > 1:
                direction = np.clip(z_gap / 4.0, -1, 1)  # Continuation
                name_suffix = "continuation"
            else:
                return None

            conviction = min(abs(z_gap) / 4.0, 1.0)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_gap)))

            return Signal(
                ticker=ticker,
                name=f"gap_{name_suffix}",
                direction=float(direction),
                conviction=float(conviction),
                p_value=float(p_value),
                holding_period=1,
                alpha_estimate=float(abs(gap_today) / 2),
                metadata={
                    "gap_pct": round(float(gap_today * 100), 3),
                    "z_gap": round(float(z_gap), 2),
                    "type": name_suffix,
                },
            )
        except Exception:
            return None

    def _combine_signals(self, ticker: str, signals: list[Signal]) -> Signal:
        """
        Birden fazla sinyali tek bir combined signal'e birleştir.
        Ağırlık: conviction ve (1 - p_value) bazlı.
        """
        if len(signals) == 1:
            return signals[0]

        total_weight = 0
        weighted_direction = 0
        min_p = 1.0
        max_conviction = 0
        holdings = []
        names = []

        for s in signals:
            weight = s.conviction * (1 - s.p_value)
            weighted_direction += s.direction * weight
            total_weight += weight
            min_p = min(min_p, s.p_value)
            max_conviction = max(max_conviction, s.conviction)
            holdings.append(s.holding_period)
            names.append(s.name)

        if total_weight > 0:
            direction = weighted_direction / total_weight
        else:
            direction = 0

        # Combined alpha
        alpha = np.mean([s.alpha_estimate for s in signals])

        return Signal(
            ticker=ticker,
            name="combined",
            direction=float(np.clip(direction, -1, 1)),
            conviction=float(max_conviction * 0.8 + 0.2 * np.mean([s.conviction for s in signals])),
            p_value=float(min_p),
            holding_period=int(np.median(holdings)),
            alpha_estimate=float(alpha),
            metadata={
                "components": names,
                "n_signals": len(signals),
                "signal_details": [s.to_dict() for s in signals],
            },
        )
