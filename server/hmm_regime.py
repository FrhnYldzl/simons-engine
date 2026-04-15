"""
hmm_regime.py — FP-02: Hidden Markov Model Regime Detector

Leonard Baum'un IDA'da geliştirdiği Baum-Welch algoritmasının
finansal piyasalara uyarlanması. Gizli piyasa rejimlerini tespit eder.

Baum-Welch algoritması orijinal olarak NSA'da şifreli iletişimlerdeki
gizli durumları çıkarmak için geliştirildi. Simons ve Baum bu tekniği
doğrudan FX piyasalarına uyguladı.

Refs:
  - Baum & Petrie (1966). Statistical Inference for Probabilistic
    Functions of Finite State Markov Chains.
  - Rabiner (1989). A Tutorial on HMMs.
  - Ryden et al. (1998). Stylized facts and HMM.
  - Zuckerman (2019), Ch. 3 — Baum's HMMs at Monemetrics.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[HMM] hmmlearn yuklu degil — fallback mode")


# Regime labels (sıralı: düşük vol → yüksek vol)
REGIME_LABELS = {
    0: "low_volatility",    # Sakin piyasa, trending
    1: "normal",            # Normal koşullar
    2: "high_volatility",   # Yüksek volatilite, regime change
    3: "crisis",            # Kriz, extreme moves
}

REGIME_DISPLAY = {
    "low_volatility": {"name": "Low Volatility", "color": "#22c55e", "risk_mult": 1.2},
    "normal":         {"name": "Normal",         "color": "#3b82f6", "risk_mult": 1.0},
    "high_volatility":{"name": "High Volatility","color": "#f59e0b", "risk_mult": 0.6},
    "crisis":         {"name": "Crisis",         "color": "#ef4444", "risk_mult": 0.2},
}


class HMMRegimeDetector:
    """
    Gaussian HMM ile piyasa rejim tespiti.

    States:
      - Low Volatility: Düşük vol, yüksek pozitif drift → agresif pozisyon
      - Normal: Orta vol, nötr drift → standart pozisyon
      - High Volatility: Yüksek vol, negatif drift → küçük pozisyon
      - Crisis: Extreme vol, büyük negatif drift → minimum pozisyon

    Model BIC/AIC ile optimal state sayısı seçer.
    """

    def __init__(self, n_states: int = 4, lookback_days: int = 252):
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.model: Optional[GaussianHMM] = None
        self.fitted = False
        self._last_regime = "normal"
        self._transition_matrix = None
        self._state_means = None
        self._state_vars = None

    def fit(self, returns: np.ndarray) -> dict:
        """
        HMM'i return serisine fit et.

        Args:
            returns: 1D numpy array of daily returns

        Returns:
            dict with regime info, transition matrix, state params
        """
        # Temizlik
        returns = returns[~np.isnan(returns)]
        if len(returns) < 60:
            return self._fallback_regime(returns)

        # Son N gün
        returns = returns[-self.lookback_days:]

        # Feature matrix: [returns, abs_returns (volatility proxy)]
        X = np.column_stack([
            returns,
            np.abs(returns),  # Volatility proxy
        ])

        if not HMM_AVAILABLE:
            return self._fallback_regime(returns)

        # Optimal state sayısı BIC ile
        best_model = None
        best_bic = np.inf

        for n in range(2, min(self.n_states + 1, 6)):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42,
                    tol=0.01,
                )
                model.fit(X)
                bic = self._compute_bic(model, X)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            return self._fallback_regime(returns)

        self.model = best_model
        self.n_states = best_model.n_components
        self.fitted = True

        # State'leri volatiliteye göre sırala (düşük → yüksek)
        state_vols = [best_model.covars_[i][1, 1] for i in range(best_model.n_components)]
        sorted_indices = np.argsort(state_vols)

        # Mapping: eski state → yeni sıralı state
        state_map = {old: new for new, old in enumerate(sorted_indices)}

        # Hidden states
        hidden_states = best_model.predict(X)
        mapped_states = np.array([state_map[s] for s in hidden_states])

        # Current regime
        current_state = mapped_states[-1]
        current_regime = REGIME_LABELS.get(min(current_state, 3), "normal")
        self._last_regime = current_regime

        # Transition matrix (remapped)
        trans_mat = best_model.transmat_
        n = best_model.n_components
        remapped_trans = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                remapped_trans[state_map[i], state_map[j]] = trans_mat[i, j]
        self._transition_matrix = remapped_trans

        # State parameters (remapped)
        self._state_means = [best_model.means_[sorted_indices[i]][0] for i in range(n)]
        self._state_vars = [best_model.covars_[sorted_indices[i]][0, 0] for i in range(n)]

        # Regime probabilities (son gözlem)
        state_probs = best_model.predict_proba(X)[-1]
        remapped_probs = [state_probs[sorted_indices[i]] for i in range(n)]

        # Regime değişim olasılığı
        if len(mapped_states) >= 2:
            regime_change_prob = 1 - remapped_trans[current_state, current_state]
        else:
            regime_change_prob = 0

        # Sonuç
        regime_info = REGIME_DISPLAY.get(current_regime, REGIME_DISPLAY["normal"])

        return {
            "regime": current_regime,
            "regime_name": regime_info["name"],
            "regime_color": regime_info["color"],
            "risk_multiplier": regime_info["risk_mult"],
            "n_states": int(n),
            "current_state": int(current_state),
            "state_probabilities": {
                REGIME_LABELS.get(i, f"state_{i}"): round(float(remapped_probs[i]), 4)
                for i in range(n)
            },
            "regime_change_probability": round(float(regime_change_prob), 4),
            "transition_matrix": remapped_trans.tolist(),
            "state_means": [round(float(m) * 252, 4) for m in self._state_means],  # Annualized
            "state_volatilities": [round(float(np.sqrt(v) * np.sqrt(252)), 4) for v in self._state_vars],
            "bic": round(float(best_bic), 2),
            "regime_history": self._regime_history(mapped_states),
        }

    def get_current_regime(self) -> str:
        return self._last_regime

    def get_risk_multiplier(self) -> float:
        info = REGIME_DISPLAY.get(self._last_regime, REGIME_DISPLAY["normal"])
        return info["risk_mult"]

    def _compute_bic(self, model, X):
        """Bayesian Information Criterion."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_components = model.n_components

        # Parametre sayısı
        n_params = (
            n_components - 1  # Start probs
            + n_components * (n_components - 1)  # Transition
            + n_components * n_features  # Means
            + n_components * n_features * (n_features + 1) // 2  # Covariance
        )

        log_likelihood = model.score(X) * n_samples
        return -2 * log_likelihood + n_params * np.log(n_samples)

    def _regime_history(self, states, last_n: int = 30) -> list:
        """Son N günün rejim geçmişi."""
        recent = states[-last_n:]
        return [
            {"day": int(i), "regime": REGIME_LABELS.get(min(int(s), 3), "normal")}
            for i, s in enumerate(recent)
        ]

    def _fallback_regime(self, returns) -> dict:
        """Yetersiz veri durumunda basit volatilite bazlı rejim."""
        if len(returns) < 5:
            return {
                "regime": "normal",
                "regime_name": "Normal",
                "regime_color": "#3b82f6",
                "risk_multiplier": 1.0,
                "n_states": 0,
                "error": "Yetersiz veri",
            }

        vol = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else np.std(returns) * np.sqrt(252)

        if vol < 0.10:
            regime = "low_volatility"
        elif vol < 0.20:
            regime = "normal"
        elif vol < 0.35:
            regime = "high_volatility"
        else:
            regime = "crisis"

        info = REGIME_DISPLAY[regime]
        self._last_regime = regime

        return {
            "regime": regime,
            "regime_name": info["name"],
            "regime_color": info["color"],
            "risk_multiplier": info["risk_mult"],
            "n_states": 0,
            "annualized_volatility": round(float(vol), 4),
            "method": "fallback_volatility",
        }


# Singleton instance
_detector = HMMRegimeDetector()


def detect_regime(returns: np.ndarray) -> dict:
    """Global fonksiyon — HMM rejim tespiti."""
    return _detector.fit(returns)


def get_current_regime() -> str:
    return _detector.get_current_regime()


def get_risk_multiplier() -> float:
    return _detector.get_risk_multiplier()
