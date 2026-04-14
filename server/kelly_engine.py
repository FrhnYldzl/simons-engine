"""
kelly_engine.py — FP-05: Kelly Criterion & Risk Engine

Berlekamp'ın Shannon/Kelly geleneğinden optimal pozisyon boyutlandırma.

Berlekamp, Claude Shannon'ın öğrencisi ve John Kelly'nin Bell Labs'taki
meslektaşıydı. 1990'da Medallion'u yeniden tasarladı — sonuç: %77.8
brüt getiri.

Pratikte yarım Kelly kullanılır (half-Kelly):
  f* = (bp - q) / b   →   f_actual = f* / 2

Drawdown circuit breakers:
  -5%  → %25 pozisyon azalt
  -10% → %50 pozisyon azalt
  -15% → Tümünü kapat, 48 saat bekle

Refs:
  - Kelly (1956). A New Interpretation of Information Rate.
  - Thorp (2006). The Kelly Criterion.
  - Berlekamp (2015). The Kelly Criterion — An Exposition.
  - Zuckerman (2019), Ch. 8-9 — Berlekamp's overhaul.
"""

import numpy as np
from typing import Optional


class KellyEngine:
    """
    Fractional Kelly pozisyon boyutlandırma + risk yönetimi.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,     # Yarım Kelly (daha güvenli)
        max_position_pct: float = 0.05,  # Tek pozisyon max %5
        max_portfolio_risk: float = 0.02, # Portföy risk limiti %2/gün
        max_drawdown_pct: float = 0.15,  # Max drawdown %15
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_pct = max_drawdown_pct

        # Drawdown tracking
        self._peak_equity = 0
        self._current_drawdown = 0
        self._circuit_breaker_active = False
        self._risk_multiplier = 1.0

    def calculate_position_size(
        self,
        equity: float,
        signal_direction: float,
        signal_conviction: float,
        signal_p_value: float,
        entry_price: float,
        stop_loss_price: float,
        regime_risk_mult: float = 1.0,
        current_portfolio_risk: float = 0,
    ) -> dict:
        """
        Kelly Criterion ile optimal pozisyon boyutu hesapla.

        Returns:
            dict with qty, dollar_amount, risk_pct, kelly_f, etc.
        """
        if equity <= 0 or entry_price <= 0 or self._circuit_breaker_active:
            return self._zero_position("Circuit breaker aktif" if self._circuit_breaker_active else "Yetersiz sermaye")

        # Kelly fraction hesaplama
        # f* = (p * b - q) / b
        # p = win probability (conviction bazlı)
        # b = win/loss ratio (entry-to-tp / entry-to-sl)
        # q = 1 - p

        win_prob = 0.5 + signal_conviction * 0.25  # %50-75 arası
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share <= 0:
            return self._zero_position("Stop loss = Entry")

        # Win/loss oranı (risk/reward)
        # Ortalama kazanç = risk * 1.5 (conservative)
        reward_ratio = 1.5 + signal_conviction  # 1.5x - 2.5x arası

        kelly_f = (win_prob * reward_ratio - (1 - win_prob)) / reward_ratio
        kelly_f = max(kelly_f, 0)  # Negatif Kelly = don't trade

        # Fractional Kelly
        adjusted_f = kelly_f * self.kelly_fraction

        # Regime adjustment
        adjusted_f *= regime_risk_mult

        # Drawdown adjustment
        adjusted_f *= self._risk_multiplier

        # Max position limit
        adjusted_f = min(adjusted_f, self.max_position_pct)

        # Dollar amount
        dollar_amount = equity * adjusted_f

        # Quantity
        qty = int(dollar_amount / entry_price)

        # Risk calculation
        position_risk = qty * risk_per_share
        position_risk_pct = position_risk / equity if equity > 0 else 0

        # Portfolio risk check
        total_risk = current_portfolio_risk + position_risk_pct
        if total_risk > self.max_portfolio_risk:
            # Azalt
            allowed_risk = max(0, self.max_portfolio_risk - current_portfolio_risk)
            if allowed_risk > 0 and risk_per_share > 0:
                qty = int((equity * allowed_risk) / risk_per_share)
                position_risk_pct = allowed_risk
            else:
                return self._zero_position("Portfolio risk limiti aşıldı")

        if qty <= 0:
            return self._zero_position("Hesaplanan miktar 0")

        return {
            "qty": qty,
            "side": "buy" if signal_direction > 0 else "sell",
            "dollar_amount": round(qty * entry_price, 2),
            "position_pct": round(qty * entry_price / equity * 100, 2),
            "risk_per_share": round(risk_per_share, 2),
            "position_risk_pct": round(position_risk_pct * 100, 4),
            "kelly_raw": round(kelly_f, 4),
            "kelly_adjusted": round(adjusted_f, 4),
            "win_probability": round(win_prob, 3),
            "reward_ratio": round(reward_ratio, 2),
            "regime_mult": regime_risk_mult,
            "drawdown_mult": self._risk_multiplier,
            "circuit_breaker": self._circuit_breaker_active,
        }

    def update_equity(self, current_equity: float, initial_equity: float = 100_000):
        """
        Equity güncellemesi — drawdown tracking ve circuit breakers.

        Eşikler:
          -5%  → %25 pozisyon azalt
          -10% → %50 pozisyon azalt
          -15% → Tümünü kapat
        """
        self._peak_equity = max(self._peak_equity, current_equity)

        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            self._current_drawdown = 0

        # Circuit breakers
        if self._current_drawdown >= self.max_drawdown_pct:
            self._circuit_breaker_active = True
            self._risk_multiplier = 0
        elif self._current_drawdown >= 0.10:
            self._risk_multiplier = 0.5
            self._circuit_breaker_active = False
        elif self._current_drawdown >= 0.05:
            self._risk_multiplier = 0.75
            self._circuit_breaker_active = False
        else:
            self._risk_multiplier = 1.0
            self._circuit_breaker_active = False

    def reset_circuit_breaker(self):
        """Manuel circuit breaker reset."""
        self._circuit_breaker_active = False
        self._risk_multiplier = 1.0

    def get_risk_status(self) -> dict:
        return {
            "peak_equity": round(self._peak_equity, 2),
            "current_drawdown_pct": round(self._current_drawdown * 100, 2),
            "risk_multiplier": self._risk_multiplier,
            "circuit_breaker_active": self._circuit_breaker_active,
            "kelly_fraction": self.kelly_fraction,
            "max_position_pct": self.max_position_pct * 100,
            "max_portfolio_risk_pct": self.max_portfolio_risk * 100,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
        }

    def calculate_stop_loss(self, entry_price: float, atr: float,
                            direction: str = "long", multiplier: float = 2.0) -> float:
        """ATR bazlı stop loss."""
        if direction == "long":
            return round(entry_price - atr * multiplier, 2)
        else:
            return round(entry_price + atr * multiplier, 2)

    def _zero_position(self, reason: str) -> dict:
        return {
            "qty": 0,
            "side": "none",
            "dollar_amount": 0,
            "position_pct": 0,
            "risk_per_share": 0,
            "position_risk_pct": 0,
            "kelly_raw": 0,
            "kelly_adjusted": 0,
            "reason": reason,
        }
