"""
scheduler.py — Simons Engine Autonomous Scheduler

Tamamen matematik tabanlı otonom trading döngüsü.
AI API kullanmaz — sadece istatistik ve HMM.

Döngü (her 10 dakika):
  1. Data Pipeline → temiz veri
  2. HMM Regime → piyasa durumu
  3. Signal Engine → istatistiksel sinyaller
  4. Kelly Engine → pozisyon boyutlandırma
  5. Execution → Alpaca'da işlem
  6. Logging → SQLite'a kayıt
"""

import json
import numpy as np
import traceback
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from data_pipeline import fetch_and_clean, get_returns_matrix
from hmm_regime import detect_regime, get_current_regime, get_risk_multiplier
from signal_engine import SignalEngine
from kelly_engine import KellyEngine
from database import (
    init_db, log_trade, log_signal, log_regime, log_scan,
    get_trade_count_today,
)


# Watchlist — çeşitlendirilmiş evren
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Semis
    "AMD", "AVGO", "INTC",
    # Finance
    "JPM", "GS", "BAC",
    # Healthcare
    "JNJ", "PFE", "UNH",
    # Energy
    "XOM", "CVX",
    # Consumer
    "WMT", "COST", "HD",
    # ETFs (benchmark)
    "SPY", "QQQ", "IWM",
    # Crypto-adjacent
    "MARA", "COIN",
    # Volatility plays
    "NFLX", "SQ", "SHOP",
]

# Son tarama sonucu (dashboard için)
_last_scan: dict = {
    "status": "Henuz tarama yapilmadi",
    "timestamp": None,
    "regime": {},
    "signals": [],
    "trades_executed": [],
    "portfolio": {},
    "market_data_summary": {},
}

scheduler = BackgroundScheduler(timezone="UTC")
_signal_engine = SignalEngine(p_threshold=0.05)
_kelly_engine = KellyEngine()


def is_market_open() -> bool:
    """NYSE açık mı? (13:30-20:00 UTC, Mon-Fri)"""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def run_scan(broker=None, auto_execute: bool = True):
    """
    Ana tarama döngüsü — Simons Engine.

    Tamamen matematik tabanlı:
    1. Veri al ve temizle
    2. HMM ile rejim tespit et
    3. İstatistiksel sinyaller üret
    4. Kelly ile pozisyon boyutlandır
    5. Execute et
    """
    global _last_scan

    timestamp = datetime.now(timezone.utc).isoformat()
    market_open = is_market_open()

    print(f"\n[Simons] {'='*50}")
    print(f"[Simons] Tarama başlıyor — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"[Simons] Piyasa: {'AÇIK' if market_open else 'KAPALI'}")

    if broker is None:
        _last_scan = {"status": "Broker bağlantısı yok", "timestamp": timestamp}
        return

    try:
        # ─── 1. Data Pipeline ─────────────────────────────
        print("[Simons] 1/5 Data Pipeline...")
        data = fetch_and_clean(broker, UNIVERSE, days=252)
        if not data:
            _last_scan = {"status": "Veri alınamadı", "timestamp": timestamp}
            return
        print(f"[Simons]   → {len(data)} ticker verisi alındı")

        # ─── 2. HMM Regime Detection ─────────────────────
        print("[Simons] 2/5 HMM Regime Detection...")
        # SPY return'lerini kullan (market proxy)
        spy_returns = None
        if "SPY" in data and "returns" in data["SPY"].columns:
            spy_returns = data["SPY"]["returns"].dropna().values

        if spy_returns is not None and len(spy_returns) > 60:
            regime_result = detect_regime(spy_returns)
        else:
            regime_result = {"regime": "normal", "regime_name": "Normal",
                            "risk_multiplier": 1.0, "method": "no_data"}

        current_regime = regime_result.get("regime", "normal")
        risk_mult = regime_result.get("risk_multiplier", 1.0)
        print(f"[Simons]   → Rejim: {current_regime} (risk mult: {risk_mult})")

        # Log regime
        log_regime(
            regime=current_regime,
            n_states=regime_result.get("n_states", 0),
            state_probs=regime_result.get("state_probabilities", {}),
            change_prob=regime_result.get("regime_change_probability", 0),
            trans_matrix=regime_result.get("transition_matrix", []),
        )

        # ─── 3. Signal Generation ────────────────────────
        print("[Simons] 3/5 Signal Generation...")
        signals = _signal_engine.generate_signals(data, regime=current_regime)
        valid_signals = [s for s in signals if s.is_valid()]
        print(f"[Simons]   → {len(signals)} sinyal üretildi, {len(valid_signals)} geçerli (p < 0.05)")

        # Log signals
        for s in signals:
            log_signal(
                ticker=s.ticker, signal_name=s.name,
                direction=s.direction, conviction=s.conviction,
                p_value=s.p_value, alpha_estimate=s.alpha_estimate,
                metadata=s.metadata,
            )

        # ─── 4. Portfolio State ───────────────────────────
        account = broker.get_account()
        positions = broker.get_positions()
        equity = account.get("equity", 100_000)

        # Kelly engine equity güncelle
        _kelly_engine.update_equity(equity)

        # Mevcut pozisyon risk hesabı
        current_risk = sum(
            abs(p.get("unrealized_pl", 0)) / equity
            for p in positions
        ) if equity > 0 else 0

        # ─── 5. Execution ────────────────────────────────
        trades_executed = []

        if auto_execute and market_open and valid_signals:
            print("[Simons] 5/5 Execution...")

            # Mevcut ticker'lar
            held_tickers = {p["ticker"] for p in positions}

            # Max günlük trade (PDT koruması)
            trades_today = get_trade_count_today()
            max_daily = 3  # Conservative

            for signal in valid_signals:
                if trades_today + len(trades_executed) >= max_daily:
                    print(f"[Simons]   → Günlük trade limiti ({max_daily}) doldu")
                    break

                # Zaten pozisyonumuz var mı?
                if signal.ticker in held_tickers:
                    continue

                # ETF'lerde trade yapma (benchmark olarak kullan)
                if signal.ticker in ("SPY", "QQQ", "IWM"):
                    continue

                # Conviction çok düşükse atla
                if signal.conviction < 0.3:
                    continue

                try:
                    # Fiyat al
                    ticker_data = data.get(signal.ticker)
                    if ticker_data is None or len(ticker_data) < 20:
                        continue

                    price = ticker_data["close"].values[-1]
                    atr = ticker_data["true_range"].rolling(14).mean().values[-1]

                    if price <= 0 or np.isnan(atr):
                        continue

                    # Stop loss
                    sl_direction = "long" if signal.direction > 0 else "short"
                    stop_loss = _kelly_engine.calculate_stop_loss(price, atr, sl_direction)

                    # Kelly position sizing
                    sizing = _kelly_engine.calculate_position_size(
                        equity=equity,
                        signal_direction=signal.direction,
                        signal_conviction=signal.conviction,
                        signal_p_value=signal.p_value,
                        entry_price=price,
                        stop_loss_price=stop_loss,
                        regime_risk_mult=risk_mult,
                        current_portfolio_risk=current_risk,
                    )

                    qty = sizing.get("qty", 0)
                    if qty <= 0:
                        continue

                    side = sizing.get("side", "buy")

                    # Execute
                    result = broker.execute_market(signal.ticker, qty, side)

                    if result.get("status") == "submitted":
                        trade_info = {
                            "ticker": signal.ticker,
                            "side": side,
                            "qty": qty,
                            "price": price,
                            "signal": signal.name,
                            "conviction": signal.conviction,
                            "p_value": signal.p_value,
                            "kelly_f": sizing.get("kelly_adjusted", 0),
                            "stop_loss": stop_loss,
                            "regime": current_regime,
                        }
                        trades_executed.append(trade_info)

                        # DB log
                        log_trade(
                            ticker=signal.ticker, side=side, qty=qty, price=price,
                            signal_name=signal.name,
                            signal_direction=signal.direction,
                            signal_conviction=signal.conviction,
                            signal_p_value=signal.p_value,
                            kelly_f=sizing.get("kelly_adjusted", 0),
                            regime=current_regime,
                            stop_loss=stop_loss,
                        )

                        print(f"[Simons]   → {side.upper()} {signal.ticker} x{qty} @ ${price:.2f} "
                              f"(p={signal.p_value:.4f}, conv={signal.conviction:.2f}, kelly={sizing.get('kelly_adjusted', 0):.3f})")

                except Exception as e:
                    print(f"[Simons]   → HATA {signal.ticker}: {e}")

        elif not market_open:
            print("[Simons] 5/5 Piyasa kapalı — sadece analiz modu")
        elif not valid_signals:
            print("[Simons] 5/5 Geçerli sinyal yok — bekleme modu")

        # ─── Summary ─────────────────────────────────────
        _last_scan = {
            "status": "ok",
            "timestamp": timestamp,
            "market_open": market_open,
            "regime": regime_result,
            "signals": [s.to_dict() for s in signals[:20]],
            "valid_signal_count": len(valid_signals),
            "trades_executed": trades_executed,
            "portfolio": {
                "equity": equity,
                "cash": account.get("cash", 0),
                "buying_power": account.get("buying_power", 0),
                "positions": positions,
                "n_positions": len(positions),
            },
            "risk_status": _kelly_engine.get_risk_status(),
            "universe_size": len(data),
        }

        # Log scan
        log_scan(
            regime=current_regime,
            n_signals=len(valid_signals),
            n_trades=len(trades_executed),
            portfolio_value=equity,
            summary=f"Rejim: {current_regime}, {len(valid_signals)} sinyal, {len(trades_executed)} trade",
        )

        print(f"[Simons] Tarama tamamlandı | Rejim: {current_regime} | "
              f"{len(valid_signals)} sinyal | {len(trades_executed)} trade")
        print(f"[Simons] {'='*50}\n")

    except Exception as e:
        print(f"[Simons] KRITIK HATA: {e}")
        traceback.print_exc()
        _last_scan = {"status": f"Hata: {str(e)}", "timestamp": timestamp}


def get_last_scan() -> dict:
    return _last_scan


def start(broker=None, auto_execute: bool = True, interval_minutes: int = 10):
    """Scheduler başlat."""
    if scheduler.running:
        return

    init_db()

    # Ana tarama döngüsü
    scheduler.add_job(
        func=lambda: run_scan(broker=broker, auto_execute=auto_execute),
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="simons_scan",
        replace_existing=True,
    )

    # İlk tarama 10 saniye sonra
    scheduler.add_job(
        func=lambda: run_scan(broker=broker, auto_execute=auto_execute),
        trigger="date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=10),
        id="simons_first_scan",
        replace_existing=True,
    )

    scheduler.start()
    print(f"[Simons Scheduler] Başlatıldı — her {interval_minutes}dk, auto_execute={auto_execute}")


def stop():
    if scheduler.running:
        scheduler.shutdown(wait=False)
