"""
main.py -- Simons Engine Server v1.5 (FAZ 6)

Jim Simons / Medallion Fund inspired quantitative trading engine.
Tamamen matematik tabanli -- AI API kullanmaz.

FAZ 6: APScheduler otonom dongu + auto_execute
       Her 10 dakikada: Data -> HMM -> Signals -> Kelly -> Execute
"""

import os
import sys
import traceback
import numpy as np
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from database import (
    init_db, log_trade, log_signal, log_regime, log_scan,
    log_daily_performance, get_recent_trades, get_daily_performance,
    get_regime_history, get_trade_count_today,
)
from data_pipeline import fetch_and_clean, get_returns_matrix
from hmm_regime import detect_regime, get_current_regime, get_risk_multiplier
from signal_engine import SignalEngine
from kelly_engine import KellyEngine

# ─── Version & Phase Tracking ─────────────────────────
VERSION = "1.5.0"
PHASE = 6
PHASE_NAME = "Autonomous Scheduler"

# ─── Universe ─────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "INTC",
    "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH",
    "XOM", "CVX",
    "WMT", "COST", "HD",
    "SPY", "QQQ", "IWM",
    "MARA", "COIN",
    "NFLX", "SQ", "SHOP",
]

# ─── Engine Components ────────────────────────────────
broker = None
_signal_engine = SignalEngine(p_threshold=0.10)
_kelly_engine = KellyEngine(
    kelly_fraction=0.5,
    max_position_pct=0.05,
    max_portfolio_risk=0.02,
    max_drawdown_pct=0.15,
)
_cached_data = {}
_scheduler = BackgroundScheduler(timezone="UTC")
_auto_execute = True  # Piyasa acikken otomatik trade
SCAN_INTERVAL_MINUTES = 10
MAX_DAILY_TRADES = 3


def init_broker():
    global broker
    try:
        from broker import SimonsBroker
        broker = SimonsBroker()
        acc = broker.get_account()
        equity = acc['equity']
        _kelly_engine.update_equity(equity)
        print(f"[Simons] Broker baglandi -- Equity: ${equity:,.2f}")
    except Exception as e:
        print(f"[Simons] Broker hatasi (mock mode): {e}")
        broker = None


def _get_account_data() -> dict:
    if broker:
        try:
            acc = broker.get_account()
            positions = broker.get_positions()
            return {**acc, "positions": positions, "n_positions": len(positions)}
        except Exception as e:
            print(f"[Simons] Account hatasi: {e}")
    return {
        "cash": 100000.0, "equity": 100000.0, "portfolio_value": 100000.0,
        "buying_power": 200000.0, "day_trade_count": 0,
        "positions": [], "n_positions": 0,
    }


def _is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# ─── Scan State ───────────────────────────────────────
_last_scan = {
    "status": "waiting",
    "timestamp": None,
    "market_open": False,
    "regime": {
        "regime": "normal", "regime_name": "Normal", "regime_color": "#3b82f6",
        "risk_multiplier": 1.0, "n_states": 0, "state_probabilities": {},
        "regime_change_probability": 0, "method": "not_scanned_yet",
    },
    "signals": [],
    "valid_signal_count": 0,
    "trades_executed": [],
    "portfolio": {},
    "risk_status": _kelly_engine.get_risk_status(),
    "universe_size": len(UNIVERSE),
    "scheduler_active": False,
    "scan_count": 0,
    "next_scan": None,
}

_scan_count = 0


def run_scan(auto_execute: bool = None):
    """
    Ana tarama dongusu -- Simons Engine.
    1. Data Pipeline -> temiz veri
    2. HMM Regime -> piyasa durumu
    3. Signal Engine -> istatistiksel sinyaller
    4. Kelly Engine -> pozisyon boyutlandirma
    5. Execution -> Alpaca'da islem (auto_execute + market_open ise)
    """
    global _last_scan, _cached_data, _scan_count

    if auto_execute is None:
        auto_execute = _auto_execute

    _scan_count += 1
    timestamp = datetime.now(timezone.utc).isoformat()
    market_open = _is_market_open()

    print(f"\n[Simons] {'='*50}")
    print(f"[Simons] Tarama #{_scan_count} -- {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"[Simons] Piyasa: {'ACIK' if market_open else 'KAPALI'} | Auto-Execute: {auto_execute}")

    if not broker:
        _last_scan["status"] = "Broker baglantisi yok"
        _last_scan["timestamp"] = timestamp
        return

    try:
        # ─── 1. Data Pipeline ─────────────────────────────
        print("[Simons] 1/5 Data Pipeline...")
        data = fetch_and_clean(broker, UNIVERSE, days=252)
        if not data:
            _last_scan = {**_last_scan, "status": "Veri alinamadi", "timestamp": timestamp}
            return
        _cached_data = data
        print(f"[Simons]   -> {len(data)} ticker verisi alindi")

        # ─── 2. HMM Regime Detection ─────────────────────
        print("[Simons] 2/5 HMM Regime Detection...")
        spy_returns = None
        if "SPY" in data and "returns" in data["SPY"].columns:
            spy_returns = data["SPY"]["returns"].dropna().values

        if spy_returns is not None and len(spy_returns) > 60:
            regime_result = detect_regime(spy_returns)
        else:
            regime_result = {
                "regime": "normal", "regime_name": "Normal",
                "risk_multiplier": 1.0, "method": "no_spy_data",
            }

        current_regime = regime_result.get("regime", "normal")
        risk_mult = regime_result.get("risk_multiplier", 1.0)
        print(f"[Simons]   -> Rejim: {current_regime} (risk mult: {risk_mult})")

        log_regime(
            regime=current_regime,
            n_states=regime_result.get("n_states", 0),
            state_probs=regime_result.get("state_probabilities", {}),
            change_prob=regime_result.get("regime_change_probability", 0),
            trans_matrix=regime_result.get("transition_matrix", []),
        )

        # ─── 3. Signal Generation ────────────────────────
        print("[Simons] 3/5 Signal Generation...")
        all_signals = _signal_engine.generate_signals(data, regime=current_regime)
        valid_signals = [s for s in all_signals if s.is_valid(_signal_engine.p_threshold)]
        print(f"[Simons]   -> {len(all_signals)} sinyal, {len(valid_signals)} gecerli (p < {_signal_engine.p_threshold})")

        for s in all_signals:
            log_signal(
                ticker=s.ticker, signal_name=s.name,
                direction=s.direction, conviction=s.conviction,
                p_value=s.p_value, alpha_estimate=s.alpha_estimate,
                metadata=s.metadata,
            )

        # ─── 4. Kelly Position Sizing ────────────────────
        print("[Simons] 4/5 Kelly Position Sizing...")
        acc = _get_account_data()
        equity = acc.get("equity", 100000)
        positions = acc.get("positions", [])
        _kelly_engine.update_equity(equity)

        current_risk = sum(
            abs(p.get("unrealized_pl", 0)) / equity for p in positions
        ) if equity > 0 else 0

        held_tickers = {p["ticker"] for p in positions}
        sizing_results = []

        for signal in valid_signals:
            if signal.ticker in held_tickers:
                continue
            if signal.ticker in ("SPY", "QQQ", "IWM"):
                continue
            if signal.conviction < 0.2:
                continue

            ticker_data = data.get(signal.ticker)
            if ticker_data is None or len(ticker_data) < 20:
                continue

            try:
                price = float(ticker_data["close"].values[-1])
                atr_series = ticker_data["true_range"].rolling(14).mean()
                atr = float(atr_series.values[-1]) if not np.isnan(atr_series.values[-1]) else price * 0.02

                sl_direction = "long" if signal.direction > 0 else "short"
                stop_loss = _kelly_engine.calculate_stop_loss(price, atr, sl_direction)

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

                sizing_results.append({
                    "ticker": signal.ticker,
                    "signal": signal.name,
                    "side": sizing.get("side", "none"),
                    "qty": sizing.get("qty", 0),
                    "price": price,
                    "dollar_amount": sizing.get("dollar_amount", 0),
                    "kelly_raw": sizing.get("kelly_raw", 0),
                    "kelly_adjusted": sizing.get("kelly_adjusted", 0),
                    "stop_loss": stop_loss,
                    "position_risk_pct": sizing.get("position_risk_pct", 0),
                    "conviction": signal.conviction,
                    "p_value": signal.p_value,
                })
            except Exception as e:
                print(f"[Simons]   -> Kelly hata {signal.ticker}: {e}")

        print(f"[Simons]   -> {len(sizing_results)} pozisyon boyutlandirildi")

        # ─── 5. Execution ────────────────────────────────
        trades_executed = []
        print(f"[Simons] 5/5 Execution (auto={auto_execute}, market={'ACIK' if market_open else 'KAPALI'})...")

        if auto_execute and market_open and sizing_results:
            trades_today = get_trade_count_today()

            for sizing in sizing_results:
                if trades_today + len(trades_executed) >= MAX_DAILY_TRADES:
                    print(f"[Simons]   -> Gunluk trade limiti ({MAX_DAILY_TRADES}) doldu")
                    break

                qty = sizing.get("qty", 0)
                if qty <= 0:
                    continue

                side = sizing.get("side", "buy")
                ticker = sizing["ticker"]

                try:
                    result = broker.execute_market(ticker, qty, side)
                    if result.get("status") == "submitted":
                        trade_info = {
                            "ticker": ticker,
                            "side": side,
                            "qty": qty,
                            "price": sizing["price"],
                            "signal": sizing["signal"],
                            "conviction": sizing["conviction"],
                            "p_value": sizing["p_value"],
                            "kelly_f": sizing["kelly_adjusted"],
                            "stop_loss": sizing["stop_loss"],
                            "regime": current_regime,
                            "order_id": result.get("order_id", ""),
                        }
                        trades_executed.append(trade_info)

                        log_trade(
                            ticker=ticker, side=side, qty=qty, price=sizing["price"],
                            signal_name=sizing["signal"],
                            signal_direction=1.0 if side == "buy" else -1.0,
                            signal_conviction=sizing["conviction"],
                            signal_p_value=sizing["p_value"],
                            kelly_f=sizing["kelly_adjusted"],
                            regime=current_regime,
                            stop_loss=sizing["stop_loss"],
                        )
                        current_risk += sizing.get("position_risk_pct", 0) / 100

                        print(f"[Simons]   -> TRADE: {side.upper()} {ticker} x{qty} @ ${sizing['price']:.2f} "
                              f"(p={sizing['p_value']:.4f}, kelly={sizing['kelly_adjusted']:.3f})")
                    else:
                        print(f"[Simons]   -> REJECT {ticker}: {result.get('message', 'unknown')}")
                except Exception as e:
                    print(f"[Simons]   -> EXEC HATA {ticker}: {e}")

        elif not market_open:
            print("[Simons]   -> Piyasa kapali -- sadece analiz")
        elif not auto_execute:
            print("[Simons]   -> Auto-execute kapali -- sadece analiz")
        elif not sizing_results:
            print("[Simons]   -> Boyutlandirilmis pozisyon yok")

        if trades_executed:
            print(f"[Simons]   -> {len(trades_executed)} trade yapildi!")
        else:
            print("[Simons]   -> 0 trade")

        # ─── Build Dashboard Data ────────────────────────
        dashboard_signals = []
        for s in all_signals[:20]:
            d = s.to_dict()
            sizing_info = next((r for r in sizing_results if r["ticker"] == s.ticker), None)
            if sizing_info:
                d["kelly_qty"] = sizing_info["qty"]
                d["kelly_adjusted"] = sizing_info["kelly_adjusted"]
                d["stop_loss"] = sizing_info["stop_loss"]
                d["dollar_amount"] = sizing_info["dollar_amount"]
            dashboard_signals.append(d)

        peak = max(_kelly_engine._peak_equity, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0

        # Next scan time
        next_scan = None
        if _scheduler.running:
            job = _scheduler.get_job("simons_scan")
            if job and job.next_run_time:
                next_scan = job.next_run_time.isoformat()

        _last_scan = {
            "status": "ok",
            "timestamp": timestamp,
            "market_open": market_open,
            "regime": regime_result,
            "signals": dashboard_signals,
            "valid_signal_count": len(valid_signals),
            "trades_executed": trades_executed,
            "portfolio": {
                "equity": equity,
                "cash": acc.get("cash", 0),
                "buying_power": acc.get("buying_power", 0),
                "positions": positions,
                "n_positions": len(positions),
            },
            "risk_status": {
                "peak_equity": round(peak, 2),
                "current_drawdown_pct": round(dd, 2),
                "risk_multiplier": risk_mult,
                "circuit_breaker_active": _kelly_engine._circuit_breaker_active,
                "kelly_fraction": _kelly_engine.kelly_fraction,
                "max_position_pct": _kelly_engine.max_position_pct * 100,
                "max_portfolio_risk_pct": _kelly_engine.max_portfolio_risk * 100,
                "max_drawdown_pct": _kelly_engine.max_drawdown_pct * 100,
            },
            "sizing_results": sizing_results,
            "universe_size": len(data),
            "scheduler_active": _scheduler.running,
            "scan_count": _scan_count,
            "next_scan": next_scan,
            "auto_execute": _auto_execute,
            "scan_interval_minutes": SCAN_INTERVAL_MINUTES,
        }

        log_scan(
            regime=current_regime, n_signals=len(valid_signals),
            n_trades=len(trades_executed), portfolio_value=equity,
            summary=f"#{_scan_count} Rejim: {current_regime}, {len(valid_signals)} sinyal, {len(trades_executed)} trade",
        )

        print(f"[Simons] Tarama #{_scan_count} tamamlandi | Rejim: {current_regime} | "
              f"{len(valid_signals)} sinyal | {len(trades_executed)} trade")
        print(f"[Simons] {'='*50}\n")

    except Exception as e:
        print(f"[Simons] KRITIK HATA: {e}")
        traceback.print_exc()
        _last_scan["status"] = f"Hata: {str(e)}"
        _last_scan["timestamp"] = timestamp


def start_scheduler():
    """APScheduler baslat -- her N dakikada bir scan."""
    if _scheduler.running:
        print("[Scheduler] Zaten calisiyor")
        return

    _scheduler.add_job(
        func=lambda: run_scan(auto_execute=_auto_execute),
        trigger=IntervalTrigger(minutes=SCAN_INTERVAL_MINUTES),
        id="simons_scan",
        replace_existing=True,
    )
    _scheduler.start()
    print(f"[Scheduler] Baslatildi -- her {SCAN_INTERVAL_MINUTES}dk, auto_execute={_auto_execute}")


def stop_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        print("[Scheduler] Durduruldu")


def _seed_demo_data():
    import random
    random.seed(42)
    tickers = ["NVDA", "AAPL", "TSLA", "META", "AMD", "MSFT", "GOOGL", "AMZN"]
    signals = ["mean_reversion", "momentum", "volume_anomaly", "gap_reversal", "combined"]
    regimes_list = ["low_volatility", "normal", "high_volatility"]
    for i in range(15):
        ticker = random.choice(tickers)
        side = random.choice(["buy", "sell"])
        qty = random.randint(5, 50)
        price = round(random.uniform(100, 800), 2)
        log_trade(ticker=ticker, side=side, qty=qty, price=price,
                  signal_name=random.choice(signals),
                  signal_direction=1.0 if side == "buy" else -1.0,
                  signal_conviction=round(random.uniform(0.3, 0.9), 3),
                  signal_p_value=round(random.uniform(0.001, 0.05), 4),
                  kelly_f=round(random.uniform(0.01, 0.04), 4),
                  regime=random.choice(regimes_list),
                  stop_loss=round(price * 0.95, 2))
    print(f"[DB] Seed data yazildi: 15 trade")


# ─── Lifespan ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Simons] Engine v{VERSION} baslatiliyor... (Phase {PHASE}: {PHASE_NAME})")
    init_db()

    existing = get_recent_trades(1)
    if not existing:
        _seed_demo_data()

    init_broker()

    # Server HEMEN ayaga kalksin, scan arka planda calissin
    if broker:
        start_scheduler()
        # Ilk scan 15 saniye sonra (Railway health check gecsin)
        from apscheduler.triggers.date import DateTrigger
        _scheduler.add_job(
            func=lambda: run_scan(auto_execute=False),
            trigger=DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=15)),
            id="simons_first_scan",
            replace_existing=True,
        )
        print("[Simons] Ilk tarama 15sn sonra baslatilacak (startup hizlandirma)")

    print(f"[Simons] Server hazir! -> http://localhost:8000")
    yield
    stop_scheduler()
    print("[Simons] Server kapaniyor...")


app = FastAPI(title="Simons Engine", version=VERSION, lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = static_dir / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<h1>Simons Engine -- Dashboard yuklenemiyor</h1>"


# ─── API Endpoints ────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok", "version": VERSION, "phase": PHASE,
        "phase_name": PHASE_NAME, "engine": "Simons Quantitative",
        "ai_api_used": False, "broker_connected": broker is not None,
        "regime": _last_scan["regime"].get("regime", "unknown"),
        "market_open": _is_market_open(),
        "last_scan": _last_scan["timestamp"],
        "scheduler_active": _scheduler.running,
        "auto_execute": _auto_execute,
        "scan_interval": SCAN_INTERVAL_MINUTES,
        "scan_count": _scan_count,
    }


@app.get("/api/account")
async def account():
    return _get_account_data()


@app.get("/api/positions")
async def positions():
    if broker:
        try:
            return broker.get_positions()
        except Exception:
            pass
    return []


@app.get("/api/scan")
async def get_scan():
    return _last_scan


@app.post("/api/scan-now")
async def scan_now():
    """Manuel tarama -- auto_execute parametresine bakar."""
    run_scan(auto_execute=_auto_execute)
    return _last_scan


@app.post("/api/toggle-execute")
async def toggle_execute():
    """Auto-execute toggle."""
    global _auto_execute
    _auto_execute = not _auto_execute
    return {"auto_execute": _auto_execute}


@app.get("/api/trades")
async def trades(limit: int = 50):
    return get_recent_trades(limit)


@app.get("/api/performance")
async def performance(days: int = 30):
    return get_daily_performance(days)


@app.get("/api/regime-history")
async def regime_history(limit: int = 100):
    return get_regime_history(limit)


@app.get("/api/risk-status")
async def risk_status():
    return _last_scan.get("risk_status", {})


@app.get("/api/stats")
async def stats():
    return {
        "total_trades": len(get_recent_trades(1000)),
        "trades_today": get_trade_count_today(),
        "performance_days": len(get_daily_performance(30)),
        "regime_records": len(get_regime_history(100)),
        "db_status": "connected",
        "broker_status": "connected" if broker else "offline",
        "universe_size": len(UNIVERSE),
        "cached_tickers": len(_cached_data),
        "scheduler_active": _scheduler.running,
        "auto_execute": _auto_execute,
        "scan_count": _scan_count,
        "scan_interval": SCAN_INTERVAL_MINUTES,
        "phase": PHASE,
    }
