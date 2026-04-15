"""
main.py -- Simons Engine Server v1.5.1 (Railway-safe)
"""

import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

print(f"[Boot] Python {sys.version}")
print(f"[Boot] CWD: {os.getcwd()}")
print(f"[Boot] PORT: {os.environ.get('PORT', 'NOT SET')}")
print(f"[Boot] Files: {os.listdir('.')}")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Safe imports
try:
    from database import (
        init_db, log_trade, log_signal, log_regime, log_scan,
        log_daily_performance, get_recent_trades, get_daily_performance,
        get_regime_history, get_trade_count_today,
    )
    DB_OK = True
    print("[Boot] database OK")
except Exception as e:
    DB_OK = False
    print(f"[Boot] database FAIL: {e}")

try:
    from data_pipeline import fetch_and_clean, get_returns_matrix
    PIPELINE_OK = True
    print("[Boot] data_pipeline OK")
except Exception as e:
    PIPELINE_OK = False
    print(f"[Boot] data_pipeline FAIL: {e}")

try:
    from hmm_regime import detect_regime
    HMM_OK = True
    print("[Boot] hmm_regime OK")
except Exception as e:
    HMM_OK = False
    print(f"[Boot] hmm_regime FAIL: {e}")

try:
    from signal_engine import SignalEngine
    SIGNAL_OK = True
    print("[Boot] signal_engine OK")
except Exception as e:
    SIGNAL_OK = False
    print(f"[Boot] signal_engine FAIL: {e}")

try:
    from kelly_engine import KellyEngine
    KELLY_OK = True
    print("[Boot] kelly_engine OK")
except Exception as e:
    KELLY_OK = False
    print(f"[Boot] kelly_engine FAIL: {e}")

try:
    import numpy as np
    print("[Boot] numpy OK")
except Exception as e:
    print(f"[Boot] numpy FAIL: {e}")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    SCHED_OK = True
    print("[Boot] apscheduler OK")
except Exception as e:
    SCHED_OK = False
    print(f"[Boot] apscheduler FAIL: {e}")

# ─── Version ──────────────────────────────────────────
VERSION = "1.5.1"
PHASE = 6
PHASE_NAME = "Autonomous Scheduler"

# ─── Universe ─────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "INTC", "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH", "XOM", "CVX",
    "WMT", "COST", "HD",
    "SPY", "QQQ", "IWM",
    "MARA", "COIN", "NFLX", "SQ", "SHOP",
]

# ─── Engine State ─────────────────────────────────────
broker = None
_signal_engine = SignalEngine(p_threshold=0.10) if SIGNAL_OK else None
_kelly_engine = KellyEngine() if KELLY_OK else None
_cached_data = {}
_scheduler = BackgroundScheduler(timezone="UTC") if SCHED_OK else None
_auto_execute = True
SCAN_INTERVAL_MINUTES = 10
MAX_DAILY_TRADES = 3
_scan_count = 0

_last_scan = {
    "status": "waiting",
    "timestamp": None,
    "market_open": False,
    "regime": {"regime": "normal", "regime_name": "Normal", "regime_color": "#3b82f6",
               "risk_multiplier": 1.0, "n_states": 0, "state_probabilities": {},
               "regime_change_probability": 0, "method": "not_scanned_yet"},
    "signals": [],
    "valid_signal_count": 0,
    "trades_executed": [],
    "portfolio": {},
    "risk_status": {"peak_equity": 100000.0, "current_drawdown_pct": 0.0,
                    "risk_multiplier": 1.0, "circuit_breaker_active": False,
                    "kelly_fraction": 0.5, "max_position_pct": 5.0,
                    "max_portfolio_risk_pct": 2.0, "max_drawdown_pct": 15.0},
    "universe_size": len(UNIVERSE),
    "scheduler_active": False,
    "scan_count": 0,
}


def init_broker():
    global broker
    try:
        from broker import SimonsBroker
        broker = SimonsBroker()
        acc = broker.get_account()
        print(f"[Simons] Broker OK -- Equity: ${acc['equity']:,.2f}")
    except Exception as e:
        print(f"[Simons] Broker FAIL (mock mode): {e}")
        broker = None


def _get_account_data():
    if broker:
        try:
            acc = broker.get_account()
            pos = broker.get_positions()
            return {**acc, "positions": pos, "n_positions": len(pos)}
        except Exception as e:
            print(f"[Simons] Account err: {e}")
    return {"cash": 100000.0, "equity": 100000.0, "portfolio_value": 100000.0,
            "buying_power": 200000.0, "day_trade_count": 0,
            "positions": [], "n_positions": 0}


def _is_market_open():
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    mo = now.replace(hour=13, minute=30, second=0, microsecond=0)
    mc = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return mo <= now <= mc


def run_scan(auto_execute=None):
    global _last_scan, _cached_data, _scan_count
    if auto_execute is None:
        auto_execute = _auto_execute
    _scan_count += 1
    ts = datetime.now(timezone.utc).isoformat()
    market_open = _is_market_open()
    print(f"\n[Scan] #{_scan_count} -- market={'OPEN' if market_open else 'CLOSED'}")

    if not broker or not PIPELINE_OK:
        _last_scan["status"] = "no_broker" if not broker else "no_pipeline"
        _last_scan["timestamp"] = ts
        return

    try:
        # 1. Data
        data = fetch_and_clean(broker, UNIVERSE, days=252)
        if not data:
            _last_scan["status"] = "no_data"
            _last_scan["timestamp"] = ts
            return
        _cached_data = data
        print(f"[Scan] {len(data)} tickers")

        # 2. HMM
        regime_result = {"regime": "normal", "regime_name": "Normal",
                         "risk_multiplier": 1.0, "method": "fallback"}
        if HMM_OK and "SPY" in data and "returns" in data["SPY"].columns:
            spy_ret = data["SPY"]["returns"].dropna().values
            if len(spy_ret) > 60:
                regime_result = detect_regime(spy_ret)
        regime = regime_result.get("regime", "normal")
        risk_mult = regime_result.get("risk_multiplier", 1.0)
        print(f"[Scan] Regime: {regime} (risk: {risk_mult})")

        if DB_OK:
            log_regime(regime=regime, n_states=regime_result.get("n_states", 0),
                       state_probs=regime_result.get("state_probabilities", {}),
                       change_prob=regime_result.get("regime_change_probability", 0),
                       trans_matrix=regime_result.get("transition_matrix", []))

        # 3. Signals
        all_signals = []
        valid_signals = []
        if SIGNAL_OK and _signal_engine:
            all_signals = _signal_engine.generate_signals(data, regime=regime)
            valid_signals = [s for s in all_signals if s.is_valid(_signal_engine.p_threshold)]
            print(f"[Scan] {len(all_signals)} signals, {len(valid_signals)} valid")
            if DB_OK:
                for s in all_signals:
                    log_signal(ticker=s.ticker, signal_name=s.name, direction=s.direction,
                               conviction=s.conviction, p_value=s.p_value,
                               alpha_estimate=s.alpha_estimate, metadata=s.metadata)

        # 4. Kelly + sizing
        acc = _get_account_data()
        equity = acc.get("equity", 100000)
        positions = acc.get("positions", [])
        sizing_results = []
        trades_executed = []

        if KELLY_OK and _kelly_engine and valid_signals:
            _kelly_engine.update_equity(equity)
            held = {p["ticker"] for p in positions}
            cur_risk = sum(abs(p.get("unrealized_pl", 0)) / equity for p in positions) if equity > 0 else 0

            for sig in valid_signals:
                if sig.ticker in held or sig.ticker in ("SPY", "QQQ", "IWM") or sig.conviction < 0.2:
                    continue
                td = data.get(sig.ticker)
                if td is None or len(td) < 20:
                    continue
                try:
                    price = float(td["close"].values[-1])
                    atr_s = td["true_range"].rolling(14).mean()
                    atr = float(atr_s.values[-1]) if not np.isnan(atr_s.values[-1]) else price * 0.02
                    sl_dir = "long" if sig.direction > 0 else "short"
                    sl = _kelly_engine.calculate_stop_loss(price, atr, sl_dir)
                    sz = _kelly_engine.calculate_position_size(
                        equity=equity, signal_direction=sig.direction,
                        signal_conviction=sig.conviction, signal_p_value=sig.p_value,
                        entry_price=price, stop_loss_price=sl,
                        regime_risk_mult=risk_mult, current_portfolio_risk=cur_risk)
                    sizing_results.append({"ticker": sig.ticker, "signal": sig.name,
                        "side": sz.get("side", "none"), "qty": sz.get("qty", 0),
                        "price": price, "dollar_amount": sz.get("dollar_amount", 0),
                        "kelly_adjusted": sz.get("kelly_adjusted", 0), "stop_loss": sl,
                        "conviction": sig.conviction, "p_value": sig.p_value,
                        "position_risk_pct": sz.get("position_risk_pct", 0)})
                except Exception:
                    pass

            # 5. Execute
            if auto_execute and market_open:
                trades_today = get_trade_count_today() if DB_OK else 0
                for sz in sizing_results:
                    if trades_today + len(trades_executed) >= MAX_DAILY_TRADES:
                        break
                    if sz["qty"] <= 0:
                        continue
                    try:
                        res = broker.execute_market(sz["ticker"], sz["qty"], sz["side"])
                        if res.get("status") == "submitted":
                            trades_executed.append(sz)
                            if DB_OK:
                                log_trade(ticker=sz["ticker"], side=sz["side"], qty=sz["qty"],
                                          price=sz["price"], signal_name=sz["signal"],
                                          signal_direction=1.0 if sz["side"] == "buy" else -1.0,
                                          signal_conviction=sz["conviction"], signal_p_value=sz["p_value"],
                                          kelly_f=sz["kelly_adjusted"], regime=regime, stop_loss=sz["stop_loss"])
                            print(f"[Scan] TRADE: {sz['side'].upper()} {sz['ticker']} x{sz['qty']}")
                    except Exception as e:
                        print(f"[Scan] Exec err {sz['ticker']}: {e}")

        # Dashboard signals
        dash_sigs = [s.to_dict() for s in all_signals[:20]]
        peak = max(_last_scan["risk_status"].get("peak_equity", 0), equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0

        _last_scan = {
            "status": "ok", "timestamp": ts, "market_open": market_open,
            "regime": regime_result, "signals": dash_sigs,
            "valid_signal_count": len(valid_signals),
            "trades_executed": trades_executed,
            "portfolio": {"equity": equity, "cash": acc.get("cash", 0),
                          "buying_power": acc.get("buying_power", 0),
                          "positions": positions, "n_positions": len(positions)},
            "risk_status": {"peak_equity": round(peak, 2), "current_drawdown_pct": round(dd, 2),
                            "risk_multiplier": risk_mult,
                            "circuit_breaker_active": _kelly_engine._circuit_breaker_active if _kelly_engine else False,
                            "kelly_fraction": 0.5, "max_position_pct": 5.0,
                            "max_portfolio_risk_pct": 2.0, "max_drawdown_pct": 15.0},
            "sizing_results": sizing_results, "universe_size": len(data),
            "scheduler_active": _scheduler.running if _scheduler else False,
            "scan_count": _scan_count, "auto_execute": _auto_execute,
        }

        if DB_OK:
            log_scan(regime=regime, n_signals=len(valid_signals),
                     n_trades=len(trades_executed), portfolio_value=equity,
                     summary=f"#{_scan_count} {regime} {len(valid_signals)}sig {len(trades_executed)}trade")

        print(f"[Scan] Done: {regime} | {len(valid_signals)}sig | {len(trades_executed)}trade")

    except Exception as e:
        print(f"[Scan] ERROR: {e}")
        traceback.print_exc()
        _last_scan["status"] = f"error: {str(e)}"
        _last_scan["timestamp"] = ts


# ─── Lifespan ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Simons] v{VERSION} starting (Phase {PHASE})")

    if DB_OK:
        init_db()

    init_broker()

    # Scheduler + deferred first scan
    if broker and _scheduler and SCHED_OK:
        _scheduler.add_job(
            func=lambda: run_scan(auto_execute=_auto_execute),
            trigger=IntervalTrigger(minutes=SCAN_INTERVAL_MINUTES),
            id="simons_scan", replace_existing=True)
        _scheduler.add_job(
            func=lambda: run_scan(auto_execute=False),
            trigger=DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=15)),
            id="simons_first_scan", replace_existing=True)
        _scheduler.start()
        print(f"[Simons] Scheduler ON -- every {SCAN_INTERVAL_MINUTES}min, first scan in 15s")

    print(f"[Simons] READY on port {os.environ.get('PORT', '8000')}")
    yield
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    print("[Simons] Stopped")


app = FastAPI(title="Simons Engine", version=VERSION, lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = static_dir / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<h1>Simons Engine</h1><p>Dashboard loading...</p>"


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": VERSION, "phase": PHASE,
            "broker": broker is not None, "db": DB_OK, "hmm": HMM_OK,
            "signals": SIGNAL_OK, "kelly": KELLY_OK, "scheduler": SCHED_OK,
            "regime": _last_scan["regime"].get("regime", "unknown"),
            "market_open": _is_market_open(), "scan_count": _scan_count}


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
    run_scan(auto_execute=_auto_execute)
    return _last_scan


@app.post("/api/toggle-execute")
async def toggle_execute():
    global _auto_execute
    _auto_execute = not _auto_execute
    return {"auto_execute": _auto_execute}


@app.get("/api/trades")
async def trades(limit: int = 50):
    return get_recent_trades(limit) if DB_OK else []


@app.get("/api/performance")
async def performance(days: int = 30):
    return get_daily_performance(days) if DB_OK else []


@app.get("/api/regime-history")
async def regime_history(limit: int = 100):
    return get_regime_history(limit) if DB_OK else []


@app.get("/api/risk-status")
async def risk_status():
    return _last_scan.get("risk_status", {})


@app.get("/api/stats")
async def stats():
    return {"phase": PHASE, "version": VERSION,
            "broker": broker is not None, "db": DB_OK, "hmm": HMM_OK,
            "scan_count": _scan_count, "universe": len(UNIVERSE),
            "cached": len(_cached_data),
            "scheduler": _scheduler.running if _scheduler else False}
