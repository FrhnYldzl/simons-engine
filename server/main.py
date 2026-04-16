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
        log_ai_decision, approve_ai_decision, reject_ai_decision,
        record_ai_outcome, get_pending_decisions, get_all_decisions,
        log_ai_analysis, export_training_data,
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

# New modules (Phase 9+)
try:
    from backtest_engine import (
        run_backtest, compute_performance_metrics, deflated_sharpe_ratio,
        monte_carlo_stress_test, validate_signal,
        mean_reversion_signal, momentum_signal,
    )
    BACKTEST_OK = True
    print("[Boot] backtest_engine OK")
except Exception as e:
    BACKTEST_OK = False
    print(f"[Boot] backtest_engine FAIL: {e}")

try:
    from portfolio_optimizer import (
        optimize_portfolio, compute_factor_exposures, apply_turnover_penalty,
    )
    PORTFOLIO_OK = True
    print("[Boot] portfolio_optimizer OK")
except Exception as e:
    PORTFOLIO_OK = False
    print(f"[Boot] portfolio_optimizer FAIL: {e}")

try:
    from kernel_engine import kernel_predict, svr_signal, kernel_pca_features
    KERNEL_OK = True
    print("[Boot] kernel_engine OK")
except Exception as e:
    KERNEL_OK = False
    print(f"[Boot] kernel_engine FAIL: {e}")

try:
    from nlp_engine import score_text, score_multiple_texts, ticker_news_sentiment
    NLP_OK = True
    print("[Boot] nlp_engine OK")
except Exception as e:
    NLP_OK = False
    print(f"[Boot] nlp_engine FAIL: {e}")

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
VERSION = "2.0.0"
PHASE = 9
PHASE_NAME = "Full FP-01..09 Modules"

# ─── Universe ─────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "INTC", "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH", "XOM", "CVX",
    "WMT", "COST", "HD",
    "SPY", "QQQ", "IWM",
    "MARA", "COIN", "NFLX", "SQ", "SHOP",
]

# ─── Engine Configuration ─────────────────────────────
# Trading kurallari (canliya gecmeden once sikilastirildi)
P_VALUE_THRESHOLD = 0.05          # Medallion'a yakin sikilik (onceki 0.10)
MIN_CONVICTION = 0.30             # Dusuk conviction sinyallerini reddet
MIN_POSITION_VALUE = 500.0        # $500 altindaki pozisyonlari atla
TICKER_COOLDOWN_HOURS = 24        # Ayni ticker'a 24 saat icinde 2. trade yasak
SCAN_INTERVAL_MINUTES = 30        # Saatte 2 tarama (onceki 10dk cok agresif)
MAX_DAILY_TRADES = 3              # PDT koruma + risk sinirlama
EXIT_CHECK_ENABLED = True         # Acik pozisyonlari her scan'de kontrol et

# ─── Engine State ─────────────────────────────────────
broker = None
_signal_engine = SignalEngine(p_threshold=P_VALUE_THRESHOLD) if SIGNAL_OK else None
_kelly_engine = KellyEngine() if KELLY_OK else None
_cached_data = {}
_scheduler = BackgroundScheduler(timezone="UTC") if SCHED_OK else None
# AI-in-the-loop: engine artik trade yapmaz, sadece oneride bulunur.
# AI (Claude veya baska model) /api/agent/* endpoint'leri ile
# kararlari onaylayinca trade yapilir.
_auto_execute = False  # FAZ 16: Engine auto-trade KAPALI (AI kontrolu)
_scan_count = 0
_reject_log = []  # Son scan'de reddedilen sinyaller (debug icin)

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


def _get_recent_trade_tickers(hours: int = 24) -> set:
    """Son N saat icinde trade yapilmis ticker'lari don (duplicate prevention)."""
    if not DB_OK:
        return set()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = get_recent_trades(limit=100)
        tickers = set()
        for t in recent:
            ts_str = t.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts > cutoff:
                        tickers.add(t["ticker"])
                except Exception:
                    pass
        return tickers
    except Exception as e:
        print(f"[Exit] recent trades hata: {e}")
        return set()


def _check_exits(positions: list, data: dict, current_regime: str) -> list:
    """
    Acik pozisyonlari kontrol et ve exit kriterlerine gore kapat.

    Exit kriterleri:
    1. Stop loss hit
    2. Take profit hit
    3. Holding period dolmus (10 gun)
    4. Crisis rejimi -> tum pozisyonlari kapat
    5. Fallback: -5% / +10%
    """
    if not broker:
        return []

    exits = []

    # Crisis mode: hepsini kapat
    if current_regime == "crisis":
        print("[Exit] CRISIS MODE -- tum pozisyonlar kapatiliyor")
        for p in positions:
            try:
                r = broker.close_position(p["ticker"])
                if r.get("status") == "closed":
                    exits.append({"ticker": p["ticker"], "reason": "crisis_regime",
                                  "pnl": p.get("unrealized_pl", 0)})
                    print(f"[Exit] CLOSED {p['ticker']} (crisis)")
            except Exception as e:
                print(f"[Exit] err {p['ticker']}: {e}")
        return exits

    # Normal exit kontrolleri
    for p in positions:
        ticker = p["ticker"]
        current_price = p.get("current_price", 0)
        side = p.get("side", "long")
        unrealized_pl = p.get("unrealized_pl", 0)
        unrealized_pct = p.get("unrealized_plpc", 0)

        should_exit = False
        reason = ""

        try:
            recent = get_recent_trades(limit=100) if DB_OK else []
            last_trade = next((t for t in recent if t["ticker"] == ticker), None)
            if last_trade:
                sl = last_trade.get("stop_loss", 0)
                tp = last_trade.get("take_profit", 0)
                entry_ts = last_trade.get("timestamp", "")

                if sl > 0 and current_price > 0:
                    if side == "long" and current_price <= sl:
                        should_exit = True
                        reason = f"stop_loss_hit ({current_price:.2f} <= {sl:.2f})"
                    elif side == "short" and current_price >= sl:
                        should_exit = True
                        reason = f"stop_loss_hit ({current_price:.2f} >= {sl:.2f})"

                if not should_exit and tp > 0 and current_price > 0:
                    if side == "long" and current_price >= tp:
                        should_exit = True
                        reason = f"take_profit_hit ({current_price:.2f} >= {tp:.2f})"
                    elif side == "short" and current_price <= tp:
                        should_exit = True
                        reason = f"take_profit_hit ({current_price:.2f} <= {tp:.2f})"

                if not should_exit and entry_ts:
                    try:
                        entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                        days_held = (datetime.now(timezone.utc) - entry_dt).days
                        if days_held >= 10:
                            should_exit = True
                            reason = f"holding_period_expired ({days_held} gun)"
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Exit] db lookup err {ticker}: {e}")

        # Fallback: %5 kayip veya %10 kar
        if not should_exit and unrealized_pct <= -5.0:
            should_exit = True
            reason = f"emergency_stop ({unrealized_pct:.2f}%)"
        elif not should_exit and unrealized_pct >= 10.0:
            should_exit = True
            reason = f"emergency_profit ({unrealized_pct:.2f}%)"

        if should_exit:
            try:
                r = broker.close_position(ticker)
                if r.get("status") == "closed":
                    exits.append({"ticker": ticker, "reason": reason,
                                  "pnl": unrealized_pl, "pnl_pct": unrealized_pct})
                    print(f"[Exit] CLOSED {ticker} -- {reason} (PnL: ${unrealized_pl:.2f} / {unrealized_pct:.2f}%)")
            except Exception as e:
                print(f"[Exit] close err {ticker}: {e}")

    return exits


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

        # ─── 4a. Exit Strategy (acik pozisyonlari kontrol et) ─────
        exits_executed = []
        if EXIT_CHECK_ENABLED and positions and auto_execute and market_open:
            exits_executed = _check_exits(positions, data, current_regime=regime)
            if exits_executed:
                print(f"[Scan] EXITS: {len(exits_executed)} pozisyon kapatildi")
                # Pozisyonlari yeniden yukle
                acc = _get_account_data()
                positions = acc.get("positions", [])

        # ─── 4b. Yeni sinyal filtreleme ve sizing ─────────────
        rejects: list = []  # (ticker, reason)

        if KELLY_OK and _kelly_engine and valid_signals:
            _kelly_engine.update_equity(equity)
            held = {p["ticker"] for p in positions}
            cur_risk = sum(abs(p.get("unrealized_pl", 0)) / equity for p in positions) if equity > 0 else 0

            # Duplicate prevention: son 24 saatteki trade'ler
            recent_tickers = _get_recent_trade_tickers(hours=TICKER_COOLDOWN_HOURS)

            for sig in valid_signals:
                # Filtre 1: ETF'lerde trade yok (benchmark)
                if sig.ticker in ("SPY", "QQQ", "IWM"):
                    rejects.append((sig.ticker, "ETF (benchmark)"))
                    continue
                # Filtre 2: Zaten acik pozisyon var
                if sig.ticker in held:
                    rejects.append((sig.ticker, "zaten acik pozisyon"))
                    continue
                # Filtre 3: Cooldown (son 24 saat icinde trade)
                if sig.ticker in recent_tickers:
                    rejects.append((sig.ticker, f"cooldown ({TICKER_COOLDOWN_HOURS}h)"))
                    continue
                # Filtre 4: Min conviction
                if sig.conviction < MIN_CONVICTION:
                    rejects.append((sig.ticker, f"conviction {sig.conviction:.2f} < {MIN_CONVICTION}"))
                    continue
                # Filtre 5: Veri yeterli mi?
                td = data.get(sig.ticker)
                if td is None or len(td) < 20:
                    rejects.append((sig.ticker, "yetersiz veri"))
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

                    dollar_amount = sz.get("dollar_amount", 0)

                    # Filtre 6: Min position value
                    if dollar_amount < MIN_POSITION_VALUE:
                        rejects.append((sig.ticker, f"pozisyon ${dollar_amount:.0f} < ${MIN_POSITION_VALUE:.0f}"))
                        continue

                    # Filtre 7: qty > 0 olmali
                    if sz.get("qty", 0) <= 0:
                        rejects.append((sig.ticker, "qty=0 (kelly=0 veya risk limiti)"))
                        continue

                    # Filtre 8: Take profit hesapla (2:1 R:R)
                    risk_per_share = abs(price - sl)
                    tp = price + (risk_per_share * 2.0) if sl_dir == "long" else price - (risk_per_share * 2.0)

                    sizing_results.append({"ticker": sig.ticker, "signal": sig.name,
                        "side": sz.get("side", "none"), "qty": sz.get("qty", 0),
                        "price": price, "dollar_amount": dollar_amount,
                        "kelly_adjusted": sz.get("kelly_adjusted", 0), "stop_loss": sl,
                        "take_profit": round(tp, 2),
                        "holding_period": sig.holding_period,
                        "conviction": sig.conviction, "p_value": sig.p_value,
                        "position_risk_pct": sz.get("position_risk_pct", 0)})
                except Exception as e:
                    rejects.append((sig.ticker, f"hesaplama hatasi: {e}"))

            if rejects:
                print(f"[Scan] REJECTS ({len(rejects)}):")
                for ticker, reason in rejects[:10]:
                    print(f"[Scan]   {ticker}: {reason}")

            # ─── 5. Execute (piyasa acikken) ──────────────────
            if auto_execute and market_open:
                trades_today = get_trade_count_today() if DB_OK else 0
                for sz in sizing_results:
                    if trades_today + len(trades_executed) >= MAX_DAILY_TRADES:
                        print(f"[Scan] Gunluk trade limiti ({MAX_DAILY_TRADES}) doldu")
                        break
                    try:
                        res = broker.execute_market(sz["ticker"], sz["qty"], sz["side"])
                        if res.get("status") == "submitted":
                            trades_executed.append(sz)
                            if DB_OK:
                                log_trade(ticker=sz["ticker"], side=sz["side"], qty=sz["qty"],
                                          price=sz["price"], signal_name=sz["signal"],
                                          signal_direction=1.0 if sz["side"] == "buy" else -1.0,
                                          signal_conviction=sz["conviction"], signal_p_value=sz["p_value"],
                                          kelly_f=sz["kelly_adjusted"], regime=regime,
                                          stop_loss=sz["stop_loss"], take_profit=sz.get("take_profit", 0))
                            print(f"[Scan] TRADE: {sz['side'].upper()} {sz['ticker']} x{sz['qty']} "
                                  f"@ ${sz['price']:.2f} (SL ${sz['stop_loss']:.2f}, TP ${sz.get('take_profit',0):.2f})")
                    except Exception as e:
                        print(f"[Scan] Exec err {sz['ticker']}: {e}")

        _reject_log.clear()
        _reject_log.extend(rejects[:20])

        # Dashboard signals
        dash_sigs = [s.to_dict() for s in all_signals[:20]]
        peak = max(_last_scan["risk_status"].get("peak_equity", 0), equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0

        _last_scan = {
            "status": "ok", "timestamp": ts, "market_open": market_open,
            "regime": regime_result, "signals": dash_sigs,
            "valid_signal_count": len(valid_signals),
            "trades_executed": trades_executed,
            "exits_executed": exits_executed,
            "rejects": [{"ticker": t, "reason": r} for t, r in _reject_log[:20]],
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


@app.get("/api/diagnostics")
async def diagnostics():
    """Sistem saglik kontrolu (key prefix'leri yok, sadece bool)."""
    return {
        "env_configured": {
            "alpaca_api_key": bool(os.getenv("ALPACA_API_KEY")),
            "alpaca_secret_key": bool(os.getenv("ALPACA_SECRET_KEY")),
            "alpaca_base_url": bool(os.getenv("ALPACA_BASE_URL")),
        },
        "broker_connected": broker is not None,
        "modules": {
            "db": DB_OK, "pipeline": PIPELINE_OK, "hmm": HMM_OK,
            "signals": SIGNAL_OK, "kelly": KELLY_OK, "scheduler": SCHED_OK,
            "backtest": BACKTEST_OK, "portfolio": PORTFOLIO_OK,
            "kernel": KERNEL_OK, "nlp": NLP_OK,
        },
        "config": {
            "p_threshold": P_VALUE_THRESHOLD,
            "min_conviction": MIN_CONVICTION,
            "min_position_value": MIN_POSITION_VALUE,
            "cooldown_hours": TICKER_COOLDOWN_HOURS,
            "scan_interval_min": SCAN_INTERVAL_MINUTES,
            "max_daily_trades": MAX_DAILY_TRADES,
            "exit_check": EXIT_CHECK_ENABLED,
            "auto_execute": _auto_execute,
        },
        "scheduler_running": _scheduler.running if _scheduler else False,
        "scan_count": _scan_count,
    }


# ─── FP-09 Backtesting API ────────────────────────────
@app.get("/api/backtest/{strategy}")
async def api_backtest(strategy: str = "mean_reversion"):
    """Run backtest for a given strategy against cached data."""
    if not BACKTEST_OK:
        return {"error": "backtest module not loaded"}
    if not _cached_data:
        return {"error": "no data cached -- run /api/scan-now first"}

    strategies = {
        "mean_reversion": mean_reversion_signal,
        "momentum": momentum_signal,
    }
    fn = strategies.get(strategy)
    if not fn:
        return {"error": f"unknown strategy: {strategy}", "available": list(strategies.keys())}

    result = run_backtest(_cached_data, fn)
    return result


@app.get("/api/validate/{strategy}")
async def api_validate(strategy: str = "mean_reversion"):
    """Run full validation (backtest + DSR + verdict)."""
    if not BACKTEST_OK or not _cached_data:
        return {"error": "backtest not available or no cached data"}

    strategies = {
        "mean_reversion": mean_reversion_signal,
        "momentum": momentum_signal,
    }
    fn = strategies.get(strategy)
    if not fn:
        return {"error": f"unknown strategy: {strategy}"}

    return validate_signal(fn, _cached_data)


# ─── FP-07 Portfolio API ──────────────────────────────
@app.get("/api/portfolio/optimize")
async def api_portfolio_optimize(method: str = "market_neutral"):
    """Run portfolio optimization on last scan's signals."""
    if not PORTFOLIO_OK:
        return {"error": "portfolio module not loaded"}

    signals = _last_scan.get("signals", [])
    # Engine'in p_threshold (0.05) ile filtrele, is_valid()'in default 0.01 degil
    valid_signals = [s for s in signals if s.get("p_value", 1.0) < P_VALUE_THRESHOLD
                     and s.get("conviction", 0) >= MIN_CONVICTION]
    if not valid_signals:
        return {"error": "no valid signals -- run scan first", "signals_count": len(signals),
                "threshold": {"p": P_VALUE_THRESHOLD, "conviction": MIN_CONVICTION}}

    result = optimize_portfolio(valid_signals, method=method,
                                  max_weight=0.05, target_gross_leverage=1.0)

    # Factor exposures
    if _cached_data:
        try:
            exposures = compute_factor_exposures(result.get("weights", {}), _cached_data)
            result["factor_exposures"] = exposures
        except Exception as e:
            result["factor_exposures_error"] = str(e)

    return result


# ─── FP-04 Kernel API ─────────────────────────────────
@app.get("/api/kernel/predict/{ticker}")
async def api_kernel_predict(ticker: str, kernel: str = "rbf"):
    """Kernel regression prediction for a ticker."""
    if not KERNEL_OK:
        return {"error": "kernel module not loaded"}
    if not _cached_data or ticker not in _cached_data:
        return {"error": f"ticker {ticker} not in cached data"}
    return kernel_predict(_cached_data[ticker], kernel=kernel)


@app.get("/api/kernel/embeddings")
async def api_kernel_embeddings():
    """Cross-asset KernelPCA embeddings."""
    if not KERNEL_OK:
        return {"error": "kernel module not loaded"}
    if not _cached_data:
        return {"error": "no cached data"}
    return {"embeddings": kernel_pca_features(_cached_data, n_components=3)}


# ─── FP-08 NLP API ────────────────────────────────────
@app.post("/api/nlp/score")
async def api_nlp_score(payload: dict):
    """Score text with financial sentiment dictionary."""
    if not NLP_OK:
        return {"error": "nlp module not loaded"}
    text = payload.get("text", "")
    if not text:
        return {"error": "text required"}
    return score_text(text)


@app.post("/api/nlp/batch")
async def api_nlp_batch(payload: dict):
    """Score multiple texts (headlines, news feed)."""
    if not NLP_OK:
        return {"error": "nlp module not loaded"}
    texts = payload.get("texts", [])
    if not texts:
        return {"error": "texts array required"}
    return score_multiple_texts(texts)


# ═══════════════════════════════════════════════════════
# FAZ 16-20: AI AGENT INTERFACE (Claude AI icin tool)
# ═══════════════════════════════════════════════════════

@app.get("/api/agent/prompt")
async def agent_prompt():
    """Jim Simmons Brain Operator system prompt -- Claude session baslatinca yukle."""
    try:
        from operator_prompt import get_operator_prompt, get_daily_playbook, get_decision_template
        return {
            "system_prompt": get_operator_prompt(),
            "daily_playbook": get_daily_playbook(),
            "decision_template": get_decision_template(),
            "version": VERSION,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/agent/context")
async def agent_context():
    """
    AI icin komple market context ozeti.
    Claude bu endpoint'ten piyasa durumunu alir ve karar verir.
    """
    acc = _get_account_data()
    market_open = _is_market_open()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_status": "open" if market_open else "closed",
        "regime": _last_scan.get("regime", {}),
        "portfolio": {
            "equity": acc.get("equity", 0),
            "cash": acc.get("cash", 0),
            "buying_power": acc.get("buying_power", 0),
            "positions": acc.get("positions", []),
            "n_positions": acc.get("n_positions", 0),
        },
        "signals": _last_scan.get("signals", []),
        "valid_signal_count": _last_scan.get("valid_signal_count", 0),
        "risk_status": _last_scan.get("risk_status", {}),
        "rejects": _last_scan.get("rejects", []),
        "last_scan": _last_scan.get("timestamp"),
        "config": {
            "p_threshold": P_VALUE_THRESHOLD,
            "min_conviction": MIN_CONVICTION,
            "min_position_value": MIN_POSITION_VALUE,
            "max_position_pct": 5.0,
            "max_daily_trades": MAX_DAILY_TRADES,
        },
        "trades_today": get_trade_count_today() if DB_OK else 0,
        "auto_execute": _auto_execute,
        "instruction": (
            "Bu Simons Engine'in current state'idir. Medallion Fund metodlariyla "
            "calisiyor: HMM regime, StatArb signals (p<0.05), Kelly sizing, market "
            "neutral portfolio. Sen karar verici AI'sin. /api/agent/propose-trade ile "
            "trade onerirsin; /api/agent/approve ile onaylarsin. Saf matematik + senin "
            "yargilarin beraber. Risk oncelikli."
        ),
    }


@app.post("/api/agent/propose-trade")
async def agent_propose_trade(payload: dict):
    """
    AI trade oneriyor. Karar pending'e dusser; onay bekler.

    Body: {
        ai_model: "claude-opus-4-6",
        reasoning: "Long NVDA because...",
        ticker: "NVDA",
        side: "buy",
        qty: 10,
        price: 150.0,  # optional (current market)
        conviction: 0.7
    }

    Returns: {decision_id, status: "pending"}
    """
    if not DB_OK:
        return {"error": "db not available"}

    ai_model = payload.get("ai_model", "unknown")
    reasoning = payload.get("reasoning", "")
    ticker = payload.get("ticker", "").upper()
    side = payload.get("side", "buy").lower()
    qty = int(payload.get("qty", 0))
    price = float(payload.get("price", 0))
    conviction = float(payload.get("conviction", 0.5))

    if not ticker or qty <= 0:
        return {"error": "ticker and qty required"}

    # Current context snapshot
    context = {
        "regime": _last_scan.get("regime", {}).get("regime", "unknown"),
        "market_open": _is_market_open(),
        "equity": _get_account_data().get("equity", 0),
    }

    decision_id = log_ai_decision(
        ai_model=ai_model,
        context_snapshot=context,
        reasoning=reasoning,
        decision_type="propose_trade",
        decision_detail=payload,
        ticker=ticker, side=side, qty=qty, price=price, conviction=conviction,
    )

    return {
        "decision_id": decision_id,
        "status": "pending",
        "message": f"Trade proposal logged. Call /api/agent/approve/{decision_id} to execute.",
    }


@app.post("/api/agent/approve/{decision_id}")
async def agent_approve(decision_id: int, payload: dict = None):
    """
    AI (veya human) pending kararı onayliyor. Trade gerceklesir.

    Body (optional): {approved_by: "claude-opus-4-6", override_qty: 10}
    """
    if not DB_OK or not broker:
        return {"error": "db or broker not available"}

    payload = payload or {}
    approved_by = payload.get("approved_by", "system")

    # Pending decision'i al
    decisions = get_all_decisions(limit=100, status="pending")
    decision = next((d for d in decisions if d["id"] == decision_id), None)
    if not decision:
        return {"error": f"decision_id {decision_id} not found or not pending"}

    ticker = decision["ticker"]
    side = decision["side"]
    qty = int(payload.get("override_qty") or decision["qty"])

    if qty <= 0:
        reject_ai_decision(decision_id, approved_by, "qty is zero")
        return {"error": "qty is zero"}

    # Market kontrol
    if not _is_market_open():
        # Piyasa kapali -- trade gonderilir ama open'da execute olur
        pass

    # Execute
    try:
        result = broker.execute_market(ticker, qty, side)
        if result.get("status") == "submitted":
            # Trade'i DB'ye yaz
            trade_id = log_trade(
                ticker=ticker, side=side, qty=qty, price=decision["price"],
                signal_name=f"ai_approved_{approved_by}",
                signal_direction=1.0 if side == "buy" else -1.0,
                signal_conviction=decision["conviction"],
                signal_p_value=0.0,  # AI decision, not signal-based
                kelly_f=0,
                regime=_last_scan.get("regime", {}).get("regime", "unknown"),
                stop_loss=0,
            )
            # Decision'i approve et
            approve_ai_decision(decision_id, approved_by, trade_id)

            return {
                "status": "executed",
                "decision_id": decision_id,
                "trade_id": trade_id,
                "order_id": result.get("order_id", ""),
                "ticker": ticker, "side": side, "qty": qty,
            }
        else:
            reject_ai_decision(decision_id, approved_by,
                               f"broker rejected: {result.get('message', 'unknown')}")
            return {"error": "broker rejected", "detail": result}
    except Exception as e:
        reject_ai_decision(decision_id, approved_by, f"exception: {str(e)}")
        return {"error": str(e)}


@app.post("/api/agent/reject/{decision_id}")
async def agent_reject(decision_id: int, payload: dict = None):
    """AI veya human pending kararı reddediyor."""
    if not DB_OK:
        return {"error": "db not available"}
    payload = payload or {}
    rejected_by = payload.get("rejected_by", "system")
    reason = payload.get("reason", "manual rejection")

    reject_ai_decision(decision_id, rejected_by, reason)
    return {"status": "rejected", "decision_id": decision_id, "reason": reason}


@app.get("/api/agent/pending")
async def agent_pending():
    """Onay bekleyen AI kararlari."""
    if not DB_OK:
        return []
    return get_pending_decisions(limit=50)


@app.get("/api/agent/decisions")
async def agent_decisions(limit: int = 100, status: str = None):
    """Tum AI kararlari (history)."""
    if not DB_OK:
        return []
    return get_all_decisions(limit=limit, status=status)


@app.post("/api/agent/record-outcome/{decision_id}")
async def agent_record_outcome(decision_id: int, payload: dict):
    """
    AI kararinin outcome'unu kaydet (pozisyon kapandiktan sonra).

    Body: {pnl: 250.50, label: "win"}  # label: win|loss|flat
    """
    if not DB_OK:
        return {"error": "db not available"}
    pnl = float(payload.get("pnl", 0))
    label = payload.get("label", "flat")
    record_ai_outcome(decision_id, pnl, label)
    return {"status": "recorded", "decision_id": decision_id, "pnl": pnl, "label": label}


@app.post("/api/agent/log-analysis")
async def agent_log_analysis(payload: dict):
    """
    AI analizi logla (training data icin).

    Body: {ai_model, analysis_type, ticker, input_context, output_reasoning, output_verdict, tokens_used}
    """
    if not DB_OK:
        return {"error": "db not available"}

    analysis_id = log_ai_analysis(
        ai_model=payload.get("ai_model", "unknown"),
        analysis_type=payload.get("analysis_type", "generic"),
        ticker=payload.get("ticker", ""),
        input_context=payload.get("input_context", {}),
        output_reasoning=payload.get("output_reasoning", ""),
        output_verdict=payload.get("output_verdict", ""),
        tokens_used=payload.get("tokens_used", 0),
    )
    return {"analysis_id": analysis_id, "status": "logged"}


@app.get("/api/training/export")
async def training_export(limit: int = 1000):
    """
    Training data export (JSONL-ready).
    Sadece outcome'u kaydedilmis kararlar.
    """
    if not DB_OK:
        return {"error": "db not available"}
    data = export_training_data(limit=limit)
    return {
        "n_samples": len(data),
        "format": "jsonl-ready",
        "samples": data,
    }


@app.get("/api/agent/stats")
async def agent_stats():
    """AI performans istatistikleri."""
    if not DB_OK:
        return {"error": "db not available"}

    all_decisions = get_all_decisions(limit=10000)
    pending = sum(1 for d in all_decisions if d["status"] == "pending")
    approved = sum(1 for d in all_decisions if d["status"] == "approved")
    rejected = sum(1 for d in all_decisions if d["status"] == "rejected")

    # Outcome stats (feedback)
    with_outcome = [d for d in all_decisions if d.get("outcome_pnl") is not None]
    wins = sum(1 for d in with_outcome if (d.get("outcome_pnl") or 0) > 0)
    losses = sum(1 for d in with_outcome if (d.get("outcome_pnl") or 0) < 0)
    total_pnl = sum(d.get("outcome_pnl", 0) or 0 for d in with_outcome)

    return {
        "total_decisions": len(all_decisions),
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "with_outcome": len(with_outcome),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(with_outcome) * 100, 2) if with_outcome else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(with_outcome), 2) if with_outcome else 0,
    }
