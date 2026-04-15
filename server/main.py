"""
main.py — Simons Engine Server

Jim Simons / Medallion Fund inspired quantitative trading engine.
Tamamen matematik tabanlı — AI API kullanmaz.

HMM + StatArb + Kelly Criterion + Autonomous Execution

Dashboard: http://localhost:8000
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from broker import SimonsBroker
from database import init_db, get_recent_trades, get_daily_performance, get_regime_history
import scheduler as sched

# ─── Env ──────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# ─── Broker (set in lifespan, not at import time) ─────
broker = None


def _init_broker():
    """Broker başlat — thread executor'da çalışır, event loop'u bloke etmez."""
    b = SimonsBroker()
    acc = b.get_account()
    return b, acc


# ─── Lifespan ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global broker
    init_db()

    # Broker'ı async olarak başlat (10s timeout — event loop'u bloke etmez)
    try:
        loop = asyncio.get_running_loop()
        b, acc = await asyncio.wait_for(
            loop.run_in_executor(None, _init_broker),
            timeout=10.0,
        )
        broker = b
        print(f"[Simons] Broker bağlandı — Equity: ${acc['equity']:,.2f}")
    except asyncio.TimeoutError:
        print("[Simons] Broker bağlantısı timeout (10s) — demo mod aktif")
    except Exception as e:
        print(f"[Simons] Broker başlatılamadı (demo mod): {e}")

    sched.start(broker=broker, auto_execute=True, interval_minutes=10)
    yield
    sched.stop()


app = FastAPI(title="Simons Engine", version="1.0", lifespan=lifespan)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Dashboard ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = static_dir / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<h1>Simons Engine — Dashboard yükleniyor...</h1>"


# ─── API Endpoints ────────────────────────────────────
@app.get("/api/health")
async def health():
    last_scan = sched.get_last_scan()
    return {
        "status": "ok",
        "version": "1.0",
        "engine": "Simons Quantitative",
        "ai_api_used": False,
        "regime": last_scan.get("regime", {}).get("regime", "unknown"),
        "market_open": last_scan.get("market_open", False),
        "last_scan": last_scan.get("timestamp"),
    }


@app.get("/api/account")
async def account():
    if broker is None:
        return {"error": "Broker bağlantısı yok"}
    acc = broker.get_account()
    positions = broker.get_positions()
    return {**acc, "positions": positions, "n_positions": len(positions)}


@app.get("/api/positions")
async def positions():
    if broker is None:
        return []
    return broker.get_positions()


@app.get("/api/scan")
async def get_scan():
    return sched.get_last_scan()


@app.post("/api/scan-now")
async def scan_now():
    sched.run_scan(broker=broker, auto_execute=True)
    return sched.get_last_scan()


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
    from kelly_engine import KellyEngine
    ke = KellyEngine()
    if broker:
        acc = broker.get_account()
        ke.update_equity(acc.get("equity", 100_000))
    return ke.get_risk_status()
