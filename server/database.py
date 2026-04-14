"""
database.py — Simons Engine SQLite Database

Trade logları, sinyal geçmişi, rejim geçmişi, performans metrikleri.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "simons.db"


def init_db():
    """Veritabanı tablolarını oluştur."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                price REAL NOT NULL,
                dollar_amount REAL,
                signal_name TEXT,
                signal_direction REAL,
                signal_conviction REAL,
                signal_p_value REAL,
                kelly_f REAL,
                regime TEXT,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS signals_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_name TEXT,
                direction REAL,
                conviction REAL,
                p_value REAL,
                alpha_estimate REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                n_states INTEGER,
                state_probabilities TEXT,
                regime_change_prob REAL,
                transition_matrix TEXT
            );

            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT,
                n_signals INTEGER,
                n_trades INTEGER,
                portfolio_value REAL,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                equity REAL,
                daily_pnl REAL,
                daily_return REAL,
                n_trades INTEGER,
                regime TEXT,
                sharpe_running REAL,
                max_drawdown REAL
            );
        """)
    print("[DB] Simons Engine database initialized")


def log_trade(ticker: str, side: str, qty: int, price: float,
              signal_name: str = "", signal_direction: float = 0,
              signal_conviction: float = 0, signal_p_value: float = 1,
              kelly_f: float = 0, regime: str = "unknown",
              stop_loss: float = 0, take_profit: float = 0) -> int:
    """Trade logla, ID döndür."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (timestamp, ticker, side, qty, price, dollar_amount,
                signal_name, signal_direction, signal_conviction, signal_p_value,
                kelly_f, regime, stop_loss, take_profit)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                ticker, side, qty, price, round(qty * price, 2),
                signal_name, signal_direction, signal_conviction, signal_p_value,
                kelly_f, regime, stop_loss, take_profit,
            )
        )
        conn.commit()
        return cur.lastrowid


def log_signal(ticker: str, signal_name: str, direction: float,
               conviction: float, p_value: float, alpha_estimate: float,
               metadata: dict = None):
    """Sinyal geçmişini logla."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO signals_history
               (timestamp, ticker, signal_name, direction, conviction, p_value, alpha_estimate, metadata)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                ticker, signal_name, direction, conviction, p_value, alpha_estimate,
                json.dumps(metadata or {}),
            )
        )
        conn.commit()


def log_regime(regime: str, n_states: int, state_probs: dict,
               change_prob: float, trans_matrix: list):
    """Rejim geçmişini logla."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO regime_history
               (timestamp, regime, n_states, state_probabilities, regime_change_prob, transition_matrix)
               VALUES (?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                regime, n_states, json.dumps(state_probs),
                change_prob, json.dumps(trans_matrix),
            )
        )
        conn.commit()


def log_scan(regime: str, n_signals: int, n_trades: int,
             portfolio_value: float, summary: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO scans (timestamp, regime, n_signals, n_trades, portfolio_value, summary) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), regime, n_signals, n_trades, portfolio_value, summary)
        )
        conn.commit()


def log_daily_performance(equity: float, daily_pnl: float, daily_return: float,
                          n_trades: int, regime: str, sharpe: float, max_dd: float):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO daily_performance
               (date, equity, daily_pnl, daily_return, n_trades, regime, sharpe_running, max_drawdown)
               VALUES (?,?,?,?,?,?,?,?)""",
            (today, equity, daily_pnl, daily_return, n_trades, regime, sharpe, max_dd)
        )
        conn.commit()


def get_recent_trades(limit: int = 50) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade_count_today() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()
        return row[0] if row else 0


def get_daily_performance(days: int = 30) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM daily_performance ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_regime_history(limit: int = 100) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM regime_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
