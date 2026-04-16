"""
broker.py — Simons Engine Alpaca Broker

Ayrı paper trading hesabı ile çalışır.
Claude Brain'den tamamen bağımsız.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame

# .env yolunu guvenlice bul (hem server/ icinden hem disaridan calisir)
_this_dir = Path(__file__).resolve().parent  # server/
_env_path = _this_dir.parent / ".env"
if not _env_path.exists():
    _env_path = _this_dir / ".env"  # fallback
load_dotenv(_env_path, override=True)
_env_vals = dotenv_values(_env_path) if _env_path.exists() else {}

def _get(key): return os.getenv(key) or _env_vals.get(key, "")


class SimonsBroker:
    """Jim Simons paper trading broker."""

    def __init__(self):
        api_key = _get("ALPACA_API_KEY")
        secret_key = _get("ALPACA_SECRET_KEY")
        base_url = _get("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets/v2"

        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY ve ALPACA_SECRET_KEY gerekli")

        self.paper = "paper" in base_url
        self.client = TradingClient(api_key, secret_key, paper=self.paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self) -> dict:
        acc = self.client.get_account()
        return {
            "cash": float(acc.cash),
            "equity": float(acc.equity),
            "portfolio_value": float(acc.portfolio_value),
            "buying_power": float(acc.buying_power),
            "day_trade_count": acc.daytrade_count,
        }

    def get_positions(self) -> list:
        positions = self.client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2),
            }
            for p in positions
        ]

    def execute_market(self, ticker: str, qty: int, side: str = "buy") -> dict:
        """Market order ile işlem yap."""
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            return {
                "status": "submitted",
                "order_id": str(order.id),
                "ticker": ticker,
                "qty": qty,
                "side": side,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_limit(self, ticker: str, qty: int, price: float, side: str = "buy") -> dict:
        """Limit order ile işlem yap."""
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            req = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(price, 2),
            )
            order = self.client.submit_order(req)
            return {
                "status": "submitted",
                "order_id": str(order.id),
                "ticker": ticker,
                "qty": qty,
                "side": side,
                "limit_price": price,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close_position(self, ticker: str) -> dict:
        """Pozisyonu kapat."""
        try:
            self.client.close_position(ticker)
            return {"status": "closed", "ticker": ticker}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self) -> dict:
        try:
            self.client.cancel_orders()
            return {"status": "ok", "message": "Tum emirler iptal edildi"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_bars(self, ticker: str, timeframe=TimeFrame.Day, limit: int = 200):
        """Tarihsel bar verisi al."""
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=limit * 2)  # Hafta sonları için buffer

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start,
            limit=limit,
        )
        bars = self.data_client.get_stock_bars(req)
        # alpaca-py v0.30+: BarSet obje, .data dict veya dogrudan dict access
        if hasattr(bars, 'data') and isinstance(bars.data, dict):
            return bars.data.get(ticker, [])
        elif hasattr(bars, '__getitem__'):
            try:
                return bars[ticker]
            except (KeyError, TypeError):
                return []
        return []

    def get_portfolio_history(self, period: str = "1D", timeframe: str = None) -> dict:
        """
        Alpaca portfolio history -- equity egrisi cizebilmek icin.

        period: "1D", "1W", "1M", "3M", "1A", "all"
        timeframe: "1Min", "5Min", "15Min", "1H", "1D" (auto based on period if None)

        Returns:
            {
                timestamp: [unix_ts_1, unix_ts_2, ...],
                equity: [val_1, val_2, ...],
                profit_loss: [...],
                profit_loss_pct: [...],
                base_value: float,
                timeframe: str,
            }
        """
        try:
            # Auto-pick timeframe based on period
            if timeframe is None:
                tf_map = {"1D": "5Min", "1W": "15Min", "1M": "1H",
                          "3M": "1D", "1A": "1D", "all": "1D"}
                timeframe = tf_map.get(period, "1H")

            # Alpaca TradingClient uses direct HTTP call for this endpoint
            # Fallback: raw HTTP via requests
            import urllib.parse
            import urllib.request
            import json as _json

            api_key = _get("ALPACA_API_KEY")
            secret = _get("ALPACA_SECRET_KEY")
            base = _get("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"
            if not base.endswith("/v2"):
                base = base.rstrip("/") + "/v2"

            params = urllib.parse.urlencode({
                "period": period,
                "timeframe": timeframe,
                "extended_hours": "true",
            })
            url = f"{base}/account/portfolio/history?{params}"

            req = urllib.request.Request(
                url,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": secret,
                },
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode("utf-8"))

            return {
                "timestamp": data.get("timestamp", []),
                "equity": data.get("equity", []),
                "profit_loss": data.get("profit_loss", []),
                "profit_loss_pct": data.get("profit_loss_pct", []),
                "base_value": data.get("base_value", 0),
                "timeframe": data.get("timeframe", timeframe),
                "period": period,
            }
        except Exception as e:
            return {"error": str(e), "period": period}


    def get_snapshot(self, ticker: str) -> dict:
        """Anlık fiyat verisi."""
        try:
            req = StockSnapshotRequest(symbol_or_symbols=ticker)
            snap_raw = self.data_client.get_stock_snapshot(req)
            snap = snap_raw.data if hasattr(snap_raw, 'data') and isinstance(snap_raw.data, dict) else snap_raw
            if ticker in snap:
                s = snap[ticker]
                return {
                    "price": float(s.latest_trade.price) if s.latest_trade else 0,
                    "volume": int(s.latest_trade.size) if s.latest_trade else 0,
                    "bid": float(s.latest_quote.bid_price) if s.latest_quote else 0,
                    "ask": float(s.latest_quote.ask_price) if s.latest_quote else 0,
                }
            return {}
        except Exception as e:
            return {"error": str(e)}
