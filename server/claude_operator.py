"""
claude_operator.py -- Anthropic Claude AI Operator Integration

Claude AI her scan sonrasi Simons Engine'in context'ini analiz eder,
pending decisions'i onaylar/reddeder, yeni trade proposals uretir.

Model: claude-sonnet-4-5 (balance: kalite + maliyet)
Cost budget: $5/gun, $100/ay

Fail-safe: Claude API down ise hicbir trade YAPILMAZ (safe default).
Tum hard limitler (p<0.05, max %5 pozisyon, -15% circuit breaker)
Claude override edemez -- engine enforces.
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Model ve config
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "3000"))
CLAUDE_DAILY_BUDGET_USD = float(os.getenv("CLAUDE_DAILY_BUDGET_USD", "5.0"))

# Sonnet 4.5 pricing (per 1M tokens)
PRICING = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}


class ClaudeOperator:
    """
    Jim Simmons Brain Operator -- Anthropic Claude wrapper.
    """

    def __init__(self):
        self.available = False
        self.client = None
        self.model = CLAUDE_MODEL
        self.max_tokens = CLAUDE_MAX_TOKENS

        # Daily token/cost tracking
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.daily_cost_usd = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.call_count_today = 0
        self.last_call_at = None
        self.last_response = None
        self.last_error = None

        if not ANTHROPIC_AVAILABLE:
            self.last_error = "anthropic SDK not installed"
            return

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.last_error = "ANTHROPIC_API_KEY not set"
            return

        try:
            self.client = Anthropic(api_key=api_key)
            self.available = True
            print(f"[Claude] Operator ready -- model: {self.model}, budget: ${CLAUDE_DAILY_BUDGET_USD}/day")
        except Exception as e:
            self.last_error = f"init failed: {e}"
            print(f"[Claude] Init error: {e}")

    def _reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            self.daily_input_tokens = 0
            self.daily_output_tokens = 0
            self.daily_cost_usd = 0.0
            self.call_count_today = 0
            self.last_reset_date = today
            print(f"[Claude] Daily counters reset for {today}")

    def _budget_ok(self) -> bool:
        self._reset_daily_if_needed()
        return self.daily_cost_usd < CLAUDE_DAILY_BUDGET_USD

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        prices = PRICING.get(self.model, PRICING["claude-sonnet-4-5"])
        cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
        return cost

    def decide(self, system_prompt: str, context: dict, pending: list = None) -> dict:
        """
        Claude'u cagir, karar iste.

        Args:
            system_prompt: Jim Simmons Brain Operator prompt
            context: /api/agent/context output
            pending: Bekleyen AI kararlari listesi

        Returns:
            {
                "success": bool,
                "reasoning": str,
                "actions": [  # Claude'un kararlari
                    {"type": "propose_trade", ...},
                    {"type": "approve", "decision_id": X},
                    {"type": "reject", "decision_id": X, "reason": ...},
                    {"type": "hold", "reason": ...},
                ],
                "input_tokens": int,
                "output_tokens": int,
                "cost_usd": float,
                "error": str | None,
            }
        """
        if not self.available:
            return {"success": False, "error": self.last_error or "not available", "actions": []}

        if not self._budget_ok():
            return {
                "success": False,
                "error": f"daily budget ${CLAUDE_DAILY_BUDGET_USD} exceeded (${self.daily_cost_usd:.2f})",
                "actions": [],
            }

        pending = pending or []

        # User message: context + pending + task
        user_msg = f"""## CURRENT MARKET CONTEXT

{json.dumps(context, indent=2, default=str)[:6000]}

## PENDING DECISIONS (awaiting your approval)

{json.dumps(pending, indent=2, default=str)[:2000] if pending else "(no pending)"}

## YOUR TASK

1. Market durumunu ve sinyalleri analiz et
2. Her pending decision icin: APPROVE, REJECT, or HOLD kararla
3. Eger yeni yuksek-kalite firsat varsa, NEW trade proposal yaz

## OUTPUT FORMAT (JSON)

Yanitini **sadece gecerli JSON** olarak ver. Baska text yazma.

```json
{{
  "reasoning": "Brief market analysis + reasoning (max 300 words)",
  "actions": [
    {{"type": "approve", "decision_id": 1, "reason": "Strong signal, good R:R"}},
    {{"type": "reject", "decision_id": 2, "reason": "Conviction too low"}},
    {{"type": "propose_trade", "ticker": "NVDA", "side": "buy", "qty": 20, "price": 150.0, "conviction": 0.65, "reasoning": "..."}},
    {{"type": "hold", "reason": "Market waiting for Fed decision"}}
  ]
}}
```

Sadece JSON. Disiplin kurallarina uy. Riskli pozisyon onerme.
"""

        try:
            self.call_count_today += 1
            self.last_call_at = datetime.now(timezone.utc).isoformat()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )

            # Token usage
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            cost = self._compute_cost(in_tok, out_tok)

            self.daily_input_tokens += in_tok
            self.daily_output_tokens += out_tok
            self.daily_cost_usd += cost

            # Parse response
            text = response.content[0].text if response.content else ""
            self.last_response = text[:500]

            # Extract JSON
            parsed = self._parse_response(text)

            return {
                "success": True,
                "reasoning": parsed.get("reasoning", ""),
                "actions": parsed.get("actions", []),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 4),
                "daily_cost_usd": round(self.daily_cost_usd, 4),
                "raw_text": text[:500],
                "model": self.model,
            }

        except Exception as e:
            self.last_error = str(e)
            print(f"[Claude] API error: {e}")
            return {"success": False, "error": str(e), "actions": []}

    def _parse_response(self, text: str) -> dict:
        """Claude'un response'undan JSON cikart."""
        if not text:
            return {"reasoning": "", "actions": []}

        # Direct JSON attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Markdown JSON block
        if "```json" in text:
            try:
                start = text.index("```json") + len("```json")
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except Exception:
                pass

        if "```" in text:
            try:
                start = text.index("```") + 3
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except Exception:
                pass

        # Find first { and last }
        try:
            first = text.index("{")
            last = text.rindex("}")
            return json.loads(text[first:last+1])
        except Exception:
            pass

        return {"reasoning": text[:500], "actions": [], "parse_error": True}

    def get_status(self) -> dict:
        self._reset_daily_if_needed()
        return {
            "available": self.available,
            "model": self.model,
            "calls_today": self.call_count_today,
            "tokens_today_input": self.daily_input_tokens,
            "tokens_today_output": self.daily_output_tokens,
            "cost_today_usd": round(self.daily_cost_usd, 4),
            "daily_budget_usd": CLAUDE_DAILY_BUDGET_USD,
            "budget_remaining_usd": round(CLAUDE_DAILY_BUDGET_USD - self.daily_cost_usd, 4),
            "last_call_at": self.last_call_at,
            "last_error": self.last_error,
        }


# Singleton
_operator: Optional[ClaudeOperator] = None


def get_operator() -> ClaudeOperator:
    global _operator
    if _operator is None:
        _operator = ClaudeOperator()
    return _operator


def is_claude_available() -> bool:
    return get_operator().available
