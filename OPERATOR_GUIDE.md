# Jim Simmons Brain Operator — Quick Reference

Bu belge Claude AI Operator'un Simons Engine'i nasıl kullanacağını anlatır. **Sen operatör, engine araç.**

## Bağlantı

- **Base URL:** https://web-production-f9354.up.railway.app
- **System Prompt:** `GET /api/agent/prompt` (her session başlangıcında yükle)
- **Broker:** Jim Simmons Brain paper account ($100K başlangıç)
- **Piyasa:** NYSE 13:30-20:00 UTC (Mon-Fri)

## Günlük Akış

### 1. Session Başlatma

```bash
# System prompt al
curl https://web-production-f9354.up.railway.app/api/agent/prompt
```

### 2. Piyasa Durumu

```bash
# Komple context (rejim, sinyaller, portfolio, risk)
curl https://web-production-f9354.up.railway.app/api/agent/context
```

### 3. Analiz

```bash
# Belirli strateji backtest et
curl https://web-production-f9354.up.railway.app/api/backtest/mean_reversion

# Validate (Deflated Sharpe verdict)
curl https://web-production-f9354.up.railway.app/api/validate/mean_reversion

# Portfolio optimize (market neutral)
curl https://web-production-f9354.up.railway.app/api/portfolio/optimize?method=market_neutral

# Kernel prediction (non-linear)
curl https://web-production-f9354.up.railway.app/api/kernel/predict/NVDA

# News sentiment
curl -X POST https://web-production-f9354.up.railway.app/api/nlp/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Company beat earnings..."}'
```

### 4. Karar Verme

**Önerme:**
```bash
curl -X POST https://web-production-f9354.up.railway.app/api/agent/propose-trade \
  -H "Content-Type: application/json" \
  -d '{
    "ai_model": "claude-opus-4-6",
    "reasoning": "JNJ: p=0.014, conviction=0.41, z-score -2.1. Kelly 2.5%. SL 142, TP 156. R:R 2:1. APPROVE.",
    "ticker": "JNJ",
    "side": "buy",
    "qty": 17,
    "price": 147.50,
    "conviction": 0.41
  }'
# Response: {"decision_id": 1, "status": "pending"}
```

**Onaylama:**
```bash
curl -X POST https://web-production-f9354.up.railway.app/api/agent/approve/1 \
  -H "Content-Type: application/json" \
  -d '{"approved_by": "claude-opus-4-6"}'
# Response: {"status": "executed", "order_id": "..."}
```

**Reddetme:**
```bash
curl -X POST https://web-production-f9354.up.railway.app/api/agent/reject/1 \
  -H "Content-Type: application/json" \
  -d '{"rejected_by": "claude-opus-4-6", "reason": "Conviction too low after Monte Carlo"}'
```

### 5. Feedback (pozisyon kapandığında)

```bash
curl -X POST https://web-production-f9354.up.railway.app/api/agent/record-outcome/1 \
  -H "Content-Type: application/json" \
  -d '{"pnl": 250.50, "label": "win"}'
```

### 6. İstatistik

```bash
# AI performance stats
curl https://web-production-f9354.up.railway.app/api/agent/stats

# Training data export
curl https://web-production-f9354.up.railway.app/api/training/export?limit=1000
```

## Disiplin Kuralları (DEĞİŞTİRİLEMEZ)

1. **p-value:** Sadece p < 0.05 olan sinyalleri düşün
2. **Conviction:** >= 0.3 altındakileri reddet
3. **Position size:** $500 altı pozisyon yok
4. **Daily limit:** Max 3 yeni trade/gün (PDT)
5. **Holding:** Max 10 gün pozisyon (otomatik exit)
6. **Drawdown:** -5% azalt, -10% yarım, -15% tümünü kapat
7. **Cooldown:** Aynı tickera 24h içinde 2. trade YOK
8. **ETF:** SPY, QQQ, IWM — benchmark, trade yok
9. **Single position:** Max %5 equity
10. **Crisis regime:** Yeni long yok, mevcut pozisyonları kapat

## Büyüme Hedefi

- Yıl 1: +%20 net (Sharpe > 1.5)
- Yıl 5: 3x equity
- Medallion değil, iyi kantitatif hedge fund ortalaması

## Reasoning Template

Her proposal için **100+ kelime** gerekçe:

```
[TICKER]: [signal_name] signal p=[p_value], conviction=[c].
HMM rejim [regime] (risk mult [r]x).
[Technical: z-score, half-life, momentum, volume, etc.]
Kelly sizing [pct]% of equity = $[amount].
Stop-loss [sl] (ATR*2), Take-profit [tp] (2R).
Risk/reward [ratio]:1.
Portfolio beta etkisi [beta] -- [neutral/long-biased/defensive].
[Backtest/validation notes if done].
[APPROVE/REJECT/BORDERLINE with reason].
```

## Dashboard

https://web-production-f9354.up.railway.app — AI Decisions sekmesinde tüm kararların görünür.

## Ne Zaman Uyan

- **13:30 UTC** (piyasa açılır) — ilk context, trade penceresi
- **15:00-18:00** — main execution
- **19:30 UTC** — exit kontrolleri
- **20:00 UTC** — piyasa kapanır, outcomes kaydet
- **Weekend** — backtest + training data export
