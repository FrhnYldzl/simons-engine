"""
nlp_engine.py -- FP-08: NLP & Speech Recognition Signals

Brown & Mercer'in IBM konusma tanima biriminden getirdigi NLP yaklasimi.
Metin verisinden sinyal uretme.

Patterson: "Finansal piyasalara uyguladigimiz matematik, konusma tanimaya cok benzerdi."

Bu module basitlestirilmis: FinBERT transformer yerine Loughran-McDonald
finansal sozluk bazli sentiment (hizli, bagimsiz, 0 model indirmesi).

Teknikler:
  1. Loughran-McDonald finansal sozluk sentiment
  2. News headline scoring
  3. SEC filing text diff (change detection)

Refs:
  - Loughran & McDonald (2011). When Is a Liability Not a Liability? J. Finance.
  - Brown et al. (1990). Statistical Machine Translation.
  - Zuckerman (2019), Ch. 11.
"""

import re
import numpy as np
from datetime import datetime, timezone

# ─── Loughran-McDonald Financial Sentiment Dictionary (subset) ─────
# Tam sozluk ~86,000 kelime. Biz en yuksek etkili olanlari aliyoruz.
POSITIVE_WORDS = {
    "gain", "gains", "gained", "profit", "profits", "profitable", "profitability",
    "strong", "stronger", "strongest", "robust", "improved", "improvement", "improves",
    "exceeded", "beat", "beats", "outperform", "outperformed", "record", "growth",
    "expanded", "expansion", "accelerated", "surge", "surged", "rally", "rallied",
    "innovation", "innovative", "breakthrough", "winning", "momentum", "leadership",
    "upgraded", "upgrade", "upside", "bullish", "optimistic", "confident", "confidence",
    "achievement", "succeed", "successful", "success", "pleased", "benefit", "benefits",
    "positive", "favorable", "excellent", "exceptional", "outstanding", "solid",
}

NEGATIVE_WORDS = {
    "loss", "losses", "loss", "declining", "decline", "declined", "drop", "dropped",
    "weak", "weaker", "weakness", "poor", "poorly", "deteriorated", "deterioration",
    "missed", "miss", "misses", "underperform", "shortfall", "disappointing", "disappointed",
    "contracted", "contraction", "plunge", "plunged", "crash", "crashed", "slump",
    "risk", "risks", "risky", "uncertain", "uncertainty", "volatile", "volatility",
    "downgrade", "downgraded", "downside", "bearish", "pessimistic", "cautious",
    "failure", "failed", "concern", "concerns", "concerning", "worried", "worry",
    "negative", "adverse", "unfavorable", "challenging", "difficulty", "difficulties",
    "lawsuit", "litigation", "investigation", "scandal", "fraud", "bankruptcy",
    "restructuring", "layoffs", "dismissed", "recall", "defect", "warning",
}

UNCERTAINTY_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "uncertain", "unclear",
    "depend", "depends", "dependent", "fluctuation", "fluctuate", "variability",
    "approximately", "roughly", "estimate", "estimated", "estimates",
}

LITIGATION_WORDS = {
    "lawsuit", "litigation", "settlement", "claim", "claims", "allege", "alleged",
    "court", "judge", "plaintiff", "defendant", "SEC", "DOJ", "FTC", "regulatory",
    "investigation", "probe", "violation", "penalty", "fine", "fined",
}


def _tokenize(text: str) -> list:
    """Basit tokenization: lowercase, alphanumeric only."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z]+", text.lower())


def score_text(text: str) -> dict:
    """
    Tek bir metni (news headline, earnings call, filing vb.) scorelar.

    Returns:
        {
            "sentiment": float in [-1, 1],
            "positive_count": int,
            "negative_count": int,
            "uncertainty_count": int,
            "litigation_count": int,
            "total_words": int,
            "net_sentiment_pct": float (0-100),
        }
    """
    if not text:
        return {"sentiment": 0, "positive_count": 0, "negative_count": 0,
                "uncertainty_count": 0, "litigation_count": 0, "total_words": 0,
                "net_sentiment_pct": 0}

    tokens = _tokenize(text)
    if not tokens:
        return {"sentiment": 0, "error": "no_tokens"}

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    unc = sum(1 for t in tokens if t in UNCERTAINTY_WORDS)
    lit = sum(1 for t in tokens if t in LITIGATION_WORDS)
    total = len(tokens)

    # Sentiment = (pos - neg) / (pos + neg + 1)  -- Loughran normalization
    denom = pos + neg + 1
    sentiment = (pos - neg) / denom

    # Uncertainty penalty: yuksek uncertainty sinyali zayiflatir
    uncertainty_penalty = unc / total if total > 0 else 0
    sentiment_adj = sentiment * (1 - min(uncertainty_penalty * 3, 0.5))

    return {
        "sentiment": round(float(sentiment_adj), 4),
        "positive_count": pos,
        "negative_count": neg,
        "uncertainty_count": unc,
        "litigation_count": lit,
        "total_words": total,
        "net_sentiment_pct": round(float(sentiment_adj * 100), 2),
        "has_litigation": lit >= 3,  # 3+ litigation word = red flag
    }


def score_multiple_texts(texts: list) -> dict:
    """
    Birden fazla metin (news feed, multiple headlines) icin aggregate score.
    """
    if not texts:
        return {"aggregate_sentiment": 0, "n_texts": 0}

    scores = [score_text(t) for t in texts if t]
    scores = [s for s in scores if s.get("total_words", 0) > 0]

    if not scores:
        return {"aggregate_sentiment": 0, "n_texts": 0}

    sentiments = [s["sentiment"] for s in scores]
    litigations = sum(1 for s in scores if s.get("has_litigation"))

    return {
        "aggregate_sentiment": round(float(np.mean(sentiments)), 4),
        "sentiment_std": round(float(np.std(sentiments)), 4),
        "sentiment_median": round(float(np.median(sentiments)), 4),
        "n_texts": len(scores),
        "n_positive": sum(1 for s in sentiments if s > 0.1),
        "n_negative": sum(1 for s in sentiments if s < -0.1),
        "n_neutral": sum(1 for s in sentiments if abs(s) <= 0.1),
        "n_with_litigation": litigations,
        "signal": round(float(np.clip(np.mean(sentiments) * 2, -1, 1)), 4),
    }


def text_diff_score(old_text: str, new_text: str) -> dict:
    """
    Iki metin arasindaki anlamli degisiklikleri tespit et.

    Ornek: 10-K'nin onceki yil vs bu yil "Risk Factors" bolumu.
    Degisen cumleler -> sinyal.
    """
    if not old_text or not new_text:
        return {"change_score": 0, "new_sentiment": 0, "removed_sentiment": 0}

    old_tokens = set(_tokenize(old_text))
    new_tokens = set(_tokenize(new_text))

    added = new_tokens - old_tokens
    removed = old_tokens - new_tokens

    # Added words'lerin sentiment'i
    added_pos = sum(1 for w in added if w in POSITIVE_WORDS)
    added_neg = sum(1 for w in added if w in NEGATIVE_WORDS)
    added_lit = sum(1 for w in added if w in LITIGATION_WORDS)

    removed_pos = sum(1 for w in removed if w in POSITIVE_WORDS)
    removed_neg = sum(1 for w in removed if w in NEGATIVE_WORDS)

    # Change score
    # Yeni negatif kelime = kotu haber, yeni pozitif = iyi
    change_score = (added_pos - added_neg) / max(len(added), 1)

    # Red flag: yeni litigation kelimeleri eklenmis
    red_flag = added_lit >= 2

    return {
        "change_score": round(float(change_score), 4),
        "n_added_words": len(added),
        "n_removed_words": len(removed),
        "added_positive": added_pos,
        "added_negative": added_neg,
        "added_litigation": added_lit,
        "removed_positive": removed_pos,
        "removed_negative": removed_neg,
        "red_flag": red_flag,
        "signal": round(float(np.clip(change_score * 3, -1, 1)), 4),
    }


def ticker_news_sentiment(ticker: str, headlines: list = None) -> dict:
    """
    Bir ticker icin haber sentiment skoru.
    headlines parametresi None ise mock/sample haberler kullan.

    Uretimde: news API (finnhub, polygon.io) ile entegre edilir.
    """
    if headlines is None:
        # Mock/sample for testing
        headlines = []

    if not headlines:
        return {
            "ticker": ticker,
            "n_headlines": 0,
            "signal": 0,
            "status": "no_data",
        }

    result = score_multiple_texts(headlines)
    return {
        "ticker": ticker,
        "n_headlines": result["n_texts"],
        "sentiment": result["aggregate_sentiment"],
        "signal": result["signal"],
        "n_positive": result.get("n_positive", 0),
        "n_negative": result.get("n_negative", 0),
        "has_red_flags": result.get("n_with_litigation", 0) > 0,
        "status": "ok",
    }


# ─── Self test examples ──────────────────────────────
def _test_examples():
    examples = {
        "earnings_beat": "Company reported strong quarterly earnings, beat analyst estimates, with robust revenue growth and expanded margins.",
        "earnings_miss": "Company missed estimates, weak performance, declining sales, warned of further challenges ahead.",
        "neutral": "The company announced its quarterly results today. Revenue was $10 billion.",
        "litigation": "SEC announced investigation into company, multiple lawsuits filed alleging fraud. Court proceedings continue.",
    }
    for name, text in examples.items():
        r = score_text(text)
        print(f"{name}: sentiment={r['sentiment']}, pos={r['positive_count']}, neg={r['negative_count']}, lit={r['litigation_count']}")


if __name__ == "__main__":
    _test_examples()
