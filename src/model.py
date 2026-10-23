"""Lightweight sentiment engine based on VADER (no PyTorch, no model download).

VADER is a lexicon- and rule-based sentiment analyzer. It is small, fast and
works on old computers. It is NOT a deep-learning model.
"""
from typing import Callable, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_NAME = "VADER (lexicon & rule-based sentiment analyzer)"

MAX_CHARS = 1000  # characters kept from each review (input-length limit)

SENTIMENTS = ["Positive", "Negative", "Neutral"]

# Standard VADER thresholds on the "compound" score (-1 to +1).
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


class ModelLoadError(RuntimeError):
    """Raised when the analyzer cannot be created."""


def load_classifier():
    """Create the VADER analyzer (instant, nothing to download)."""
    try:
        return SentimentIntensityAnalyzer()
    except Exception as exc:
        raise ModelLoadError(f"Could not start the sentiment analyzer: {exc}") from exc


def clean_text(text) -> str:
    """Convert to string, trim spaces and apply the input-length limit."""
    return str(text).strip()[:MAX_CHARS]


def _analyze(analyzer, text: str) -> Dict:
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label, strength = "Positive", abs(compound)
    elif compound <= NEGATIVE_THRESHOLD:
        label, strength = "Negative", abs(compound)
    else:
        label, strength = "Neutral", scores["neu"]

    return {
        "sentiment": label,
        "score": float(strength),
        # Share of the text that VADER found positive / negative / neutral.
        "all_scores": {
            "Positive": float(scores["pos"]),
            "Negative": float(scores["neg"]),
            "Neutral": float(scores["neu"]),
        },
        "compound": float(compound),
    }


def predict_one(classifier, text: str) -> Dict:
    """Predict one text."""
    text = clean_text(text)
    if not text:
        raise ValueError("Please enter some text to analyze.")
    return _analyze(classifier, text)


def predict_batch(
    classifier,
    texts: List[str],
    batch_size: int = 16,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict]:
    """Predict many texts, reporting progress (0.0 to 1.0)."""
    total = len(texts)
    results: List[Dict] = []
    for i, text in enumerate(texts, start=1):
        results.append(_analyze(classifier, clean_text(text)))
        if progress_callback and (i % batch_size == 0 or i == total):
            progress_callback(i / total)
    return results
