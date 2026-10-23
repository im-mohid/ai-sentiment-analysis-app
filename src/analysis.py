"""CSV loading, batch analysis and summary helpers (no Streamlit code here)."""
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from .model import SENTIMENTS, predict_batch

MAX_ROWS = 5000  # keeps the demo fast on a normal laptop CPU


def load_csv(source) -> pd.DataFrame:
    """Read a CSV (uploaded file or file path) and validate it."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            df = pd.read_csv(source, encoding="utf-8-sig")
        except UnicodeDecodeError:  # e.g. files saved by older Excel versions
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be parsed. Please upload a valid CSV file.")
    except Exception as exc:
        raise ValueError(f"The file could not be read: {exc}")

    if df.empty or df.shape[1] == 0:
        raise ValueError("The CSV has no data rows.")
    return df


def suggest_text_column(df: pd.DataFrame):
    """Guess which column holds the review text."""
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not text_cols:
        return df.columns[0]

    # 1) A text column whose name looks like review text (e.g. "review_text").
    hints = ("review", "text", "comment", "feedback", "content", "message")
    for col in text_cols:
        if any(h in str(col).lower() for h in hints):
            return col

    # 2) Otherwise the text column with the longest average text.
    return max(text_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())


def analyze_reviews(
    df: pd.DataFrame,
    column,
    classifier,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Run sentiment analysis on one column.

    Returns (results_dataframe, number_of_skipped_rows).
    Results have the columns: Review, Sentiment, Score.
    """
    reviews = df[column].dropna().astype(str).str.strip()
    reviews = reviews[reviews != ""]
    skipped = len(df) - len(reviews)

    if reviews.empty:
        raise ValueError("The selected column has no usable review text.")
    if len(reviews) > MAX_ROWS:
        skipped += len(reviews) - MAX_ROWS
        reviews = reviews.head(MAX_ROWS)

    predictions = predict_batch(
        classifier, reviews.tolist(), progress_callback=progress_callback
    )
    results = pd.DataFrame(
        {
            "Review": reviews.tolist(),
            "Sentiment": [p["sentiment"] for p in predictions],
            "Score": [round(p["score"], 4) for p in predictions],
        }
    )
    return results, skipped


def summarize(results: pd.DataFrame) -> Dict:
    """Counts and percentages per sentiment."""
    total = len(results)
    counts = results["Sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)
    percent = (counts / total * 100).round(1) if total else counts * 0.0
    return {"total": total, "counts": counts.to_dict(), "percent": percent.to_dict()}


def to_csv_bytes(results: pd.DataFrame) -> bytes:
    """CSV bytes with a UTF-8 BOM so Excel opens special characters correctly."""
    return results.to_csv(index=False).encode("utf-8-sig")
