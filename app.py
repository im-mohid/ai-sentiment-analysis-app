"""Sentiment Analysis Web App - lightweight VADER edition (Streamlit front-end).

Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import (
    analyze_reviews,
    load_csv,
    suggest_text_column,
    summarize,
    to_csv_bytes,
)
from src.model import (
    MAX_CHARS,
    MODEL_NAME,
    SENTIMENTS,
    ModelLoadError,
    load_classifier,
    predict_one,
)

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="wide")

COLORS = {"Positive": "#2e9e5b", "Negative": "#d64545", "Neutral": "#4a7bd0"}
ICONS = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sample_reviews.csv"

EXAMPLES = {
    "Positive": "This product is amazing! It works perfectly and exceeded my expectations.",
    "Negative": "The service was terrible. I waited an hour and nobody helped me.",
    "Neutral": "The product arrived today.",
}

CSS = """
<style>
.kpi {border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(128,128,128,.25);
      background: rgba(128,128,128,.07); border-left: 6px solid var(--c);}
.kpi .label {font-size: 0.85rem; opacity: .75;}
.kpi .value {font-size: 2rem; font-weight: 700; line-height: 1.2;}
.kpi .sub {font-size: 0.85rem; opacity: .75;}
.result {border-radius: 12px; padding: 18px 22px; margin: 12px 0;
         border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);
         border-left: 8px solid var(--c);}
.result .title {font-size: 1.9rem; font-weight: 700;}
.tablewrap {max-height: 420px; overflow: auto; border: 1px solid rgba(128,128,128,.25);
            border-radius: 8px;}
.tablewrap table {border-collapse: collapse; width: 100%; font-size: .9rem;}
.tablewrap th, .tablewrap td {padding: 6px 10px; text-align: left;
                              border-bottom: 1px solid rgba(128,128,128,.2);}
</style>
"""


def kpi_card(label: str, value: str, sub: str, color: str) -> str:
    return (
        f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def show_table(df: pd.DataFrame, max_rows: int = 500):
    """Show a DataFrame as a simple HTML table (works without extra libraries)."""
    shown = df.head(max_rows).copy()
    if "Sentiment" in shown.columns:
        shown["Sentiment"] = shown["Sentiment"].map(lambda s: f"{ICONS.get(s, '')} {s}")
    if "Score" in shown.columns:
        shown["Score"] = shown["Score"].map(lambda v: f"{v:.2f}")
    html = shown.to_html(index=False, border=0, na_rep="")
    html = " ".join(line.strip() for line in html.splitlines())
    st.markdown(f'<div class="tablewrap">{html}</div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows} of {len(df):,} rows.")


@st.cache_resource(show_spinner="Starting sentiment analyzer...")
def get_classifier():
    """Load the model once and keep it in memory across Streamlit reruns."""
    return load_classifier()


def ensure_model():
    try:
        return get_classifier()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()


# ---------------------------------------------------------------- Page 1
def set_example(text: str):
    st.session_state["review_text"] = text


def page_single():
    st.header("Single Review Analysis")
    st.write("Type a review or comment and the AI will classify it as Positive, Negative or Neutral.")

    st.caption("Try an example:")
    cols = st.columns(3)
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(f"{ICONS[name]} {name} example", on_click=set_example, args=(text,))

    review = st.text_area(
        "Your review", key="review_text", height=150,
        placeholder="Write an English review here...",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
            return
        if len(review) > MAX_CHARS:
            st.info(f"Only the first {MAX_CHARS} characters will be analyzed.")

        classifier = ensure_model()
        try:
            with st.spinner("Analyzing..."):
                result = predict_one(classifier, review)
        except Exception as exc:
            st.error(f"Something went wrong while analyzing the text: {exc}")
            return

        label, color = result["sentiment"], COLORS[result["sentiment"]]
        st.markdown(
            f'<div class="result" style="--c:{color}">'
            f'<div class="title" style="color:{color}">{ICONS[label]} {label}</div>'
            f'<div>Sentiment score: <b>{result["score"]:.1%}</b></div></div>',
            unsafe_allow_html=True,
        )

        scores = pd.DataFrame(
            {"Sentiment": list(result["all_scores"].keys()),
             "Share of text": list(result["all_scores"].values())}
        )
        fig = px.bar(scores, x="Share of text", y="Sentiment", orientation="h",
                     color="Sentiment", color_discrete_map=COLORS, range_x=[0, 1])
        fig.update_layout(showlegend=False, height=240, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("The score shows how strong the sentiment is (from VADER). "
                   "It is not a guarantee that the prediction is correct.")


# ---------------------------------------------------------------- Page 2
def page_batch():
    st.header("Batch Analysis & Dashboard")
    st.write("Upload a CSV file of reviews, choose the review column and analyze everything at once.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    use_sample = st.checkbox("Use the built-in sample CSV instead")

    source, name = None, None
    if use_sample:
        if SAMPLE_PATH.exists():
            source, name = SAMPLE_PATH, SAMPLE_PATH.name
        else:
            st.error("sample_data/sample_reviews.csv was not found.")
    elif uploaded is not None:
        source, name = uploaded, uploaded.name

    if source is not None:
        df = None
        try:
            df = load_csv(source)
        except ValueError as exc:
            st.error(str(exc))

        if df is not None:
            st.success(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {name}.")
            show_table(df.head(10))

            columns = list(df.columns)
            default = suggest_text_column(df)
            column = st.selectbox(
                "Which column contains the review text?", columns,
                index=columns.index(default),
            )

            if st.button("Analyze Reviews", type="primary"):
                classifier = ensure_model()
                bar = st.progress(0, text="Starting...")

                def update(fraction: float):
                    bar.progress(fraction, text=f"Analyzing... {fraction:.0%}")

                try:
                    results, skipped = analyze_reviews(df, column, classifier, update)
                except ValueError as exc:
                    bar.empty()
                    st.error(str(exc))
                except Exception as exc:
                    bar.empty()
                    st.error(f"Analysis failed: {exc}")
                else:
                    bar.empty()
                    st.session_state["results"] = results
                    st.session_state["skipped"] = skipped
                    st.success(f"Analyzed {len(results):,} reviews.")

    if "results" in st.session_state:
        st.divider()
        render_dashboard(st.session_state["results"], st.session_state.get("skipped", 0))


def render_dashboard(results: pd.DataFrame, skipped: int):
    st.subheader("Sentiment Dashboard")
    summary = summarize(results)

    if skipped:
        st.info(f"{skipped} row(s) were skipped (missing/empty text or over the row limit).")

    cards = st.columns(4)
    cards[0].markdown(kpi_card("Total reviews", f"{summary['total']:,}", "analyzed", "#888888"),
                      unsafe_allow_html=True)
    for card, label in zip(cards[1:], SENTIMENTS):
        card.markdown(
            kpi_card(f"{ICONS[label]} {label}", f"{summary['counts'][label]:,}",
                     f"{summary['percent'][label]}% of reviews", COLORS[label]),
            unsafe_allow_html=True,
        )

    counts_df = pd.DataFrame(
        {"Sentiment": SENTIMENTS, "Reviews": [summary["counts"][s] for s in SENTIMENTS]}
    )
    left, right = st.columns(2)
    with left:
        donut = px.pie(counts_df, names="Sentiment", values="Reviews", hole=0.5,
                       color="Sentiment", color_discrete_map=COLORS,
                       title="Sentiment distribution")
        st.plotly_chart(donut)
    with right:
        bars = px.bar(counts_df, x="Sentiment", y="Reviews", color="Sentiment",
                      color_discrete_map=COLORS, text="Reviews",
                      title="Reviews per sentiment")
        bars.update_layout(showlegend=False)
        st.plotly_chart(bars)

    st.subheader("Review-level results")
    selected = st.multiselect("Filter by sentiment", SENTIMENTS, default=SENTIMENTS)
    filtered = results[results["Sentiment"].isin(selected)]
    if filtered.empty:
        st.info("No reviews match the selected filter.")
    else:
        show_table(filtered)

    st.download_button(
        "⬇️ Download results (CSV)",
        data=to_csv_bytes(results),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------- Page 3
def page_about():
    st.header("About this project")
    st.markdown(
        f"""
**Engine:** {MODEL_NAME}, running locally on your computer.
**Framework:** Streamlit · **Charts:** Plotly · **Data:** Pandas

**How it works**
VADER looks up each word in a sentiment dictionary, applies simple rules
(negation like "not good", intensifiers like "very", punctuation, capital letters)
and combines everything into one *compound score* between -1 and +1.
Score ≥ 0.05 → Positive, score ≤ -0.05 → Negative, otherwise Neutral.

**Important notes**
- VADER is a **rule-based** tool, not a deep-learning model. It is fast and light,
  but it can miss sarcasm, slang and mixed opinions.
- It works for **English** text only. It is **not** a Bengali sentiment tool.
- The score shows how strong the sentiment is. It is **not** a guarantee of accuracy,
  and predictions can be wrong.
- Reviews longer than {MAX_CHARS} characters are truncated.
- Everything runs on your computer; no API key or internet connection is needed.
"""
    )


# ---------------------------------------------------------------- Main
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("💬 Sentiment Analyzer")
    page = st.sidebar.radio(
        "Navigate", ["Single Review", "Batch Analysis & Dashboard", "About"]
    )
    st.sidebar.caption("English text · VADER · Streamlit")

    if page == "Single Review":
        page_single()
    elif page == "Batch Analysis & Dashboard":
        page_batch()
    else:
        page_about()


main()
