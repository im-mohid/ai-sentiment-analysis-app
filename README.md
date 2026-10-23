# 💬 Sentiment Analysis App (Lightweight VADER Edition)

A Streamlit web app that classifies English reviews as **Positive**, **Negative** or **Neutral**, with batch CSV analysis, an interactive dashboard and CSV export.

This edition uses **VADER** (a lexicon- and rule-based sentiment tool). It needs **no PyTorch, no model download** and runs on old computers. It is not a deep-learning model, so it is less accurate on sarcasm and tricky sentences.

Stack: Python · Streamlit · VADER (`vaderSentiment`) · Pandas · Plotly

## Project structure
```
sentiment-analysis-app-lite/
├── app.py                  # Streamlit user interface
├── requirements.txt
├── README.md / LINKEDIN.md
├── sample_data/sample_reviews.csv
└── src/
    ├── __init__.py
    ├── model.py            # VADER sentiment engine
    └── analysis.py         # CSV loading, batch analysis, summary, export
```

## Setup (Windows + VS Code, Python 3.12)

Open the project folder in VS Code and the terminal (`` Ctrl + ` ``). If you used an older venv with torch, delete it first:

```powershell
deactivate
Remove-Item -Recurse -Force venv
```

Then:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y pyarrow
```

(`pip uninstall -y pyarrow` is a safety step for very old CPUs: pyarrow is not used by this app.)

### Check step by step (optional but useful on an old PC)
```powershell
python -c "import numpy, pandas; print('numpy + pandas OK')"
python -c "import vaderSentiment; print('vader OK')"
python -c "import plotly; print('plotly OK')"
python -c "import streamlit; print('streamlit OK')"
```
If one of them closes silently or prints an "Illegal instruction" error, tell us which one.

### Run
```powershell
python -m streamlit run app.py
```
Open **http://localhost:8501** if the browser does not open automatically. Stop with `Ctrl + C`.

## Testing checklist
- [ ] "This product is amazing!" → Positive
- [ ] "The service was terrible." → Negative
- [ ] "The product arrived today." → Neutral
- [ ] Empty input → warning, no crash
- [ ] Sample CSV: preview + column selector; 2 rows skipped (missing values)
- [ ] Analyze Reviews: progress bar, KPI cards, donut chart, bar chart, table, filter
- [ ] Download button gives a CSV (Review, Sentiment, Score) that opens in Excel
- [ ] Non-CSV or empty file → friendly error message

> Predictions are rule-based and probabilistic in spirit: some sentences will be misjudged (for example "I expected better" is scored Positive because "better" is a positive word). Use clear examples in your demo and be honest about this limitation.

## Troubleshooting
| Problem | Fix |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then activate again |
| `pip` not recognized | The venv is not active. Run `.\venv\Scripts\Activate.ps1` or use `.\venv\Scripts\python.exe -m pip ...` |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |
| Port busy | `python -m streamlit run app.py --server.port 8502` |
| Python crashes silently on import | Note which import (see step-by-step check) and report it |

## Limitations
English only · rule-based (can miss sarcasm and slang) · score ≠ accuracy · reviews truncated to 1000 characters.
