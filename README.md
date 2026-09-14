# Fraud Detection Dashboard — Run Instructions

This is a **Streamlit** dashboard built for non-technical stakeholders (managers, business
reviewers) to explore the fraud detection model without touching any code or notebooks.

## 1. Install requirements

```bash
pip install streamlit joblib pandas numpy matplotlib scikit-learn xgboost
```

## 2. Place your trained model files next to `app.py`

Copy these two files (produced by `xgboost_fraud_pipeline_from_scratch.ipynb`) into the
same folder as `app.py`:

```
fraud_xgboost_pipeline.joblib
fraud_xgboost_metadata.json
```

If these files are missing, the dashboard still runs in **DEMO MODE** with a simple
illustrative scorer, clearly labeled as such, so you can still preview the interface.

## 3. Run the dashboard

```bash
streamlit run app.py
```

This opens the dashboard automatically in your browser (usually at `http://localhost:8501`).

## What's inside

| Page | Purpose |
|---|---|
| **Overview** | Executive KPIs — precision, recall, false alarms, confusion matrix — in plain language |
| **Try a Transaction** | Interactive form: fill in a hypothetical transaction and get an instant fraud score with a plain-English explanation |
| **How the Model Decides** | Simplified + technical feature importance view |
| **Limitations** | Honest disclosure of the model's current limitations (synthetic data, random split, etc.) |

## Notes

- The dashboard calls `pipeline.predict_proba()` directly on raw transaction-level columns —
  it does **not** re-implement any encoding or feature engineering manually. All of that logic
  lives inside the saved pipeline itself, exactly as designed in the training notebook.
- The "Overview" page shows the model's final reported test-set metrics (fixed, historical
  numbers). The "Try a Transaction" page computes a **live** prediction using the loaded model.
