"""
Fraud Detection Dashboard — for non-technical stakeholders
============================================================
Run with:
    streamlit run app.py

Expects two files in the same folder (produced by the training notebook):
    fraud_xgboost_pipeline.joblib
    fraud_xgboost_metadata.json

If these files are not found, the dashboard still runs in DEMO MODE
with a clearly-labeled illustrative scorer, so the UI can be reviewed
even without the trained model.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection — Executive Dashboard",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("fraud_xgboost_pipeline.joblib")
METADATA_PATH = Path("fraud_xgboost_metadata.json")

# ------------------------------------------------------------------
# Final, reported evaluation results (from the project's test-set run).
# These are fixed historical numbers used only for the Overview page;
# they are NOT recomputed live.
# ------------------------------------------------------------------
REPORTED_RESULTS = {
    "precision": 0.9189,
    "recall": 0.6270,
    "f1": 0.7454,
    "pr_auc": 0.7587,
    "roc_auc": 0.9896,
    "threshold": 0.634,
    "confusion_matrix": np.array([[399485, 27], [182, 306]]),
    "fraud_rate_real_world": 0.00122,
}

TOP_FEATURES_STORY = [
    ("Different state from home address", "same_state"),
    ("Merchant category (e.g. money transfer, travel)", "MCC"),
    ("Different city from home address", "same_city"),
    ("Rarely-used merchant", "is_rare_merchant"),
    ("How often this merchant is used overall", "merchant_frequency"),
    ("Different ZIP code from home address", "same_zip"),
    ("Transaction made outside the home country", "is_international"),
]

MCC_CHOICES = {
    "Grocery Store / Supermarket": 5411,
    "Gas Station": 5541,
    "Restaurant": 5812,
    "Drug Store / Pharmacy": 5912,
    "Money Transfer Service": 4829,
    "Airline": 3000,
    "Cruise Line": 4411,
    "Electronics Store": 5732,
    "Department Store": 5311,
    "Online / Digital Goods": 5968,
}

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
]
FOREIGN_COUNTRIES = ["Nigeria", "Russia", "China", "Romania", "Vietnam", "Ukraine", "Brazil"]

ERROR_OPTIONS = [
    "Bad PIN", "Insufficient Balance", "Technical Glitch",
    "Bad Card Number", "Bad CVV", "Bad Expiration", "Bad Zipcode",
]


# ------------------------------------------------------------------
# Model loading (cached so the file is only read once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return pipeline, metadata, True
    return None, {"threshold": 0.5}, False


pipeline, metadata, MODEL_LOADED = load_model()
threshold = metadata.get("threshold", 0.5)


# ------------------------------------------------------------------
# Demo-mode fallback scorer (used only if no trained model is found)
# ------------------------------------------------------------------
def demo_mode_score(row: dict) -> float:
    """A simple, clearly-illustrative rule-based score — NOT the real model."""
    score = 0.02
    if row["_same_state"] == "No":
        score += 0.35
    if row["_same_city"] == "No":
        score += 0.15
    if row["_is_international"]:
        score += 0.35
    if row["_is_rare_merchant"]:
        score += 0.10
    if row["Amount"] > 800:
        score += 0.05
    if row["Errors?"]:
        score += 0.05
    return float(min(score, 0.99))


# ------------------------------------------------------------------
# Build a single raw transaction row matching the pipeline's expected
# input schema (same columns as X before feature engineering).
# ------------------------------------------------------------------
def build_transaction_row(inputs: dict) -> pd.DataFrame:
    merchant_state = inputs["merchant_state"]
    home_state = inputs["home_state"]

    row = {
        "User": 0,
        "Card": 0,
        "Year": inputs["date"].year,
        "Month": inputs["date"].month,
        "Day": inputs["date"].day,
        "Time": inputs["time"].strftime("%H:%M"),
        "Amount": inputs["amount"],
        "Use Chip": inputs["use_chip"],
        "Merchant Name": inputs["merchant_name"],
        "Merchant City": inputs["merchant_city"],
        "Merchant State": merchant_state,
        "Zip": inputs["merchant_zip"],
        "MCC": inputs["mcc"],
        "Errors?": ",".join(inputs["errors"]) if inputs["errors"] else np.nan,
        "Current Age": inputs["age"],
        "Retirement Age": inputs["retirement_age"],
        "Gender": inputs["gender"],
        "City": inputs["home_city"],
        "State": home_state,
        "Zipcode": inputs["home_zip"],
        "Per Capita Income - Zipcode": inputs["per_capita_income"],
        "Yearly Income - Person": inputs["yearly_income"],
        "Total Debt": inputs["total_debt"],
        "FICO Score": inputs["fico"],
        "Num Credit Cards": inputs["num_cards"],
        "Card Brand": inputs["card_brand"],
        "Card Type": inputs["card_type"],
        "Expires": inputs["expires"],
        "Has Chip": inputs["has_chip"],
        "Cards Issued": inputs["cards_issued"],
        "Credit Limit": inputs["credit_limit"],
        "Acct Open Date": inputs["acct_open_date"],
        "Year PIN last Changed": inputs["pin_year"],
    }
    return pd.DataFrame([row])


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Try a Transaction", "How the Model Decides", "Limitations"],
)

if not MODEL_LOADED:
    st.sidebar.warning(
        "⚠️ Trained model files not found in this folder.\n\n"
        "Running in **DEMO MODE** with an illustrative scorer.\n\n"
        f"Expected: `{MODEL_PATH.name}` and `{METADATA_PATH.name}`"
    )
else:
    st.sidebar.success("✅ Trained model loaded successfully.")


# ====================================================================
# PAGE 1 — OVERVIEW
# ====================================================================
if page == "Overview":
    st.title("Fraud Detection System — Executive Overview")
    st.markdown(
        "This system automatically reviews every card transaction and flags the ones "
        "that look like fraud, so a human team only needs to check a small, "
        "highly-targeted list instead of every transaction."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud caught (Recall)", f"{REPORTED_RESULTS['recall']*100:.0f}%")
    col2.metric("Alert accuracy (Precision)", f"{REPORTED_RESULTS['precision']*100:.0f}%")
    col3.metric("False alarms", f"{REPORTED_RESULTS['confusion_matrix'][0,1]}",
                help="Legitimate transactions incorrectly flagged, out of 400,000 tested")
    col4.metric("Fraud missed", f"{REPORTED_RESULTS['confusion_matrix'][1,0]}",
                help="Fraud cases the system did not catch, out of 400,000 tested")

    st.markdown("---")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("What this means in plain terms")
        st.markdown(
            f"""
- Out of every **100 fraud alerts** the system raises, about **{REPORTED_RESULTS['precision']*100:.0f} are genuine fraud** —
  the review team is rarely sent on a wild goose chase.
- The system successfully catches about **{REPORTED_RESULTS['recall']*100:.0f} out of every 100** real fraud cases.
- Only **{REPORTED_RESULTS['confusion_matrix'][0,1]} legitimate customers** out of 399,512 tested were
  ever incorrectly flagged — a false-alarm rate of **0.007%**.
            """
        )
        st.info(
            "The real-world fraud rate is extremely low — only about "
            f"**{REPORTED_RESULTS['fraud_rate_real_world']*100:.3f}%** of all transactions are fraud. "
            "Catching them without upsetting genuine customers is a genuinely hard problem, "
            "and this system was evaluated on that real, realistic rate — not an inflated one."
        )

    with right:
        st.subheader("Test-set results (400,000 unseen transactions)")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        cm = REPORTED_RESULTS["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        labels = ["Legitimate", "Fraud"]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=13, color=color, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Model said"); ax.set_ylabel("Actually was")
        plt.tight_layout()
        st.pyplot(fig)


# ====================================================================
# PAGE 2 — TRY A TRANSACTION (interactive demo)
# ====================================================================
elif page == "Try a Transaction":
    st.title("Try It Yourself — Score a Transaction")
    st.markdown(
        "Fill in a hypothetical transaction below and see how the system would score it. "
        "The customer's profile fields are pre-filled with a sample profile — feel free to change them too."
    )

    with st.form("transaction_form"):
        st.markdown("#### Transaction details")
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction amount ($)", min_value=0.0, value=85.0, step=1.0)
            use_chip = st.selectbox("Transaction channel", ["Chip Transaction", "Swipe Transaction", "Online Transaction"])
            mcc_label = st.selectbox("Merchant category", list(MCC_CHOICES.keys()))
        with c2:
            tx_date = st.date_input("Transaction date")
            tx_time = st.time_input("Transaction time")
            merchant_name = st.text_input("Merchant name / ID", value="Corner Grocery Store")
        with c3:
            location_choice = st.radio(
                "Merchant location relative to cardholder's home",
                ["Same city & state (typical)", "Different city, same state",
                 "Different state (domestic)", "Outside the country"],
            )
            merchant_known = st.radio("Is this a well-known / frequently-used merchant?", ["Yes", "No"])
            errors = st.multiselect("Any errors reported on this attempt?", ERROR_OPTIONS)

        st.markdown("#### Cardholder profile (pre-filled — edit if needed)")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            age = st.number_input("Age", 18, 100, 42)
            gender = st.selectbox("Gender", ["Male", "Female"])
            home_city = st.text_input("Home city", "Chicago")
            home_state = st.selectbox("Home state", US_STATES, index=US_STATES.index("IL"))
        with p2:
            home_zip = st.text_input("Home ZIP code", "60614")
            fico = st.slider("FICO score", 300, 850, 720)
            yearly_income = st.number_input("Yearly income ($)", 0, 1_000_000, 55000, step=1000)
            total_debt = st.number_input("Total debt ($)", 0, 1_000_000, 40000, step=1000)
        with p3:
            per_capita_income = st.number_input("Zip per-capita income ($)", 0, 200000, 28000, step=1000)
            retirement_age = st.number_input("Planned retirement age", 50, 80, 65)
            num_cards = st.number_input("Number of credit cards owned", 1, 10, 3)
            card_brand = st.selectbox("Card brand", ["Visa", "Mastercard", "Amex", "Discover"])
        with p4:
            card_type = st.selectbox("Card type", ["Debit", "Credit", "Debit (Prepaid)"])
            has_chip = st.selectbox("Card has chip", ["YES", "NO"])
            credit_limit = st.number_input("Credit limit ($)", 0, 200000, 15000, step=500)
            cards_issued = st.number_input("Times this card was reissued", 1, 5, 1)

        submitted = st.form_submit_button("Score this transaction", type="primary")

    if submitted:
        if location_choice == "Same city & state (typical)":
            same_city, same_state, is_intl = "Yes", "Yes", False
            merchant_city, merchant_state = home_city, home_state
        elif location_choice == "Different city, same state":
            same_city, same_state, is_intl = "No", "Yes", False
            merchant_city, merchant_state = "Nearby Town", home_state
        elif location_choice == "Different state (domestic)":
            same_city, same_state, is_intl = "No", "No", False
            merchant_city = "Other City"
            merchant_state = "TX" if home_state != "TX" else "CA"
        else:
            same_city, same_state, is_intl = "No", "No", True
            merchant_city = "Abroad"
            merchant_state = FOREIGN_COUNTRIES[0]

        inputs = dict(
            amount=amount, use_chip=use_chip, mcc=MCC_CHOICES[mcc_label],
            date=tx_date, time=tx_time, merchant_name=merchant_name,
            merchant_city=merchant_city, merchant_state=merchant_state,
            merchant_zip="00000" if is_intl else home_zip,
            errors=errors, age=age, retirement_age=retirement_age, gender=gender,
            home_city=home_city, home_zip=home_zip,
            per_capita_income=per_capita_income, yearly_income=yearly_income,
            total_debt=total_debt, fico=fico, num_cards=num_cards,
            card_brand=card_brand, card_type=card_type,
            expires="12/2027", has_chip=has_chip, cards_issued=cards_issued,
            credit_limit=credit_limit, acct_open_date="06/2015", pin_year=2019,
        )
        inputs["home_state"] = home_state

        tx_row = build_transaction_row(inputs)

        if MODEL_LOADED:
            probability = float(pipeline.predict_proba(tx_row)[:, 1][0])
        else:
            probability = demo_mode_score({
                "_same_state": same_state, "_same_city": same_city,
                "_is_international": is_intl,
                "_is_rare_merchant": merchant_known == "No",
                "Amount": amount, "Errors?": errors,
            })

        is_fraud = probability >= threshold

        st.markdown("---")
        st.subheader("Result")

        r1, r2 = st.columns([1, 1.4])
        with r1:
            if is_fraud:
                st.error(f"### 🚨 Flagged as likely FRAUD\n**Fraud probability: {probability*100:.1f}%**")
            else:
                st.success(f"### ✅ Looks like a legitimate transaction\n**Fraud probability: {probability*100:.1f}%**")
            st.caption(f"Decision threshold in use: {threshold*100:.1f}%")

        with r2:
            fig, ax = plt.subplots(figsize=(5, 1.1))
            ax.barh([0], [1], color="#e6e6e6")
            ax.barh([0], [probability], color="#d62728" if is_fraud else "#2ca02c")
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Fraud probability")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### Why the model likely reached this conclusion")
        reasons = []
        if is_intl:
            reasons.append("🌍 The transaction took place **outside the cardholder's home country** — historically the single strongest fraud signal in this data.")
        if same_state == "No":
            reasons.append("📍 The merchant is in a **different state** from the cardholder's home address.")
        if same_city == "No" and same_state == "Yes":
            reasons.append("🏙️ The merchant is in a **different city** than the cardholder's home city.")
        if merchant_known == "No":
            reasons.append("🏪 This is a **rarely-used merchant**, which is statistically more associated with fraud.")
        if amount > 500:
            reasons.append("💵 The transaction amount is **relatively high**.")
        if errors:
            reasons.append(f"⚠️ The attempt reported an error ({', '.join(errors)}).")
        if not reasons:
            reasons.append("✅ Nothing unusual was detected — location, merchant, and amount all match the cardholder's typical pattern.")
        for r in reasons:
            st.markdown(f"- {r}")


# ====================================================================
# PAGE 3 — HOW THE MODEL DECIDES
# ====================================================================
elif page == "How the Model Decides":
    st.title("How the Model Makes Its Decisions")
    st.markdown(
        "The model looks at dozens of details on every transaction, but a handful of "
        "patterns matter far more than the rest. These are ranked by how much each one "
        "actually influenced the model during training."
    )

    st.subheader("The 7 signals that matter most")
    for i, (plain_text, _) in enumerate(TOP_FEATURES_STORY, start=1):
        st.markdown(f"**{i}.** {plain_text}")

    st.markdown("---")
    st.subheader("Feature importance (technical view)")

    if MODEL_LOADED:
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importances = pipeline.named_steps["classifier"].feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
        except Exception:
            fi = None
    else:
        fi = None

    if fi is None:
        # Fallback to the reported values from the project report
        fi = pd.Series({
            "same_state": 0.281, "MCC_4784": 0.056, "same_city": 0.051,
            "is_rare_merchant": 0.043, "merchant_frequency": 0.028,
            "same_zip": 0.027, "is_international": 0.023,
        }).sort_values(ascending=False)
        st.caption("Showing previously reported values (no live model loaded).")

    fig, ax = plt.subplots(figsize=(7, 5))
    fi.sort_values().plot(kind="barh", ax=ax, color="#1f5fa8")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)


# ====================================================================
# PAGE 4 — LIMITATIONS
# ====================================================================
elif page == "Limitations":
    st.title("Honest Limitations")
    st.markdown(
        """
This system is a strong proof of concept, but a few things should be understood
before treating it as production-ready:

- **Training data is synthetic.** It was generated by a simulation, not collected from
  real banking incidents. Real-world performance must be re-validated on real transaction data.
- **The train/validation/test split was random, not time-based.** For a production deployment,
  the model should be re-tested using a strict "train on the past, test on the future" split,
  which better reflects how the system will actually be used.
- **It will still make mistakes.** About 37 out of every 100 real fraud cases may go
  undetected, and a small number of genuine customers will occasionally be flagged
  by mistake. This system is meant to prioritize human review, not fully replace it.
- **It should be periodically retested.** Fraud patterns change over time; a model
  trained today should be monitored and refreshed as new data becomes available.
        """
    )
