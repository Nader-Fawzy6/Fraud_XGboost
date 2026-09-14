"""
Fraud Detection Dashboard — for non-technical stakeholders
============================================================
Built against the actual trained pipeline:
    fraud_xgboost_pipeline.joblib
    fraud_xgboost_metadata.json

Run with:
    streamlit run app.py

IMPORTANT: the custom transformer classes below (FraudFeatureEngineer,
MerchantFrequencyEncoder) MUST be defined in this file, with the exact
same names and logic used during training, or joblib.load() will fail
to deserialize the pipeline.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

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


# ====================================================================
# EXACT custom transformer definitions used during training.
# These must match the training notebook precisely.
# ====================================================================
US_STATES_AND_TERRITORIES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC","AA","AE","AP",
}


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering for one transaction per row."""

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _money_to_float(series):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        return pd.to_numeric(
            series.astype("string")
                  .str.replace("$", "", regex=False)
                  .str.replace(",", "", regex=False),
            errors="coerce",
        )

    def transform(self, X):
        X = X.copy()

        money_cols = ["Amount", "Credit Limit", "Yearly Income - Person",
                      "Total Debt", "Per Capita Income - Zipcode"]
        for col in money_cols:
            if col in X.columns:
                X[col] = self._money_to_float(X[col])

        dt = pd.to_datetime(
            X["Year"].astype("Int64").astype(str) + "-" +
            X["Month"].astype("Int64").astype(str) + "-" +
            X["Day"].astype("Int64").astype(str) + " " +
            X["Time"].astype(str),
            errors="coerce",
        )

        X["hour"] = dt.dt.hour
        X["day_of_week"] = dt.dt.dayofweek
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype("int8")
        X["month"] = dt.dt.month
        X["day_of_month"] = dt.dt.day
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

        errors = X["Errors?"].fillna("").astype(str)
        error_patterns = {
            "bad_pin": "Bad PIN", "insufficient_balance": "Insufficient Balance",
            "technical_glitch": "Technical Glitch", "bad_card_number": "Bad Card Number",
            "bad_cvv": "Bad CVV", "bad_expiration": "Bad Expiration", "bad_zipcode": "Bad Zipcode",
        }
        X["has_error"] = errors.ne("").astype("int8")
        for new_col, pattern in error_patterns.items():
            X[new_col] = errors.str.contains(pattern, regex=False, na=False).astype("int8")
        error_cols = list(error_patterns)
        X["error_count"] = X[error_cols].sum(axis=1)

        X["is_international"] = (
            X["Merchant State"].notna() & ~X["Merchant State"].isin(US_STATES_AND_TERRITORIES)
        ).astype("int8")
        X["same_state"] = X["Merchant State"].eq(X["State"]).astype("int8")
        X["same_city"] = X["Merchant City"].eq(X["City"]).astype("int8")

        merchant_zip = X["Zip"].astype("string").str.replace(r"\.0$", "", regex=True)
        home_zip = X["Zipcode"].astype("string").str.replace(r"\.0$", "", regex=True)
        X["same_zip"] = merchant_zip.eq(home_zip).fillna(False).astype("int8")

        X["is_online"] = X["Use Chip"].eq("Online Transaction").astype("int8")

        acct_open = pd.to_datetime(X["Acct Open Date"], format="%m/%Y", errors="coerce")
        expires = pd.to_datetime(X["Expires"], format="%m/%Y", errors="coerce")
        X["account_age_days"] = (dt - acct_open).dt.days
        X["months_until_expiry"] = (expires.dt.year - dt.dt.year) * 12 + (expires.dt.month - dt.dt.month)

        drop_cols = [
            "User", "Card", "Merchant State", "Merchant City", "City", "State",
            "Year", "Month", "Day", "Time", "Expires", "Acct Open Date",
            "Year PIN last Changed", "Zip", "Zipcode", "Errors?",
        ]
        return X.drop(columns=drop_cols, errors="ignore")


class MerchantFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Learn merchant counts only from the training rows seen by fit()."""

    def __init__(self, rare_threshold=5):
        self.rare_threshold = rare_threshold

    def fit(self, X, y=None):
        X = X.copy()
        self.freq_map_ = X["Merchant Name"].value_counts(dropna=False)
        return self

    def transform(self, X):
        X = X.copy()
        freq = X["Merchant Name"].map(self.freq_map_).fillna(0)
        X["merchant_frequency"] = freq.astype("float32")
        X["is_rare_merchant"] = (freq <= self.rare_threshold).astype("int8")
        return X.drop(columns=["Merchant Name"])


# ------------------------------------------------------------------
# IMPORTANT FOR STREAMLIT CLOUD / JOBLIB DESERIALIZATION
# ------------------------------------------------------------------
# The pipeline was trained/saved from a notebook, so these custom
# transformers were pickled as belonging to the module "__main__".
# Streamlit executes app.py through its own runner, where sys.modules
# ["__main__"] is not necessarily this script's namespace. Register
# the classes explicitly so joblib/pickle can resolve the original
# references stored inside fraud_xgboost_pipeline.joblib.
_main_module = sys.modules.get("__main__")
if _main_module is not None:
    setattr(_main_module, "FraudFeatureEngineer", FraudFeatureEngineer)
    setattr(_main_module, "MerchantFrequencyEncoder", MerchantFrequencyEncoder)


# ------------------------------------------------------------------
# Fixed, historical evaluation numbers for the Overview page
# (adjust these if you re-train and get new final test-set numbers).
# ------------------------------------------------------------------
REPORTED_RESULTS = {
    "precision": 0.9189,
    "recall": 0.6270,
    "f1": 0.7454,
    "pr_auc": 0.7587,
    "roc_auc": 0.9896,
    "confusion_matrix": np.array([[399485, 27], [182, 306]]),
    "fraud_rate_real_world": 0.00122,
}

TOP_FEATURES_STORY = [
    "Whether the merchant's ZIP code matches the cardholder's home ZIP code",
    "Whether the merchant's city matches the cardholder's home city",
    "Whether the merchant's state matches the cardholder's home state",
    "The merchant's category (e.g. money transfer, travel, cash-like services)",
    "Whether this is a rarely-used / newly-seen merchant",
    "Whether the transaction happened outside the home country",
]

# Curated, human-readable subset of the 109 MCC codes actually seen in training
MCC_CHOICES = {
    "Grocery Store / Supermarket": 5411,
    "Miscellaneous Food Store": 5499,
    "Gas Station": 5541,
    "Restaurant": 5812,
    "Fast Food Restaurant": 5814,
    "Drug Store / Pharmacy": 5912,
    "Money Transfer Service": 4829,
    "Utilities": 4900,
    "Telecom Services": 4814,
    "Cable / Other Pay TV": 4899,
    "Airline": 3000,
    "Cruise Line": 4411,
    "Car Rental": 3389,
    "Hotel / Lodging": 3504,
    "Electronics Store": 5732,
    "Department Store": 5311,
    "Discount Store": 5310,
    "Book Store": 5942,
    "Digital Goods / Software": 5045,
    "Government Services": 9402,
    "Wire Transfer / Financial Institution": 6300,
    "Other (Miscellaneous)": 7995,
}

US_STATES = sorted([s for s in US_STATES_AND_TERRITORIES if s not in {"AA", "AE", "AP"}])
FOREIGN_COUNTRIES = ["Nigeria", "Russia", "China", "Romania", "Vietnam", "Ukraine", "Brazil", "Canada"]

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
def demo_mode_score(same_state, same_city, same_zip, is_intl, is_rare, amount, has_errors):
    score = 0.02
    if same_zip == 0:
        score += 0.20
    if same_city == 0:
        score += 0.15
    if same_state == 0:
        score += 0.10
    if is_intl:
        score += 0.05
    if is_rare:
        score += 0.10
    if amount > 800:
        score += 0.05
    if has_errors:
        score += 0.05
    return float(min(score, 0.99))


# ------------------------------------------------------------------
# Build a single raw transaction row matching the pipeline's expected
# input schema — exactly the raw merged columns, before any
# feature engineering (the pipeline does all of that internally).
# ------------------------------------------------------------------
def build_transaction_row(inputs: dict) -> pd.DataFrame:
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
        "Merchant State": inputs["merchant_state"],
        "Zip": inputs["merchant_zip"],
        "MCC": inputs["mcc"],
        "Errors?": ",".join(inputs["errors"]) if inputs["errors"] else np.nan,
        "Current Age": inputs["age"],
        "Retirement Age": inputs["retirement_age"],
        "Gender": inputs["gender"],
        "City": inputs["home_city"],
        "State": inputs["home_state"],
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
    st.sidebar.success(f"✅ Model loaded — decision threshold: {threshold*100:.1f}%")


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
- Out of every **100 fraud alerts** the system raises, about **{REPORTED_RESULTS['precision']*100:.0f} are genuine fraud**.
- The system successfully catches about **{REPORTED_RESULTS['recall']*100:.0f} out of every 100** real fraud cases.
- Only **{REPORTED_RESULTS['confusion_matrix'][0,1]} legitimate customers** out of 399,512 tested were
  ever incorrectly flagged — a false-alarm rate of **0.007%**.
            """
        )
        st.info(
            "The real-world fraud rate is extremely low — only about "
            f"**{REPORTED_RESULTS['fraud_rate_real_world']*100:.3f}%** of all transactions are fraud. "
            "This system was evaluated on that realistic rate, not an inflated one."
        )

    with right:
        st.subheader("Test-set results (400,000 unseen transactions)")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        cm = REPORTED_RESULTS["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
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
        "Fill in a hypothetical transaction below and see how the system scores it live. "
        "The cardholder's profile fields are pre-filled — feel free to change them too."
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
                ["Same city, state & ZIP (typical)", "Same city & state, different ZIP",
                 "Different city, same state", "Different state (domestic)",
                 "Outside the country"],
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
            card_type = st.selectbox("Card type", ["Credit", "Debit", "Debit (Prepaid)"])
            has_chip = st.selectbox("Card has chip", ["YES", "NO"])
            credit_limit = st.number_input("Credit limit ($)", 0, 200000, 15000, step=500)
            cards_issued = st.number_input("Times this card was reissued", 1, 5, 1)

        submitted = st.form_submit_button("Score this transaction", type="primary")

    if submitted:
        if location_choice == "Same city, state & ZIP (typical)":
            merchant_city, merchant_state, merchant_zip = home_city, home_state, home_zip
        elif location_choice == "Same city & state, different ZIP":
            merchant_city, merchant_state = home_city, home_state
            merchant_zip = str(int(home_zip) + 5) if home_zip.isdigit() else "99999"
        elif location_choice == "Different city, same state":
            merchant_city, merchant_state, merchant_zip = "Nearby Town", home_state, "99999"
        elif location_choice == "Different state (domestic)":
            merchant_city = "Other City"
            merchant_state = "TX" if home_state != "TX" else "CA"
            merchant_zip = "88888"
        else:  # Outside the country
            merchant_city, merchant_state, merchant_zip = "Abroad", FOREIGN_COUNTRIES[0], np.nan

        inputs = dict(
            amount=amount, use_chip=use_chip, mcc=MCC_CHOICES[mcc_label],
            date=tx_date, time=tx_time, merchant_name=merchant_name,
            merchant_city=merchant_city, merchant_state=merchant_state, merchant_zip=merchant_zip,
            errors=errors, age=age, retirement_age=retirement_age, gender=gender,
            home_city=home_city, home_state=home_state, home_zip=home_zip,
            per_capita_income=per_capita_income, yearly_income=yearly_income,
            total_debt=total_debt, fico=fico, num_cards=num_cards,
            card_brand=card_brand, card_type=card_type,
            expires="12/2027", has_chip=has_chip, cards_issued=cards_issued,
            credit_limit=credit_limit, acct_open_date="06/2015", pin_year=2019,
        )

        tx_row = build_transaction_row(inputs)

        if MODEL_LOADED:
            probability = float(pipeline.predict_proba(tx_row)[:, 1][0])
            engineered = pipeline.named_steps["feature_engineering"].transform(tx_row)
            same_state = int(engineered["same_state"].iloc[0])
            same_city = int(engineered["same_city"].iloc[0])
            same_zip = int(engineered["same_zip"].iloc[0])
            is_intl = int(engineered["is_international"].iloc[0])
        else:
            same_state = int(merchant_state == home_state)
            same_city = int(merchant_city == home_city)
            same_zip = int(str(merchant_zip) == str(home_zip))
            is_intl = int(location_choice == "Outside the country")
            probability = demo_mode_score(
                same_state, same_city, same_zip, is_intl,
                merchant_known == "No", amount, bool(errors),
            )

        is_fraud = probability >= threshold

        st.markdown("---")
        st.subheader("Result")

        r1, r2 = st.columns([1, 1.4])
        with r1:
            if is_fraud:
                st.error(f"### 🚨 Flagged as likely FRAUD\n**Fraud probability: {probability*100:.2f}%**")
            else:
                st.success(f"### ✅ Looks like a legitimate transaction\n**Fraud probability: {probability*100:.2f}%**")
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
        if same_zip == 0:
            reasons.append("📮 The merchant's **ZIP code differs** from the cardholder's home ZIP — the single strongest signal for this model.")
        if same_city == 0:
            reasons.append("🏙️ The merchant is in a **different city** than the cardholder's home city.")
        if same_state == 0:
            reasons.append("📍 The merchant is in a **different state** from the cardholder's home address.")
        if is_intl:
            reasons.append("🌍 The transaction took place **outside the cardholder's home country**.")
        if merchant_known == "No":
            reasons.append("🏪 This is a **rarely-used merchant**.")
        if amount > 500:
            reasons.append("💵 The transaction amount is **relatively high**.")
        if errors:
            reasons.append(f"⚠️ The attempt reported an error ({', '.join(errors)}).")
        if not reasons:
            reasons.append("✅ Nothing unusual was detected — location, merchant, and amount all match the cardholder's typical pattern.")
        for r in reasons:
            st.markdown(f"- {r}")

        st.caption(
            "Note: location mismatches (ZIP / city / state) are, for this trained model, "
            "stronger fraud signals than crossing an international border by itself — "
            "an international transaction that otherwise looks unremarkable may still "
            "score as legitimate."
        )


# ====================================================================
# PAGE 3 — HOW THE MODEL DECIDES
# ====================================================================
elif page == "How the Model Decides":
    st.title("How the Model Makes Its Decisions")
    st.markdown(
        "The model looks at dozens of details on every transaction, but a handful of "
        "patterns matter far more than the rest."
    )

    st.subheader("The signals that matter most, in plain terms")
    for i, text in enumerate(TOP_FEATURES_STORY, start=1):
        st.markdown(f"**{i}.** {text}")

    st.markdown("---")
    st.subheader("Feature importance (technical view)")

    fi = None
    if MODEL_LOADED:
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importances = pipeline.named_steps["classifier"].feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
        except Exception as e:
            st.warning(f"Could not read feature importances from the loaded model: {e}")

    if fi is None:
        fi = pd.Series({
            "same_zip": 0.208, "same_city": 0.172, "same_state": 0.141,
            "MCC_4784": 0.076, "is_rare_merchant": 0.024, "is_international": 0.016,
        }).sort_values(ascending=False)
        st.caption("Showing previously reported values (no live model loaded).")

    # A ranked table makes the most influential features easy to inspect or export.
    feature_ranking = (
        fi.rename_axis("Feature")
          .reset_index(name="Importance")
    )
    feature_ranking.index = feature_ranking.index + 1
    feature_ranking.index.name = "Rank"
    st.dataframe(
        feature_ranking.style.format({"Importance": "{:.2%}"}),
        use_container_width=True,
    )

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
- **The train/validation/test split was random, not time-based.** For production, the model
  should be re-tested using a strict "train on the past, test on the future" split.
- **It will still make mistakes.** Roughly 37 out of every 100 real fraud cases may go
  undetected, and a small number of genuine customers will occasionally be flagged
  by mistake. This system is meant to prioritize human review, not fully replace it.
- **Location-mismatch signals dominate this model.** An international transaction that
  otherwise matches the cardholder's normal spending pattern may not be flagged — this
  reflects the specific training sample and should be re-validated on more diverse data.
- **It should be periodically retested.** Fraud patterns change over time; a model
  trained today should be monitored and refreshed as new data becomes available.
        """
    )
