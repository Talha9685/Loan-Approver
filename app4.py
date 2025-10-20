import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="💰 Loan Approval Predictor v2.0", layout="wide", page_icon="🏦")

# -------------------------
# Custom CSS (Dark/Light)
# -------------------------
st.markdown("""
<style>
    @media (prefers-color-scheme: dark) {
        .main { background-color: #121212; color: #E0E0E0; }
        h1, h2, h3 { color: #00E676 !important; }
    }
    .stButton>button {
        background: linear-gradient(90deg, #36d1dc, #5b86e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model + Scaler
# -------------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        st.error("🚨 Model or Scaler file missing! Please ensure both are in the same directory.")
        st.stop()

model, scaler = load_assets()

# -------------------------
# Helper: Encode categoricals
# -------------------------
def encode_manual_input(df):
    mapping = {
        "Sex": {"male": 0, "female": 1},
        "Housing": {"own": 0, "rent": 1, "free": 2},
        "Saving accounts": {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3, "unknown": 4},
        "Checking account": {"little": 0, "moderate": 1, "rich": 2, "unknown": 3},
        "Purpose": {"car": 0, "education": 1, "furniture": 2, "business": 3, "repairs": 4}
    }

    for col, map_dict in mapping.items():
        df[col] = df[col].map(map_dict)
    return df

# -------------------------
# App Title
# -------------------------
st.title("🏦 Loan Approval Predictor v2.0")
st.markdown("#### Predict your loan approval chances instantly 🔥")

# -------------------------
# Mode Selector
# -------------------------
mode = st.radio("🧠 Choose Prediction Mode", ["Manual Input", "Upload CSV"], horizontal=True)

# -------------------------
# Manual Input
# -------------------------
if mode == "Manual Input":
    st.subheader("🔧 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=80, step=1)
        Job = st.selectbox("Job Type", [0, 1, 2, 3])
        Housing = st.selectbox("Housing", ["own", "rent", "free"])
    with col2:
        CreditAmount = st.number_input("Credit Amount (€)", min_value=100, step=100)
        Duration = st.number_input("Duration (months)", min_value=4, max_value=72, step=1)
        Purpose = st.selectbox("Purpose", ["car", "education", "furniture", "business", "repairs"])
    with col3:
        Sex = st.selectbox("Sex", ["male", "female"])
        SavingAccounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
        CheckingAccount = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])

    input_data = pd.DataFrame({
        "Age": [Age],
        "Job": [Job],
        "Housing": [Housing],
        "Credit amount": [CreditAmount],
        "Duration": [Duration],
        "Purpose": [Purpose],
        "Sex": [Sex],
        "Saving accounts": [SavingAccounts],
        "Checking account": [CheckingAccount]
    })

    if st.button("🚀 Predict Loan Approval"):
        with st.spinner("Crunching numbers... 🔄"):
            time.sleep(1)

            # Encode + Scale
            encoded = encode_manual_input(input_data)
            scaled = scaler.transform(encoded)

            # Prediction
            pred = model.predict(scaled)[0]
            decision_val = model.decision_function(scaled)[0]
            confidence = 1 / (1 + np.exp(-abs(decision_val)))  # Sigmoid approximation

            result = "✅ Approved" if pred == 1 else "❌ Rejected"

            st.success(f"**Result:** {result}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")
            st.balloons()
            st.toast("Prediction completed! 🎯")

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={"text": "Confidence Level (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "limegreen" if pred == 1 else "red"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# CSV Upload Mode
# -------------------------
else:
    st.subheader("📂 Upload Your Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File Uploaded Successfully!")
        st.dataframe(df.head())

        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Analyzing bulk data... 📊"):
                time.sleep(1)

                encoded_df = encode_manual_input(df.copy())
                scaled = scaler.transform(encoded_df.select_dtypes(include='number'))
                preds = model.predict(scaled)
                decision_vals = model.decision_function(scaled)
                confs = 1 / (1 + np.exp(-np.abs(decision_vals)))

                df["Prediction"] = ["Approved" if p == 1 else "Rejected" for p in preds]
                df["Confidence (%)"] = (confs * 100).round(2)

                st.success("🎯 Predictions Completed!")
                st.balloons()
                st.toast("All predictions are ready!")

                # Pie chart visualization
                fig_summary = px.pie(df, names="Prediction", title="Loan Approval Distribution")
                st.plotly_chart(fig_summary, use_container_width=True)

                # Display final data
                st.write("### 📄 Final Results")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results as CSV", csv, "loan_predictions.csv", "text/csv")
