import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
import time

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="🏦 Loan Approval Predictor", layout="wide", page_icon="💰")

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main {
        padding: 1.5rem 3rem;
        background-color: #F9FAFB;
        border-radius: 15px;
    }
    h1 {
        text-align: center;
        color: #2E7D32;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #43A047, #66BB6A);
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #388E3C, #4CAF50);
        transform: scale(1.03);
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4565/4565950.png", use_container_width=True)
st.sidebar.markdown("### 💳 Loan Approval Predictor")
st.sidebar.info("AI-powered model to predict loan approval chances based on user details or bulk CSV data.")

# -------------------------
# Load Model and Scaler
# -------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("🚨 Model or Scaler not found. Please check your files.")
    st.stop()

# -------------------------
# Header
# -------------------------
st.title("🏦 Smart Loan Approval Predictor")
st.caption("Use AI to evaluate loan approval chances — manually or in bulk CSV mode.")
st.divider()

# -------------------------
# Feature Mappings
# -------------------------
required_features = [
    "Age", "Sex", "Job", "Housing", "Saving accounts",
    "Checking account", "Credit amount", "Duration", "Purpose"
]

sex_map = {"male": 0, "female": 1}
housing_map = {"own": 0, "rent": 1, "free": 2}
saving_accounts_map = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3}
checking_account_map = {"little": 0, "moderate": 1, "rich": 2}
purpose_map = {
    "car": 0, "furniture/equipment": 1, "radio/TV": 2, "education": 3,
    "business": 4, "domestic appliances": 5, "repairs": 6,
    "vacation/others": 7
}
job_options = {
    "0 - Unskilled & Non-Resident": 0,
    "1 - Unskilled & Resident": 1,
    "2 - Skilled Employee": 2,
    "3 - Highly Skilled / Management": 3
}

# -------------------------
# Mode Selection
# -------------------------
mode = st.radio("🧠 Choose Prediction Mode", ["Manual Input", "Upload CSV"], horizontal=True)
st.divider()

# -------------------------
# Manual Input Mode
# -------------------------
if mode == "Manual Input":
    st.subheader("📋 Enter Applicant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job Type", list(job_options.keys()))
        housing = st.selectbox("Housing", list(housing_map.keys()))
    with col2:
        sex = st.selectbox("Sex", list(sex_map.keys()))
        saving = st.selectbox("Saving Accounts", list(saving_accounts_map.keys()))
        checking = st.selectbox("Checking Account", list(checking_account_map.keys()))
    with col3:
        credit_amount = st.number_input("Credit Amount", 4500, 20000, 5000)
        duration = st.number_input("Duration (months)", 4, 72, 12)
        purpose = st.selectbox("Purpose", list(purpose_map.keys()))

    if st.button("🔮 Predict Loan Status"):
        with st.spinner('Analyzing applicant profile...'):
            time.sleep(1.5)
            
            input_data = pd.DataFrame([[age,
                                        sex_map[sex],
                                        job_options[job],
                                        housing_map[housing],
                                        saving_accounts_map[saving],
                                        checking_account_map[checking],
                                        credit_amount,
                                        duration,
                                        purpose_map[purpose]]],
                                      columns=required_features)
            input_scaled = scaler.transform(input_data)
            preds = model.predict(input_scaled)

            # Handle models without predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)
                prob_good = probs[0][1] * 100
                prob_bad = probs[0][0] * 100
            else:
                scores = model.decision_function(input_scaled)
                prob_good = 1 / (1 + np.exp(-scores[0])) * 100
                prob_bad = 100 - prob_good

        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.markdown(f"""
                <div style="padding:25px; border-radius:15px; box-shadow:0 4px 10px rgba(0,0,0,0.1);
                            background-color:{'#E8F5E9' if preds[0]==1 else '#FFEBEE'};">
                    <h2 style="text-align:center; color:{'#2E7D32' if preds[0]==1 else '#C62828'};">
                        {'✅ Loan Approved' if preds[0]==1 else '❌ Loan Rejected'}
                    </h2>
                    <p style="text-align:center; color:gray; font-size:17px;">
                        Confidence: {prob_good:.1f}% positive | {prob_bad:.1f}% negative
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col_res2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_good,
                title={'text': "Approval Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if preds[0] == 1 else "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFCDD2"},
                        {'range': [50, 75], 'color': "#FFF59D"},
                        {'range': [75, 100], 'color': "#C8E6C9"}
                    ],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# -------------------------
# CSV Upload Mode
# -------------------------
else:
    st.subheader("📁 Upload Applicant Data (CSV)")
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        df = df[[c for c in required_features if c in df.columns]]

        # Encode categorical fields
        if "Sex" in df.columns: df["Sex"] = df["Sex"].map(sex_map).fillna(0)
        if "Housing" in df.columns: df["Housing"] = df["Housing"].map(housing_map).fillna(0)
        if "Saving accounts" in df.columns: df["Saving accounts"] = df["Saving accounts"].map(saving_accounts_map).fillna(0)
        if "Checking account" in df.columns: df["Checking account"] = df["Checking account"].map(checking_account_map).fillna(0)
        if "Purpose" in df.columns: df["Purpose"] = df["Purpose"].map(purpose_map).fillna(0)

        df = df.fillna(0)
        input_scaled = scaler.transform(df)
        preds = model.predict(input_scaled)

        # Handle models without predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)
            df["Prob_NotApproved"] = probs[:, 0] * 100
            df["Prob_Approved"] = probs[:, 1] * 100
        else:
            scores = model.decision_function(input_scaled)
            df["Prob_Approved"] = 1 / (1 + np.exp(-scores)) * 100
            df["Prob_NotApproved"] = 100 - df["Prob_Approved"]

        df["Prediction"] = preds
        st.success(f"✅ {len(df)} Applicants Processed Successfully!")

        st.markdown("### 👤 Applicant Results")
        for i, row in df.iterrows():
            with st.expander(f"Applicant #{i+1} — {'Approved ✅' if row['Prediction']==1 else 'Rejected ❌'}"):
                colA, colB = st.columns([2, 1])
                with colA:
                    st.write(row[required_features])
                    st.markdown(f"**Confidence:** {row['Prob_Approved']:.1f}% approved | {row['Prob_NotApproved']:.1f}% not approved")
                with colB:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=row["Prob_Approved"],
                        title={'text': "Approval %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "green" if row["Prediction"] == 1 else "red"},
                            'steps': [
                                {'range': [0, 50], 'color': "#FFCDD2"},
                                {'range': [50, 75], 'color': "#FFF59D"},
                                {'range': [75, 100], 'color': "#C8E6C9"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions CSV",
            df.to_csv(index=False).encode("utf-8"),
            "loan_predictions.csv",
            "text/csv"
        )
