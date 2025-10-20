import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Ultra Credit Risk Predictor")
st.write("Upload your data or enter details manually to predict credit risk.")

# -------------------------
# Feature Definitions
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

# -------------------------
# Mode Selection
# -------------------------
mode = st.radio("Select Mode:", ["Manual Input", "Upload CSV"])

# -------------------------
# Manual Input Mode
# -------------------------
if mode == "Manual Input":
    st.subheader("📋 Enter Customer Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", [0, 1, 2, 3])
        housing = st.selectbox("Housing", list(housing_map.keys()))
    with col2:
        sex = st.selectbox("Sex", list(sex_map.keys()))
        saving = st.selectbox("Saving Accounts", list(saving_accounts_map.keys()))
        checking = st.selectbox("Checking Account", list(checking_account_map.keys()))
    with col3:
        credit_amount = st.number_input("Credit Amount", 100, 20000, 1000)
        duration = st.number_input("Duration (months)", 4, 72, 12)
        purpose = st.selectbox("Purpose", list(purpose_map.keys()))

    if st.button("🔮 Predict"):
        input_data = pd.DataFrame([[
            age, sex_map.get(sex, 0), job,
            housing_map.get(housing, 0),
            saving_accounts_map.get(saving, 0),
            checking_account_map.get(checking, 0),
            credit_amount, duration,
            purpose_map.get(purpose, 0)
        ]], columns=required_features)

        input_data = input_data.fillna(0)  # safety
        input_scaled = scaler.transform(input_data)
        preds = model.predict(input_scaled)
        probs = model.predict_proba(input_scaled)

        pred_class = preds[0]
        prob_good = probs[0][1] * 100
        prob_bad = probs[0][0] * 100

        # Result Card
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:15px; 
                        background-color:{'#E8F5E9' if pred_class==1 else '#FFEBEE'}; 
                        text-align:center;">
                <h2 style="color:{'#2E7D32' if pred_class==1 else '#C62828'};">
                    {'✅ Approved' if pred_class==1 else '❌ Not Approved'}
                </h2>
                <p style="font-size:18px; color:gray;">
                    Confidence: {prob_good:.1f}% positive | {prob_bad:.1f}% negative
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_good,
            title={'text': "Approval Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if pred_class == 1 else "red"},
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
elif mode == "Upload CSV":
    file = st.file_uploader("Upload your CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        # Keep only required features
        df = df[[c for c in required_features if c in df.columns]]

        # Encode categorical fields + replace unknowns with 0
        if "Sex" in df.columns: df["Sex"] = df["Sex"].map(sex_map).fillna(0)
        if "Housing" in df.columns: df["Housing"] = df["Housing"].map(housing_map).fillna(0)
        if "Saving accounts" in df.columns: df["Saving accounts"] = df["Saving accounts"].map(saving_accounts_map).fillna(0)
        if "Checking account" in df.columns: df["Checking account"] = df["Checking account"].map(checking_account_map).fillna(0)
        if "Purpose" in df.columns: df["Purpose"] = df["Purpose"].map(purpose_map).fillna(0)

        # Fill any remaining NaNs
        df = df.fillna(0)

        input_data = df
        input_scaled = scaler.transform(input_data)
        preds = model.predict(input_scaled)
        probs = model.predict_proba(input_scaled)

        df["Prediction"] = preds
        df["Prob_NotApproved"] = probs[:, 0]
        df["Prob_Approved"] = probs[:, 1]

        st.subheader("📊 Predictions Table")
        st.dataframe(df.head())

        # Pie Chart
        pie = px.pie(df, names="Prediction", title="Approval vs Rejection", hole=0.4,
                     color="Prediction", color_discrete_map={0: "red", 1: "green"})
        st.plotly_chart(pie, use_container_width=True)

        # Bar Chart
        bar = px.bar(
            df.groupby("Prediction")[["Prob_Approved"]].mean().reset_index(),
            x="Prediction", y="Prob_Approved",
            title="Average Approval Confidence",
            color="Prediction", color_discrete_map={0: "red", 1: "green"}
        )
        st.plotly_chart(bar, use_container_width=True)

