import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("heart.csv")
df["date_exam"] = pd.to_datetime(df["date_exam"])

st.set_page_config(page_title="Heart Dashboard + Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["📊 Dashboard", "🤖 Full Prediction Interface"])

# Shared model prep
X = df.drop(columns=["target", "date_exam"])
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# ================================
# 📊 DASHBOARD SECTION
# ================================
if page == "📊 Dashboard":
    st.title("Heart Disease Data Dashboard + Inline Prediction")

    # Sidebar Filters
    st.sidebar.header("Filters")
    sex_filter = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])
    target_filter = st.sidebar.selectbox("Heart Disease", ["All", "Positive", "Negative"])
    cp_filter = st.sidebar.selectbox("Chest Pain Type (cp)", ["All"] + sorted(df["cp"].astype(str).unique()))
    date_range = st.sidebar.date_input("Exam Date Range", [df["date_exam"].min(), df["date_exam"].max()])

    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["date_exam"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["date_exam"] <= pd.to_datetime(date_range[1]))
    ]

    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["sex"] == (1 if sex_filter == "Male" else 0)]
    if target_filter != "All":
        filtered_df = filtered_df[filtered_df["target"] == (1 if target_filter == "Positive" else 0)]
    if cp_filter != "All":
        filtered_df = filtered_df[filtered_df["cp"] == int(cp_filter)]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{filtered_df['age'].mean():.1f}")
    col2.metric("Avg Cholesterol", f"{filtered_df['chol'].mean():.1f} mg/dL")
    col3.metric("Heart Disease %", f"{filtered_df['target'].mean() * 100:.1f} %")

    # AI insights
    st.subheader("🧠 AI-Style Insights")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        insights = []
        if filtered_df["age"].mean() > 55:
            insights.append("High average age — elevated risk.")
        if filtered_df["chol"].mean() > 240:
            insights.append("Cholesterol levels above normal threshold.")
        if filtered_df["target"].mean() > 0.6:
            insights.append("Majority shows signs of heart disease.")
        for i in insights:
            st.markdown(f"🔎 {i}")

    # Histograms Grid
    st.subheader("Distributions")
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    sns.histplot(data=filtered_df, x="age", hue="target", ax=axs[0][0], multiple="stack")
    sns.histplot(data=filtered_df, x="chol", hue="target", ax=axs[0][1], multiple="stack")
    sns.histplot(data=filtered_df, x="trestbps", hue="target", ax=axs[0][2], multiple="stack")
    sns.histplot(data=filtered_df, x="thalach", hue="target", ax=axs[1][0], multiple="stack")
    sns.histplot(data=filtered_df, x="oldpeak", hue="target", ax=axs[1][1], multiple="stack")
    axs[1][2].axis("off")
    st.pyplot(fig)

    # Inline Prediction
    st.subheader("🔍 Try a Quick Prediction (Inline)")

    with st.form("inline_form"):
        age = st.slider("Age", 20, 80, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.slider("Chest Pain Type", 0, 3, 1)
        trestbps = st.slider("Resting BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.slider("Resting ECG", 0, 2, 1)
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = st.slider("Slope of ST", 0, 2, 1)
        ca = st.slider("CA", 0, 3, 0)
        thal = st.slider("Thal", 0, 3, 2)

        submitted = st.form_submit_button("Predict")

        if submitted:
            inline_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                      thalach, exang, oldpeak, slope, ca, thal]])
            inline_pred = model.predict(scaler.transform(inline_input))[0]
            inline_prob = model.predict_proba(scaler.transform(inline_input))[0][1]
            st.write(f"**Prediction:** {'Heart Disease' if inline_pred == 1 else 'No Heart Disease'}")
            st.write(f"**Confidence:** {inline_prob:.2f}")

# ================================
# 🤖 FULL PREDICTION INTERFACE
# ================================
if page == "🤖 Full Prediction Interface":
    st.title("Full Heart Disease Prediction Interface")

    col1, col2 = st.columns(2)
    age = col1.slider("Age", 20, 80, 50)
    sex = col2.selectbox("Sex", [0, 1])
    cp = col1.slider("Chest Pain Type", 0, 3, 1)
    trestbps = col2.slider("Resting BP", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 600, 250)
    fbs = col2.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = col1.slider("Resting ECG", 0, 2, 1)
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = col1.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = col1.slider("Slope", 0, 2, 1)
    ca = col2.slider("CA", 0, 3, 0)
    thal = col1.slider("Thal", 0, 3, 2)

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(scaler.transform(user_input))[0]
    probability = model.predict_proba(scaler.transform(user_input))[0][1]

    st.subheader("Prediction Results")
    st.write("🧠 Risk Result:", "Heart Disease" if prediction == 1 else "No Heart Disease")
    st.write(f"Confidence Score: **{probability:.2f}**")

    st.subheader("AI Interpretation")
    if prediction == 1:
        st.warning("Based on these parameters, this patient may be at significant cardiovascular risk.")
    else:
        st.success("These values indicate low risk for heart disease.")