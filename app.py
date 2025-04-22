import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Dashboard & Prediction", layout="wide")

# Load and process dataset
df = pd.read_csv("heart.csv")
df["date_exam"] = pd.to_datetime(df["date_exam"])

# Sidebar nav
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["📊 Data Dashboard", "🤖 Prediction"])

# ======================
# 📊 DATA DASHBOARD
# ======================
if page == "📊 Data Dashboard":
    st.title("Heart Disease Dashboard with Analysis & AI Insights")

    # Filters
    st.sidebar.header("Filters")
    sex_filter = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])
    target_filter = st.sidebar.selectbox("Heart Disease", ["All", "Positive", "Negative"])
    cp_filter = st.sidebar.selectbox("Chest Pain Type (cp)", ["All"] + sorted(df["cp"].astype(str).unique().tolist()))
    date_range = st.sidebar.date_input("Exam Date Range", [df["date_exam"].min(), df["date_exam"].max()])

    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df["date_exam"] >= pd.to_datetime(date_range[0])) & (filtered_df["date_exam"] <= pd.to_datetime(date_range[1]))]

    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["sex"] == (1 if sex_filter == "Male" else 0)]
    if target_filter != "All":
        filtered_df = filtered_df[filtered_df["target"] == (1 if target_filter == "Positive" else 0)]
    if cp_filter != "All":
        filtered_df = filtered_df[filtered_df["cp"] == int(cp_filter)]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{filtered_df['age'].mean():.1f}")
    col2.metric("Mean Cholesterol", f"{filtered_df['chol'].mean():.1f} mg/dL")
    col3.metric("Heart Disease Rate", f"{filtered_df['target'].mean() * 100:.1f} %")

    # AI-style summary
    st.subheader("🧠 AI Insight")
    insights = []
    if filtered_df["age"].mean() > 55:
        insights.append("High average age indicates elevated risk.")
    if filtered_df["chol"].mean() > 250:
        insights.append("Average cholesterol is above healthy levels.")
    if filtered_df["sex"].mean() < 0.5:
        insights.append("More females in the selected dataset.")
    if filtered_df["target"].mean() > 0.6:
        insights.append("Majority of patients show signs of heart disease.")
    if not insights:
        st.success("Data appears within healthy range.")
    else:
        for insight in insights:
            st.warning(insight)

    # Visuals
    st.subheader("Distributions")
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        st.plotly_chart(px.histogram(filtered_df, x=col, color=filtered_df["target"].map({0: "Negative", 1: "Positive"}), nbins=30))

    st.subheader("Time Trends")
    df_time = filtered_df.copy()
    df_time["Month"] = df_time["date_exam"].dt.to_period("M").astype(str)
    st.plotly_chart(px.histogram(df_time, x="Month", color=df_time["target"].map({0: "Negative", 1: "Positive"}), barmode="group"))

    st.subheader("Chest Pain & Heart Disease")
    st.plotly_chart(px.bar(filtered_df, x="cp", color=filtered_df["target"].map({0: "Negative", 1: "Positive"}), barmode="group"))

    st.subheader("Download Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered CSV", data=csv, file_name="filtered.csv", mime="text/csv")

# ======================
# 🤖 PREDICTION INTERFACE
# ======================
if page == "🤖 Prediction":
    st.title("Heart Disease Prediction Tool")

    # Input form
    col1, col2 = st.columns(2)
    age = col1.slider("Age", 20, 80, 50)
    sex = col2.selectbox("Sex", [0, 1])
    cp = col1.slider("Chest Pain Type", 0, 3, 1)
    trestbps = col2.slider("Resting BP", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 600, 250)
    fbs = col2.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = col1.slider("Resting ECG", 0, 2, 1)
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = col1.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = col1.slider("Slope of the ST segment", 0, 2, 1)
    ca = col2.slider("Major Vessels (CA)", 0, 4, 0)
    thal = col1.slider("Thalassemia", 0, 3, 2)

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Model logic
    X = df.drop(columns=["target", "date_exam"])
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    st.write("🔴 Heart Disease" if prediction == 1 else "🟢 No Heart Disease")
    st.write(f"Predicted probability: **{prob:.2f}**")

    st.subheader("AI Insight")
    if prediction == 1:
        st.warning("Patient may be at risk. Recommend further cardiovascular evaluation.")
    else:
        st.success("Patient profile appears low risk based on input data.")