import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Heart Disease App", layout="wide")

# Load dataset
df = pd.read_csv("heart.csv")
df["date_exam"] = pd.to_datetime(df["date_exam"])

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["Data Dashboard", "Heart Disease Prediction"])

# ===============================
# DASHBOARD SECTION
# ===============================
if page == "Data Dashboard":
    st.title("Heart Disease Data Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")
    sex_filter = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])
    target_filter = st.sidebar.selectbox("Heart Disease", ["All", "Positive", "Negative"])
    date_range = st.sidebar.date_input("Exam Date Range", [df["date_exam"].min(), df["date_exam"].max()])

    filtered_df = df[(df["date_exam"] >= pd.to_datetime(date_range[0])) & (df["date_exam"] <= pd.to_datetime(date_range[1]))]
    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["sex"] == (1 if sex_filter == "Male" else 0)]
    if target_filter != "All":
        filtered_df = filtered_df[filtered_df["target"] == (1 if target_filter == "Positive" else 0)]

    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{filtered_df['age'].mean():.1f} years")
    col2.metric("Cholesterol Mean", f"{filtered_df['chol'].mean():.1f} mg/dL")
    col3.metric("Heart Disease Prevalence", f"{filtered_df['target'].mean()*100:.1f} %")

    st.subheader("Affirmations & Observations")
    avg_age = filtered_df["age"].mean()
    avg_chol = filtered_df["chol"].mean()
    female_ratio = filtered_df[filtered_df["sex"] == 0].shape[0] / len(filtered_df) if len(filtered_df) > 0 else 0

    if avg_age > 55:
        st.info("Average age is above 55, indicating higher risk in the selected group.")
    if avg_chol > 250:
        st.warning("High average cholesterol level. Consider cardiovascular alert thresholds.")
    if female_ratio > 0.5:
        st.markdown("More than 50% of patients are female in this filtered data.")

    # Histogram grid for univariate analysis
    st.subheader("Distributions of Key Features")
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    for i, col in enumerate(num_cols):
        sns.histplot(data=filtered_df, x=col, hue="target", multiple="stack", ax=axes[i//3, i%3])
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = filtered_df.drop(columns=["date_exam"]).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # Time series
    st.subheader("Heart Disease Cases Over Time")
    time_series = filtered_df.copy()
    time_series["month"] = time_series["date_exam"].dt.to_period("M").astype(str)
    fig_time = px.histogram(time_series, x="month", color=time_series["target"].map({0:"Negative",1:"Positive"}))
    st.plotly_chart(fig_time, use_container_width=True)

    # Pie charts and scatter
    st.subheader("Category Distributions")
    st.plotly_chart(px.pie(filtered_df, names=filtered_df["cp"].astype(str), title="Chest Pain Types"))
    st.plotly_chart(px.pie(filtered_df, names=filtered_df["sex"].map({0: "Female", 1: "Male"}), title="Gender Distribution"))
    st.plotly_chart(px.scatter(filtered_df, x="age", y="chol", color=filtered_df["target"].map({0:"Negative", 1:"Positive"}), title="Age vs Cholesterol"))

    # Data table and download
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data", data=csv, file_name="filtered_data.csv", mime="text/csv")

# ===============================
# PREDICTION SECTION
# ===============================
elif page == "Heart Disease Prediction":
    st.title("Heart Disease Prediction Tool")

    col1, col2 = st.columns(2)
    age = col1.slider("Age", 20, 80, 50)
    sex = col2.selectbox("Sex", [0, 1])
    cp = col1.slider("Chest Pain Type", 0, 3, 1)
    trestbps = col2.slider("Resting BP", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 400, 240)
    fbs = col2.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = col1.slider("Resting ECG", 0, 2, 1)
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = col1.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = col1.slider("Slope", 0, 2, 1)
    ca = col2.slider("CA", 0, 3, 0)
    thal = col1.slider("Thal", 0, 3, 2)

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    X = df.drop(columns=["target", "date_exam"])
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction")
    st.write("Result: Positive" if prediction == 1 else "Result: Negative")