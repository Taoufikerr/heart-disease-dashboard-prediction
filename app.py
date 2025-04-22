import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
df = pd.read_csv("heart.csv")
df["date_exam"] = pd.to_datetime(df["date_exam"])

st.set_page_config(page_title="Enhanced Heart Dashboard", layout="wide")
st.title("📊 Advanced Heart Disease Dashboard with AI Insights")

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

# AI Insights Section
st.subheader("🧠 AI-Style Insights (Dynamic Recommendations)")
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    insight_lines = []
    avg_age = filtered_df["age"].mean()
    avg_chol = filtered_df["chol"].mean()
    risk_pct = filtered_df["target"].mean() * 100
    cp_mode = filtered_df["cp"].mode()[0] if not filtered_df["cp"].mode().empty else "N/A"

    if avg_age > 55:
        insight_lines.append("High average age — recommend screening in older patients.")
    if avg_chol > 240:
        insight_lines.append("Elevated cholesterol levels — potential dietary/lifestyle risk.")
    if cp_mode in [1, 2]:
        insight_lines.append(f"Common chest pain type: {cp_mode} — typically linked to angina.")
    if risk_pct > 60:
        insight_lines.append("Majority of this group shows signs of heart disease.")

    for line in insight_lines:
        st.markdown(f"🔎 {line}")

# Grid of Histograms
st.subheader("📊 Distributions of Key Variables")
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
sns.histplot(data=filtered_df, x="age", hue="target", ax=axs[0][0], multiple="stack")
sns.histplot(data=filtered_df, x="chol", hue="target", ax=axs[0][1], multiple="stack")
sns.histplot(data=filtered_df, x="trestbps", hue="target", ax=axs[0][2], multiple="stack")
sns.histplot(data=filtered_df, x="thalach", hue="target", ax=axs[1][0], multiple="stack")
sns.histplot(data=filtered_df, x="oldpeak", hue="target", ax=axs[1][1], multiple="stack")
axs[1][2].axis("off")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("📌 Correlation Heatmap")
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(filtered_df.drop(columns=["date_exam"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig_corr)

# Monthly Trend
st.subheader("📅 Monthly Trends by Heart Disease")
filtered_df["Month"] = filtered_df["date_exam"].dt.to_period("M").astype(str)
st.plotly_chart(px.histogram(filtered_df, x="Month", color=filtered_df["target"].map({0: "Negative", 1: "Positive"}), barmode="group"))

# Box Plot Comparison
st.subheader("📦 Heart Rate vs Disease Outcome")
st.plotly_chart(px.box(filtered_df, x="target", y="thalach", color=filtered_df["target"].map({0:"Negative", 1:"Positive"}), labels={"target": "Heart Disease", "thalach": "Max Heart Rate"}))

# Download
st.subheader("📥 Download Filtered Dataset")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="filtered_heart_data.csv", mime="text/csv")