import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("heart.csv")
df["date_exam"] = pd.to_datetime(df["date_exam"])

st.set_page_config(page_title="Heart Dashboard + Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["📊 Dashboard", "🤖 Full Prediction Interface"])

# Train model
X = df.drop(columns=["target", "date_exam"])
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# Mappings
cp_options = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
fbs_options = {"FBS ≤ 120": 0, "FBS > 120": 1}
restecg_options = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}
exang_options = {"No": 0, "Yes": 1}
slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_options = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2, "Unknown": 3}

# DASHBOARD
if page == "📊 Dashboard":
    st.title("Heart Disease Dashboard + Inline Prediction")
    st.caption(f"Available exam dates: {df['date_exam'].min().date()} to {df['date_exam'].max().date()}")

    sex_label = st.sidebar.selectbox("Sex", ["Male", "Female"])
    sex_filter = 1 if sex_label == "Male" else 0
    target_filter = st.sidebar.selectbox("Heart Disease", ["All", "Positive", "Negative"])
    cp_filter = st.sidebar.selectbox("Chest Pain Type", ["All"] + list(cp_options.keys()))
    date_range = st.sidebar.date_input("Exam Date Range", [df["date_exam"].min(), df["date_exam"].max()])

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered_df = df[
            (df["date_exam"] >= pd.to_datetime(date_range[0])) &
            (df["date_exam"] <= pd.to_datetime(date_range[1]))
        ]
    else:
        st.warning("Please select a valid date range (start and end).")
        filtered_df = df.iloc[0:0]

    filtered_df = filtered_df[filtered_df["sex"] == sex_filter]
    if target_filter != "All":
        filtered_df = filtered_df[filtered_df["target"] == (1 if target_filter == "Positive" else 0)]
    if cp_filter != "All":
        filtered_df = filtered_df[filtered_df["cp"] == cp_options[cp_filter]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{filtered_df['age'].mean():.1f}")
    col2.metric("Avg Cholesterol", f"{filtered_df['chol'].mean():.1f} mg/dL")
    col3.metric("Heart Disease %", f"{filtered_df['target'].mean() * 100:.1f} %")

    st.subheader("🧠 AI-Style Insights")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        if filtered_df["age"].mean() > 55:
            st.info("High average age — elevated risk.")
        if filtered_df["chol"].mean() > 240:
            st.warning("Cholesterol is above healthy threshold.")
        if filtered_df["target"].mean() > 0.6:
            st.warning("High proportion of heart disease detected.")

    st.subheader("📊 Distributions")
    if not filtered_df.empty:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        sns.histplot(data=filtered_df, x="age", hue=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), ax=axs[0][0], multiple="stack")
        sns.histplot(data=filtered_df, x="chol", hue=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), ax=axs[0][1], multiple="stack")
        sns.histplot(data=filtered_df, x="trestbps", hue=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), ax=axs[0][2], multiple="stack")
        sns.histplot(data=filtered_df, x="thalach", hue=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), ax=axs[1][0], multiple="stack")
        sns.histplot(data=filtered_df, x="oldpeak", hue=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), ax=axs[1][1], multiple="stack")
        axs[1][2].axis("off")
        st.pyplot(fig)
    else:
        st.warning("No data available to generate histograms.")

    st.subheader("🧩 Additional Visuals")
    st.plotly_chart(px.histogram(filtered_df, x="ca", color=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), barmode="group"))
    st.plotly_chart(px.box(filtered_df, x="target", y="chol", color=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"})))
    st.plotly_chart(px.bar(filtered_df, x="restecg", color=filtered_df["target"].map({0: "No Heart Disease", 1: "Heart Disease"}), barmode="group"))
    st.plotly_chart(px.pie(filtered_df, names=filtered_df["fbs"].map({0:"FBS ≤ 120", 1:"FBS > 120"})))

    st.subheader("🔍 Inline Prediction")
    with st.form("inline_form"):
        age = st.slider("Age", 20, 80, 50)
        sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
        cp = cp_options[st.selectbox("Chest Pain Type", list(cp_options.keys()))]
        trestbps = st.slider("Resting BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 250)
        fbs = fbs_options[st.selectbox("Fasting Blood Sugar", list(fbs_options.keys()))]
        restecg = restecg_options[st.selectbox("Resting ECG", list(restecg_options.keys()))]
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = exang_options[st.selectbox("Exercise Induced Angina", list(exang_options.keys()))]
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = slope_options[st.selectbox("Slope of ST", list(slope_options.keys()))]
        ca = st.slider("CA", 0, 3, 0)
        thal = thal_options[st.selectbox("Thal", list(thal_options.keys()))]
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]])
            pred = model.predict(scaler.transform(input_arr))[0]
            prob = model.predict_proba(scaler.transform(input_arr))[0][1]
            st.write(f"**Prediction:** {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
            st.write(f"**Confidence:** {prob:.2f}")

# FULL PREDICTION INTERFACE
if page == "🤖 Full Prediction Interface":
    st.title("Full Heart Disease Prediction Interface")

    col1, col2 = st.columns(2)
    age = col1.slider("Age", 20, 80, 50)
    sex = 1 if col2.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
    cp = cp_options[col1.selectbox("Chest Pain Type", list(cp_options.keys()))]
    trestbps = col2.slider("Resting BP", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 600, 250)
    fbs = fbs_options[col2.selectbox("Fasting Blood Sugar", list(fbs_options.keys()))]
    restecg = restecg_options[col1.selectbox("Resting ECG", list(restecg_options.keys()))]
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = exang_options[col1.selectbox("Exercise Induced Angina", list(exang_options.keys()))]
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = slope_options[col1.selectbox("Slope", list(slope_options.keys()))]
    ca = col2.slider("CA", 0, 3, 0)
    thal = thal_options[col1.selectbox("Thal", list(thal_options.keys()))]

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(scaler.transform(user_input))[0]
    probability = model.predict_proba(scaler.transform(user_input))[0][1]

    st.subheader("Prediction Results")
    st.write("🧠 Risk Result:", "Heart Disease" if prediction == 1 else "No Heart Disease")
    st.write(f"Confidence Score: **{probability:.2f}**")

    st.subheader("AI Interpretation")
    if prediction == 1:
        st.warning("Patient may be at cardiovascular risk. Further testing advised.")
    else:
        st.success("These values indicate low risk for heart disease.")
""")
