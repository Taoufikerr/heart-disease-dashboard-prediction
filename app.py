import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

# --- Train Model ---
@st.cache_resource
def train_model(df):
    cols_to_drop = [col for col in ["target", "date"] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["target"]

    # Encodage des variables catégorielles
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler, X.columns  # on retourne aussi les colonnes d'entrée

# --- Mappings ---
cp_options = {"Typical Angina": "typical angina", "Atypical Angina": "atypical angina",
              "Non-Anginal Pain": "non-anginal", "Asymptomatic": "asymptomatic"}
fbs_options = {"FBS ≤ 120": False, "FBS > 120": True}
restecg_options = {"Normal": "normal", "ST-T abnormality": "st-t abnormality",
                   "Left ventricular hypertrophy": "lv hypertrophy"}
exang_options = {"No": False, "Yes": True}
slope_options = {"Upsloping": "upsloping", "Flat": "flat", "Downsloping": "downsloping"}
thal_options = {"Normal": "normal", "Fixed Defect": "fixed defect", "Reversible Defect": "reversable defect"}

# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["📊 Dashboard", "🤖 Full Prediction Interface"])

# --- Load Data & Model ---
df = load_data()
model, scaler, model_columns = train_model(df)

# --- DASHBOARD ---
if page == "📊 Dashboard":
    st.title("Heart Disease Dashboard")
    st.caption(f"Available exam dates: {df['date'].min().date()} to {df['date'].max().date()}")

    # Sidebar filters
    sex_label = st.sidebar.selectbox("Sex", ["Male", "Female"])
    sex_filter = sex_label
    target_filter = st.sidebar.selectbox("Heart Disease", ["All", "Positive", "Negative"])
    cp_filter = st.sidebar.selectbox("Chest Pain Type", ["All"] + list(cp_options.keys()))
    date_range = st.sidebar.date_input("Exam Date Range", [df["date"].min(), df["date"].max()])

    # Filter data
    filtered_df = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["date"] <= pd.to_datetime(date_range[1]))
        ]

    filtered_df = filtered_df[filtered_df["sex"] == sex_filter]
    if target_filter != "All":
        filtered_df = filtered_df[filtered_df["target"] == (1 if target_filter == "Positive" else 0)]
    if cp_filter != "All":
        filtered_df = filtered_df[filtered_df["cp"] == cp_options[cp_filter]]

    st.write(f"Filtered rows: {len(filtered_df)}")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", f"{filtered_df['age'].mean():.1f}")
    col2.metric("Avg Cholesterol", f"{filtered_df['chol'].mean():.1f} mg/dL")
    col3.metric("Heart Disease %", f"{filtered_df['target'].mean() * 100:.1f} %")

    # Insights
    st.subheader("AI-Style Insights")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        insights = []
        if filtered_df["age"].mean() > 55:
            insights.append("High average age — elevated risk.")
        if filtered_df["chol"].mean() > 240:
            insights.append("Cholesterol levels above normal.")
        if filtered_df["target"].mean() > 0.6:
            insights.append("High proportion of heart disease detected.")
        for tip in insights:
            st.markdown(f"- {tip}")

    # Histograms
    st.subheader("Distributions")
    if not filtered_df.empty:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        sns.histplot(filtered_df, x="age", hue=filtered_df["target"], ax=axs[0][0])
        sns.histplot(filtered_df, x="chol", hue=filtered_df["target"], ax=axs[0][1])
        sns.histplot(filtered_df, x="trestbps", hue=filtered_df["target"], ax=axs[0][2])
        sns.histplot(filtered_df, x="thalch", hue=filtered_df["target"], ax=axs[1][0])
        sns.histplot(filtered_df, x="oldpeak", hue=filtered_df["target"], ax=axs[1][1])
        axs[1][2].axis("off")
        st.pyplot(fig)
    else:
        st.warning("No data to show for charts.")

# --- FULL PREDICTION ---
if page == "🤖 Full Prediction Interface":
    st.title("Full Prediction Form")
    st.markdown("Enter all features to get a model-based prediction.")
    col1, col2 = st.columns(2)

    age = col1.slider("Age", 20, 80, 50)
    sex = col2.selectbox("Sex", ["Male", "Female"])
    cp = cp_options[col1.selectbox("Chest Pain Type", list(cp_options.keys()))]
    trestbps = col2.slider("Resting BP", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 600, 250)
    fbs = fbs_options[col2.selectbox("Fasting Blood Sugar", list(fbs_options.keys()))]
    restecg = restecg_options[col1.selectbox("Resting ECG", list(restecg_options.keys()))]
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = exang_options[col1.selectbox("Exercise Angina", list(exang_options.keys()))]
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = slope_options[col1.selectbox("Slope of ST", list(slope_options.keys()))]
    ca = col2.slider("CA", 0, 3, 0)
    thal = thal_options[col1.selectbox("Thal", list(thal_options.keys()))]

    # Prepare input
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalch": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write("Heart Disease Risk:", "Yes" if prediction == 1 else "No")
    st.write(f"Confidence Score: {prob:.2f}")
