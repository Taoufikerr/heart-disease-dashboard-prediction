# Heart Disease Dashboard & Prediction

This project is an interactive web application built with Streamlit that allows users to explore heart disease data, apply dynamic filters, and make real-time predictions using a logistic regression model.

It combines data visualization, machine learning, and user-friendly design in a single unified dashboard.

---

## 🚀 Live App  
Access the full app here:  
https://heart-disease-dashboard-prediction-cgfvsyybaofcjqvjo4jgzu.streamlit.app/

---

## 🔍 Features

### 📊 Dashboard
- Filters by sex, chest pain type, and exam date
- Key performance indicators (age, cholesterol, heart disease rate)
- AI-style health insights based on filtered data
- Multiple visualizations (histograms, boxplots, pie charts, bar graphs)
- Exportable filtered data

### 🤖 Prediction Interface
- Inline prediction form inside the dashboard
- Full-form prediction interface in a separate section
- Real-time model results using logistic regression
- Confidence score with each prediction

### ⚙️ Technical Highlights
- Streamlit for UI and interactivity
- Scikit-learn logistic regression model
- Caching for model and data (fast performance)
- Clean layout, readable labels, user-friendly inputs
- Easily extendable to deep learning or multi-model setup

---

## 📂 Project Structure

```
heart-disease-dashboard-prediction/
├── app.py                               # Streamlit app: dashboard + prediction logic
├── heart.csv                            # Dataset with added date_exam column
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── Heart_Disease_Project_TaoufikErrajraji_PRO.ipynb  # Google Colab notebook (exploration + modeling)
```

---

## 📓 Notebook Version

The project also includes a Jupyter/Colab notebook that demonstrates:
- Dataset exploration
- Univariate & bivariate analysis
- Model training and evaluation
- Data visualization with Matplotlib and Seaborn

This file is useful for understanding the logic and experimentation behind the final Streamlit dashboard.

---

## 💻 Run Locally

1. Clone the repository:
    git clone https://github.com/Taoufikerr/heart-disease-dashboard-prediction
    cd heart-disease-dashboard-prediction

2. Install dependencies:
    pip install -r requirements.txt

3. Launch the app:
    streamlit run app.py

---

## 👨‍💻 Author

**Taoufik Errajraji**  
GitHub: https://github.com/Taoufikerr  
LinkedIn: https://www.linkedin.com/in/taoufik-errajraji13/

---

