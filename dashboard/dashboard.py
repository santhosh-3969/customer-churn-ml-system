import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Customer Churn ML System",
    page_icon="📊",
    layout="wide"
)

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Load trained model
model = pickle.load(open("model/churn_model.pkl","rb"))

# Sidebar
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 Data Insights",
        "🤖 Model Performance",
        "⭐ Feature Importance"
    ]
)

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.title("📊 Customer Churn Prediction System")

    st.write(
        "This Machine Learning application predicts whether a telecom customer "
        "is likely to churn using a **Random Forest model**."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Churn Customers", int(df["Churn"].sum()))
    col3.metric("Retention Rate", f"{(1-df['Churn'].mean())*100:.1f}%")

    st.write("---")

    st.subheader("Project Overview")

    st.write("""
    This system analyzes telecom customer behavior to predict churn risk.

    **Workflow**
    - Data preprocessing
    - Feature engineering
    - Machine learning model training
    - Model deployment
    - Prediction dashboard
    """)

# ---------------- PREDICTION ----------------

elif page == "🔍 Prediction":

    st.title("🔍 Customer Churn Prediction")

    st.write("Choose a prediction method.")

    tab1, tab2 = st.tabs(["Random Customer", "Manual Input"])

    # Random prediction
    with tab1:

        st.subheader("Predict Random Customer")

        if st.button("Predict Random Customer"):

            sample = X.sample(1)

            prediction = model.predict(sample)

            st.write("### Customer Data")
            st.dataframe(sample)

            if prediction[0] == 1:
                st.error("⚠ Customer likely to churn")
            else:
                st.success("✅ Customer likely to stay")

    # Manual input prediction
    with tab2:

        st.subheader("Enter Customer Details")

        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (Months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges", 0, 200, 70)

        with col2:
            total = st.number_input("Total Charges", 0, 10000, 2000)

        if st.button("Predict Customer"):

            sample = X.sample(1).copy()

            sample.iloc[0,0] = tenure
            sample.iloc[0,1] = monthly
            sample.iloc[0,2] = total

            prediction = model.predict(sample)

            if prediction[0] == 1:
                st.error("⚠ Customer likely to churn")
            else:
                st.success("✅ Customer likely to stay")

# ---------------- DATA INSIGHTS ----------------

elif page == "📊 Data Insights":

    st.title("📊 Customer Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("Dataset Statistics")
        st.write(df.describe())

    st.write("---")

    st.subheader("Churn Distribution")

    fig = px.histogram(
        df,
        x="Churn",
        color="Churn",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- MODEL PERFORMANCE ----------------

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model2 = RandomForestClassifier()

    model2.fit(X_train, y_train)

    pred = model2.predict(X_test)

    acc = accuracy_score(y_test, pred)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    st.write("---")

    st.write("""
    The Random Forest algorithm combines multiple decision trees
    to produce more accurate predictions and reduce overfitting.
    """)

# ---------------- FEATURE IMPORTANCE ----------------

elif page == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    st.write("These features have the highest impact on churn prediction.")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Affecting Customer Churn"
    )


    st.plotly_chart(fig, use_container_width=True)
