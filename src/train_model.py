import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import os
print("Current folder:", os.getcwd())
print("Files in data folder:", os.listdir("data"))
# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df = df.dropna()

# Remove ID column
df.drop("customerID", axis=1, inplace=True)

# Convert target
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/churn_model.pkl","wb"))
print("Model trained and saved successfully")