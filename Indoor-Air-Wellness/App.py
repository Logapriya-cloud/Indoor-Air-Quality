import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Title and description
st.title("🏠 Indoor Air Wellness - Live Prediction Dashboard")
st.markdown("Monitor and predict indoor air quality based on live sensor data.")

# 2. Load dataset
file_path = r"C:\Users\kaviy\OneDrive\iaq_live_dataset_500.csv"
  # Best and cleanest for Windows
  # Update this path if needed

try:
    data = pd.read_csv(file_path)
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please check the file path.")
    st.stop()

# 3. Preview dataset
if st.checkbox("Show Raw Data"):
    st.dataframe(data.head())

# 4. Preprocess
if 'AQI_Category' not in data.columns:
    st.warning("🟡 Dataset must include an 'AQI_Category' column as target.")
    st.stop()

X = data.drop(columns=['AQI_Category'])
y = data['AQI_Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

st.markdown(f"### ✅ Model Accuracy: {acc:.2f}")

# 6. Predict on custom input
st.subheader("🔍 Predict Air Quality for New Input")

# Get feature columns dynamically
input_data = {}
for col in X.columns:
    input_data[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

st.success(f"🏷 *Predicted Air Quality Category:* {prediction}")

# 7. Optional: Classification report
if st.checkbox("Show Classification Report"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)