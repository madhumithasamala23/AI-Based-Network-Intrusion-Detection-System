import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="AI-Based Network Intrusion Detection System",
                   layout="wide")

st.title("AI-Based Network Intrusion Detection System (NIDS)")
st.write("This system uses Machine Learning to detect malicious network activity.")

# -----------------------------
# Data Simulation
# -----------------------------
def load_data():
    np.random.seed(42)
    data = {
        "duration": np.random.randint(0, 1000, 500),
        "protocol_type": np.random.randint(0, 3, 500),
        "src_bytes": np.random.randint(0, 100000, 500),
        "dst_bytes": np.random.randint(0, 100000, 500),
        "flag": np.random.randint(0, 6, 500),
        "label": np.random.randint(0, 2, 500)  # 0 = Normal, 1 = Attack
    }
    return pd.DataFrame(data)

df = load_data()

st.subheader("Sample Network Data")
st.dataframe(df.head())

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

if st.sidebar.button("Train Model Now"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success("Model trained successfully!")
    st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

# -----------------------------
# Live Traffic Simulation
# -----------------------------
st.subheader("Live Traffic Simulator")

duration = st.number_input("Duration", min_value=0, max_value=1000, value=100)
protocol_type = st.number_input("Protocol Type (0-TCP, 1-UDP, 2-ICMP)", min_value=0, max_value=2, value=1)
src_bytes = st.number_input("Source Bytes", min_value=0, max_value=100000, value=5000)
dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=100000, value=3000)
flag = st.number_input("Flag", min_value=0, max_value=5, value=2)

if st.button("Detect Intrusion"):
    input_data = np.array([[duration, protocol_type, src_bytes, dst_bytes, flag]])
    model.fit(X_train, y_train)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Intrusion Detected! (Malicious Traffic)")
    else:
        st.success("✅ Normal Network Traffic")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Developed as an Academic Project | AI-Based Network Intrusion Detection System")
