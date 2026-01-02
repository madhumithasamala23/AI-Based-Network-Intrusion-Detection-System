# AI-Based Network Intrusion Detection System

This project implements an AI-powered Network Intrusion Detection System
using Machine Learning techniques.

## Technologies Used
- Python
- Random Forest Classifier
- Streamlit
- Pandas, NumPy
- Scikit-learn

## Project Description
The system detects whether network traffic is normal or malicious.
A Random Forest model is trained on simulated network data and
used to classify incoming traffic in real time.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python -m streamlit run nids_main.py

## Output
The system displays:
- Model accuracy
- Live traffic detection (Normal / Intrusion)

