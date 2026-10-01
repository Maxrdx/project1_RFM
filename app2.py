import streamlit as st
import joblib
import numpy as np

# 1. Load the saved Random Forest model
# Note: We do not need a scaler here because Random Forests do not require feature scaling!
model = joblib.load('rfm_churn_model.pkl')

# 2. Build the visual interface
st.title("E-Commerce RFM & Churn Predictor")
st.write("Enter a customer's purchasing behavior to predict churn risk and generate a targeted marketing strategy.")

# Create input boxes side-by-side
col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input("Recency (Days since last order)", min_value=0, value=30)
with col2:
    frequency = st.number_input("Frequency (Total orders)", min_value=1, value=5)
with col3:
    monetary = st.number_input("Monetary Value (£)", min_value=0.0, value=600.0)

# 3. Predict and recommend strategy when the button is clicked
if st.button("Analyze Customer Risk"):
    # Format input for the model
    customer_data = np.array([[recency, frequency, monetary]])
    
    # Predict probability of class '1' (Churn)
    churn_prob = model.predict_proba(customer_data)[0][1]
    
    st.subheader(f"Churn Probability: {churn_prob * 100:.1f}%")
    
    # Business Logic Mapping (Using £500 as an example high-value threshold)
    high_value = monetary > 500 
    high_risk = churn_prob >= 0.50
    
    if high_value and not high_risk:
        st.success("🟢 Action: Highest Segment (Give VIP Early Access)")
    elif high_value and high_risk:
        st.error("🔴 Action: High Value At-Risk (Send 20% Retention Offer)")
    elif not high_value and high_risk:
        st.warning("🟠 Action: Low Value At-Risk (Let Churn - Save Budget)")
    else:
        st.info("🔵 Action: Regulars (Standard Nurture Emails)")