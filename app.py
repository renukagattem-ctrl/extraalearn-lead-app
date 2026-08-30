import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
def load_model():
    return joblib.load("churn_prediction_model_v1_0.joblib")

model = load_model()

st.set_page_config(page_title="ExtraaLearn Lead Predictor", layout="centered")
st.title("ExtraaLearn Lead Conversion Prediction App")
st.write("Enter lead engagement attributes to calculate real-time conversion probability.")

# Input fields[cite: 6]
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=75, value=35)
    current_occupation = st.selectbox("Current Occupation", ["Professional", "Unemployed", "Student"])
    first_interaction = st.selectbox("First Interaction", ["Website", "Mobile App"])
    profile_completed = st.selectbox("Profile Completed", ["Low", "Medium", "High"])
    last_activity = st.selectbox("Last Activity", ["Website Activity", "Phone Activity", "Email Activity"])

with col2:
    website_visits = st.number_input("Website Visits", min_value=0, max_value=50, value=4)
    time_spent_on_website = st.number_input("Time Spent on Website (seconds)", min_value=0, max_value=5000, value=650)
    page_views_per_visit = st.number_input("Page Views Per Visit", min_value=0.0, max_value=25.0, value=3.2, step=0.1)
    referral = st.selectbox("Heard via Referral?", ["No", "Yes"])
    educational_channels = st.selectbox("Heard from Educational Channels?", ["No", "Yes"])

col3, col4, col5 = st.columns(3)
with col3:
    digital_media = st.selectbox("Digital Media Ad?", ["No", "Yes"])
with col4:
    print_media_type1 = st.selectbox("Newspaper Ad?", ["No", "Yes"])
with col5:
    print_media_type2 = st.selectbox("Magazine Ad?", ["No", "Yes"])

if st.button("Predict Lead Conversion", type="primary"):
    input_data = pd.DataFrame([{
        "age": age,
        "current_occupation": current_occupation,
        "first_interaction": first_interaction,
        "profile_completed": profile_completed,
        "website_visits": website_visits,
        "time_spent_on_website": time_spent_on_website,
        "page_views_per_visit": page_views_per_visit,
        "last_activity": last_activity,
        "print_media_type1": print_media_type1,
        "print_media_type2": print_media_type2,
        "digital_media": digital_media,
        "educational_channels": educational_channels,
        "referral": referral
    }])
    
    # Classification threshold of 0.45[cite: 6]
    classification_threshold = 0.45
    prob = model.predict_proba(input_data)[0, 1]
    pred = 1 if prob >= classification_threshold else 0
    result = "convert" if pred == 1 else "not convert"
    
    st.divider()
    if pred == 1:
        st.success(f"**Prediction: The lead is likely to {result}.**")
    else:
        st.warning(f"**Prediction: The lead is likely to {result}.**")
    st.write(f"Estimated Conversion Probability: **{prob:.2f}**")
