#streamlit app
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="datanai-yawarkhan/tourismpackageprediction-model", filename="best_tourismpackageprediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for tour operator staff that predicts whether customers are likely to buy a tourism package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to buy the package.")

# Collect user input
Age = st.number_input("Age", min_value=18, max_value=900, value=42)
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
NumberOfPersonVisiting = st.number_input("Number of persons visiting along with customer)", min_value=1, max_value=20, value=1)
PreferredPropertyStar = st.number_input("Ratings of hotels customer prefers", min_value=1, max_value=5,value=3)
NumberOfTrips = st.number_input("Avg trips customer takes annually", min_value=1, max_value=20, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children visiting along with customer", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=10000)
PitchSatisfactionScore = st.number_input("Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of followups", min_value=0, max_value=10, value=1)
DurationOfPitch = st.number_input("Duration of pitch", min_value=0, max_value=100, value=1)
TypeofContact = st.selectbox("How the customer was contacted", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business Owner", "Student", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation=st.selectbox("Job Designation",["Manager","Senior Manager","Executive","AVP","VP"])
ProductPitched=st.selectbox("Product Type Pitched",["Basic","Standard","Deluxe","Super Deluxe","King"])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 0 if Passport=="No" else 1,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "take" if prediction == 1 else "not take"
    st.write(f"Based on the information provided, the customer is likely to {result} the product/package.")
