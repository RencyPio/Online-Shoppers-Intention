import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.title("Online Shopping Purchase Prediction")
st.write("Predict whether a website visitor will make a purchase.")

# PAGE VISIT INFORMATION
Administrative = st.number_input("Administrative Pages", min_value=0, value=0)
Administrative_Duration = st.number_input("Administrative Duration", min_value=0.0, value=0.0)
Informational = st.number_input("Informational Pages", min_value=0, value=0)
Informational_Duration = st.number_input("Informational Duration", min_value=0.0, value=0.0)
ProductRelated = st.number_input("Product Related Pages", min_value=0, value=0)
ProductRelated_Duration = st.number_input("Product Related Duration", min_value=0.0, value=0.0)

# WEBSITE BEHAVIOR
BounceRates = st.slider("Bounce Rate", 0.0, 1.0, 0.05)
ExitRates = st.slider("Exit Rate", 0.0, 1.0, 0.07)
PageValues = st.number_input("Page Values", min_value=0.0, value=0.0)
SpecialDay = st.slider("Special Day", 0.0, 1.0, 0.0)

# MONTH DROPDOWN
month_options = {
    "Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5,
    "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}
month = st.selectbox("Month", list(month_options.keys()))
Month = month_options[month]

# DEVICE INFORMATION
OperatingSystems = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
Browser = st.selectbox("Browser", [1,2,3,4,5,6,7,8,9,10])
Region = st.selectbox("Region", [1,2,3,4,5,6,7,8,9])
TrafficType = st.selectbox("Traffic Type", [1,2,3,4,5,6,7,8,9,10])

# VISITOR TYPE DROPDOWN
visitor_options = {
    "New Visitor":0,
    "Other":1,
    "Returning Visitor":2}
visitor = st.selectbox("Visitor Type", list(visitor_options.keys()))
VisitorType = visitor_options[visitor]

# WEEKEND DROPDOWN
weekend_option = st.selectbox("Weekend", ["No","Yes"])
Weekend = 1 if weekend_option == "Yes" else 0

# FEATURE ENGINEERING
TotalDuration = Administrative_Duration + Informational_Duration + ProductRelated_Duration
TotalPages = Administrative + Informational + ProductRelated
BounceExitDiff = ExitRates - BounceRates
PageValuePerProduct = PageValues / (ProductRelated + 1)

# CREATE FEATURE ARRAY
features = np.array([[Administrative, Administrative_Duration,
                      Informational, Informational_Duration,
                      ProductRelated, ProductRelated_Duration,
                      BounceRates, ExitRates, PageValues,
                      SpecialDay, Month, OperatingSystems,
                      Browser, Region, TrafficType,
                      VisitorType, Weekend,
                      TotalDuration, TotalPages,
                      BounceExitDiff, PageValuePerProduct]])

# Convert to DataFrame for scaler compatibility
features = pd.DataFrame(features)
features = scaler.transform(features)

# PREDICTION BUTTON
if st.button("Predict Purchase"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Customer is likely to make a purchase.")
    else:
        st.error("Customer is unlikely to make a purchase.")