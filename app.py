import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Model and Files
# ==========================================

model = joblib.load("heart_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================================
# Streamlit Page
# ==========================================

st.set_page_config(
    page_title="Cardiovascular Risk Assessment",
    page_icon="❤️",
    layout="wide"
)
st.sidebar.title("❤️ About")

st.sidebar.info(
    """
    **Cardiovascular Risk Assessment**

    Machine Learning Model:
    - Random Forest Classifier

    Dataset Size:
    - 50,000 Patients
    
     Developed using:
    - Python
    - Streamlit
    - Scikit-Learn
    """
)
st.title("❤️ Cardiovascular Risk Assessment Dashboard")
st.write("Enter the patient's details below to predict heart disease risk.")

st.markdown("---")

# ==========================================
# Input Fields
# ==========================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 1, 120, 45)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    weight = st.number_input(
        "Weight (kg)",
        20,
        200,
        70
    )

    height = st.number_input(
        "Height (cm)",
        100,
        250,
        170
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        50.0,
        24.5
    )

    smoking = st.selectbox(
        "Smoking",
        ["Never", "Former", "Current"]
    )

    # Dataset values
    alcohol = st.selectbox(
        "Alcohol Intake",
        ["Low", "Moderate", "High"]
    )

    # Dataset values
    physical = st.selectbox(
        "Physical Activity",
        ["Sedentary", "Moderate", "Active"]
    )

    # Dataset values
    diet = st.selectbox(
        "Diet",
        ["Healthy", "Average", "Unhealthy"]
    )

    stress = st.slider(
        "Stress Level",
        1,
        10,
        5
    )

with col2:

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )

    hyperlipidemia = st.selectbox(
        "Hyperlipidemia",
        ["No", "Yes"]
    )

    family = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )

    previous = st.selectbox(
        "Previous Heart Attack",
        ["No", "Yes"]
    )

    systolic = st.number_input(
        "Systolic BP",
        70,
        250,
        120
    )

    diastolic = st.number_input(
        "Diastolic BP",
        40,
        150,
        80
    )

    heart_rate = st.number_input(
        "Heart Rate",
        40,
        180,
        72
    )

    sugar = st.number_input(
        "Blood Sugar Fasting",
        50,
        300,
        100
    )

    cholesterol = st.number_input(
        "Total Cholesterol",
        100,
        400,
        180
    )

# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Predict Risk"):

    # Encode only categorical columns
    gender = encoders["Gender"].transform([gender])[0]
    smoking = encoders["Smoking"].transform([smoking])[0]
    alcohol = encoders["Alcohol_Intake"].transform([alcohol])[0]
    physical = encoders["Physical_Activity"].transform([physical])[0]
    diet = encoders["Diet"].transform([diet])[0]

    # Convert Yes/No to numeric
    hypertension = 1 if hypertension == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    hyperlipidemia = 1 if hyperlipidemia == "Yes" else 0
    family = 1 if family == "Yes" else 0
    previous = 1 if previous == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Smoking": smoking,
        "Alcohol_Intake": alcohol,
        "Physical_Activity": physical,
        "Diet": diet,
        "Stress_Level": stress,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Hyperlipidemia": hyperlipidemia,
        "Family_History": family,
        "Previous_Heart_Attack": previous,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Heart_Rate": heart_rate,
        "Blood_Sugar_Fasting": sugar,
        "Cholesterol_Total": cholesterol
    }])

    # Match training column order
    input_data = input_data[feature_names]

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.subheader("📋 Patient Details")
    st.write(input_data)
    probability = model.predict_proba(input_data)[0]

    # Risk percentage
    risk = probability[1] * 100

    st.markdown("---")

    st.subheader("📊 Risk Score")
    st.progress(int(risk))
    st.write(f"Predicted Risk Score: **{risk:.2f}%**")

    if prediction == 1:

        st.error("🔴 High Risk of Heart Disease")
        st.metric("Confidence", f"{probability[1]*100:.2f}%")

        st.subheader("💡 Recommendations")

        st.write("• Consult a cardiologist for further evaluation.")
        st.write("• Maintain a healthy and balanced diet.")
        st.write("• Exercise regularly (at least 30 minutes daily).")
        st.write("• Avoid smoking and excessive alcohol consumption.")
        st.write("• Monitor blood pressure, cholesterol, and blood sugar regularly.")
        st.write("• Reduce stress through yoga, meditation, or adequate sleep.")

    else:

        st.success("🟢 Low Risk of Heart Disease")
        st.metric("Confidence", f"{probability[0]*100:.2f}%")

        st.subheader("✅ Recommendations")

        st.write("• Continue maintaining a healthy lifestyle.")
        st.write("• Exercise regularly.")
        st.write("• Eat a balanced diet rich in fruits and vegetables.")
        st.write("• Schedule regular health checkups.")