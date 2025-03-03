import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
import joblib
from tensorflow.keras.models import load_model



# LOADING THE SAVED MODEL
diabetes_model = pickle.load(open("C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/heart_disease_model.sav", 'rb'))
diabetes_scaler = pickle.load(open("C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/diabetes_scaler.sav", 'rb'))
heart_disease_scaler = pickle.load(open("C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/heart_disease_scaler.sav", 'rb'))




# SIDEBAR NAVIGATION
with st.sidebar:
    selected = option_menu(
        'Know Your Health',
        ['Home', 'General Disease Prediction', 'Heart Disease Prediction', 'Diabetes Prediction'],

        icons = ['file-medical', 'activity', 'heart'],
        default_index=0
    )

# LOAD THE GIF FOR FRONTPAGE
def get_base64_gif(file_path):
    with open(file_path, "rb") as file:
        encoded_gif = base64.b64encode(file.read()).decode("utf-8")
    return encoded_gif

gif_path = "C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/back.webp"
gif_base64 = get_base64_gif(gif_path)



# IF HOME PAGE IS SELECTED
if selected == "Home":
    st.markdown(
    f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        .stApp {{
            background: url("data:image/gif;base64,{gif_base64}") no-repeat center center fixed;
            height: 100vh;
        }}
        .right-align {{
            text-align: right;
            padding: 0;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown('<div class="right-align"><h4 style="margin-left:20rem; margin-right: -10rem; margin-top:5rem"><i>Discover your health insights in seconds with our AI-powered disease prediction tool</i></h4> <p style="margin-left:20rem; margin-right: -10rem; ">Leverage the power of AI to instantly analyze your symptoms and predict a range of diseases. Our tool offers accurate, data-driven insights, empowering you to make informed health decisions. Whether you\'re concerned about common illnesses or complex conditions, we provide quick predictions to help you take proactive steps toward better health.</p> <h3 style="margin-top:3rem; margin-right: -10rem"><b><i>Your First Step Toward Better Wellness!</i></b></h3></div>', unsafe_allow_html=True)




# General Disease Prediction

if selected == "General Disease Prediction":
    st.title("General Disease Prediction Page")

    # Load the trained model
    model_path = "C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/disease_prediction_model.pkl"
    encoder_path = "C:/Users/anjal/OneDrive/Desktop/Projects/ML Projects/models/symptom_encoder.pkl"

    try:
        model = joblib.load(model_path)  # Load trained model
        mlb = joblib.load(encoder_path)  # Load MultiLabelBinarizer
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        st.stop()

    # Get expected input feature size
    expected_input_size = model.n_features_in_

    # User input for symptoms
    user_input = st.text_input("Enter comma-separated symptoms:")

    if st.button("Predict"):
        symptoms = [sym.strip().lower() for sym in user_input.split(",") if sym.strip()]

        if not symptoms:
            st.warning("Please enter at least one symptom.")
        else:
            try:
                # Convert symptoms to one-hot encoding using the same encoder
                input_vector = mlb.transform([symptoms])  # Use the exact transformation
            except ValueError:
                st.error("Some symptoms are not recognized. Please check your input.")
                st.stop()

            # Ensure input matches expected shape
            if input_vector.shape[1] != expected_input_size:
                st.error(f"Feature mismatch! Expected {expected_input_size} features, but got {input_vector.shape[1]}.")
                st.stop()

            # Make prediction
            probabilities = model.predict_proba(input_vector)[0]
            diseases = model.classes_

            # Get top 3 predictions
            results = sorted(zip(diseases, probabilities), key=lambda x: x[1], reverse=True)[:3]

            # Display predictions
            st.success("Top 3 Predicted Diseases:")
            for disease, probability in results:
                st.write(f"**{disease}**: {probability * 100:.2f}%")
            st.markdown(
                "***:blue[⚠️ We always encourage our users to consult a healthcare professional for personalized advice. 😊]***")


# Function to convert input to float and handle errors
def convert_input_to_float(input_value, input_name):
    if input_value.strip() == "":
        st.error(f"Please enter a value for {input_name}")
        return None
    try:
        return float(input_value)
    except ValueError:
        st.error(f"Invalid input for {input_name}. Please enter a numeric value.")
        return None


# Diabetes prediction page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction Page")

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.text_input("Gender")

    with col2:
        AGE = st.text_input("Age")

    with col3:
        Urea = st.text_input("Urea")

    with col1:
        Cr = st.text_input("Creatinine ratio")

    with col2:
        HbA1c = st.text_input("Blood Sugar Level")

    with col3:
        Chol = st.text_input("Cholesterol")

    with col1:
        TG = st.text_input("Triglycerides")

    with col2:
        HDL = st.text_input("HDL")

    with col3:
        LDL = st.text_input("LDL")

    with col1:
        VLDL = st.text_input("VLDL")

    with col2:
        BMI = st.text_input("BMI Value")

    Gender = 1 if Gender.lower() == 'male' else 0

    # Initialize a flag to check if prediction is done
    prediction_done = False

    if st.button('Diabetes Test Result'):
        # Convert inputs to float
        AGE = convert_input_to_float(AGE, "Age")
        Urea = convert_input_to_float(Urea, "Urea")
        Cr = convert_input_to_float(Cr, "Cr")
        HbA1c = convert_input_to_float(HbA1c, "Blood Sugar Level")
        Chol = convert_input_to_float(Chol, "Chol")
        TG = convert_input_to_float(TG, "TG")
        HDL = convert_input_to_float(HDL, "HDL")
        LDL = convert_input_to_float(LDL, "LDL")
        VLDL = convert_input_to_float(VLDL, "VLDL")
        BMI = convert_input_to_float(BMI, "BMI Value")

        if None not in [AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]:
            # Prepare the input data
            input_data = np.array([[Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]])
            # Standardize the input data using the same scaler
            input_data_standardized = diabetes_scaler.transform(input_data)

            # Predict using the model
            diab_prediction = diabetes_model.predict(input_data_standardized)

            # Interpret the result
            if diab_prediction[0] == 'N':
                diab_diagnosis = 'The person is not diabetic'
            elif diab_prediction[0] == 'P':
                diab_diagnosis = 'The person is Pre-diabetic'
            elif diab_prediction[0] == 'Y':
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'Not able to predict'

            st.success(diab_diagnosis)
            prediction_done = True

            # Check for risk factors based on input values
            risk_factors = []
            if HbA1c is not None and HbA1c > 5.7:
                risk_factors.append("Blood Sugar level indicate a risk for diabetes.")
            if BMI is not None and BMI > 25:
                risk_factors.append("BMI indicates a risk of being overweight or obese.")
            if Chol is not None and Chol > 200:
                risk_factors.append("Cholesterol levels are high, indicating a risk for cardiovascular diseases.")

            # Display risk factors if any
            if risk_factors:
                st.warning("Risk Factors:")
                for factor in risk_factors:
                    st.write(f"- {factor}")

    # Show graphs and additional details only if prediction is done
    if prediction_done:
        st.subheader("Additional Details")

        if HbA1c is not None and HbA1c > 5.7:
            st.write("Maintain blood sugar level less than 5.7")
        if BMI is not None and BMI > 25:
            st.write("BMI should be less than 25, exercise and follow proper diet")
        if Chol is not None and Chol > 200:
            st.write("Eat healthy food and exercise to avoid high cholesterol level and keep it in check of range less than 200")
        st.write("Maintaining a balanced diet and regular exercise can help manage these risk factors.")
        st.markdown(
            "***:blue[⚠️ We always encourage our users to consult a healthcare professional for personalized advice. 😊]***")

# Heart Disease prediction page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction Page")
    
    col1, col2, col3 = st.columns(3)


    with col1:
        Gender = st.text_input("Gender")

    with col2:
        AGE = st.text_input("Age")

    with col3:
        CP= st.text_input("Chest Pain", help="**0: Asymptomatic**  \n**1: Atypical Angina**  \n**2: Non-Anginal Pain**  \n**3: Typical Angina**")

    with col1:
        trestbps = st.text_input("Resting Blood Pressure")

    with col2:
        Chol = st.text_input("Cholestoral ")
        
    with col3:
        fbs = st.text_input("Fasting Blood Sugar")

    with col1:
        restecg = st.text_input("Resting Electrocardiographic Results")

    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")

    with col3:
        exang = st.text_input("Exercise Induced Angina")

    with col1:
        oldpeak = st.text_input("Oldpeak")

    with col2:
        slope = st.text_input("Slope")
    
    with col3:
        ca = st.text_input("Number of Major Vessels (0-3) Colored by Flourosopy")
    with col1:
        thal = st.text_input("Thalassemia", help="**1 = Normal blood flow during a thal stress test**  \n**2 = Fixed Defect, there's no blood flow in a specific part of the heart during the stress test**  \n**3 = Reversible Defect, blood flow is observed, but it's not normal during the stress test**")

    Gender = 1 if Gender.lower() == 'male' else 0



    #code for prediction
    heart_diagnosis = ''

    # Initialize a flag to check if prediction is done
    prediction_done = False
    #creating a button for prdiction
    # Check if all inputs are valid
    if None not in [AGE, CP, trestbps, Chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
        # Code for prediction
        heart_diagnosis = ''
        if st.button('Heart Disease Test Result'):

            # Convert other inputs to float
            AGE = convert_input_to_float(AGE, "Age")
            CP = convert_input_to_float(CP, "Chest Pain")
            trestbps = convert_input_to_float(trestbps, "Resting Blood Pressure")
            Chol = convert_input_to_float(Chol, "Cholestoral ")
            fbs = convert_input_to_float(fbs, "Fasting Blood Sugar")
            restecg = convert_input_to_float(restecg, "Resting Electrocardiographic Results")
            thalach = convert_input_to_float(thalach, "Maximum Heart Rate Achieved")
            exang = convert_input_to_float(exang, "Exercise Induced Angina")
            oldpeak = convert_input_to_float(oldpeak, "Oldpeak")
            slope = convert_input_to_float(slope, "Slope")
            ca = convert_input_to_float(ca, "Number of Major Vessels (0-3) Colored by Flourosopy")
            thal = convert_input_to_float(thal, "Thalassemia ")
            # Prepare the input data
            heart_input_data = np.array([[AGE, Gender, CP, trestbps, Chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # # Standardize the input data using the same scaler
            heart_input_data_standardized = heart_disease_scaler.transform(heart_input_data)

    # Predict using the model
            heart_prediction = heart_disease_model.predict(heart_input_data_standardized)

            # Interpret the result
            if heart_prediction[0] == 0:
                heart_diagnosis = 'The person has a healthy heart.😊'
            elif heart_prediction[0] == 1:
                heart_diagnosis = 'The person is affected by heart disease ⚠️'
            else:
                heart_diagnosis = 'Not able to predict'

            st.success(heart_diagnosis)
            prediction_done = True
            if prediction_done:
                st.subheader("Additional Details")

                if Chol is not None and Chol > 200:
                    st.write(
                        "Eat healthy food and exercise to avoid high cholesterol level and keep it in check of range less than 200")
                st.write("Maintaining a balanced diet and regular exercise can help manage these risk factors.")
                st.markdown(
                    "***:blue[⚠️ We always encourage our users to consult a healthcare professional for personalized advice. 😊]***")


    else:
        st.warning("Please correct the input errors before proceeding.")
