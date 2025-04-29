import streamlit as st
import sqlite3
import hashlib
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu
# Register function
import re 

# Load models
diabetes_model = joblib.load('catboost_diabetes_pipeline.pkl')
heart_disease_model = joblib.load("catboost_heart_pipeline.pkl")
parkinsons_model = joblib.load('catboost_parkinson_pipeline.pkl')


# Database connection
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Create tables
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS results (
        username TEXT,
        disease TEXT,
        result TEXT,
        timestamp TEXT
    )
''')

# Session management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- CSS Styling ---
def set_css():
    st.markdown(
        """
        <style>
        .login-card, .register-card {
            padding: 1rem 1rem;
            border-radius: 1.5rem;
            width: 750px;
            height:70px;
            margin: 5% auto;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            position: absolute;
           }
        .login-title, .register-title {
       
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 2rem;
            margin-top: 1rem;
            margin-left: 4rem;
            color: #00796b;  
            position: relative;
            z-index: 1;  
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def set_bg_from_url(url, opacity=1):
    
    st.markdown(
        f"""
        <style>
            body{{
                margin: 0;
                padding: 0;
                height: 100vh;  /* Full height */
                width: 100vw;   /* Full width */
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
            .content {{
                position: relative;
                z-index: 1;  /* Ensure content is above the background */
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def home_page():
    st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

            body {
                font-family: 'Poppins', sans-serif;
            }
            .home-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                border-radius: 20px;
                justify-content: center;
                padding: 50px 20px;
                background: linear-gradient(to bottom right, #74ebd5, #ACB6E5);
                min-height: 82vh;
                width: 100%;
                 position: absolute;
        
            }
            .home-card {
                background: white;
                padding: 50px 60px;
                border-radius: 20px;
                box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 700px;
                width: 90%;
                 position: absolute;
                 margin-left:40px;
            z-index: 1; 
            }
            .home-title {
                font-size: 36px;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 20px;
                 position: relative;
                 text-align:center;
            z-index: 1; 
            }
            .home-description {
                font-size: 18px;
                color: #555;
                margin-bottom: 40px;
                padding: 50px 20px;
                text-align:center;
            }
            .button-container {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
            }
            .stButton>button {
                background: linear-gradient(to right, #6a11cb, #2575fc);
                color: white;
                padding: 12px 30px;
                font-size: 18px;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background: linear-gradient(to right, #2575fc, #6a11cb);
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.markdown('<div class="home-card">', unsafe_allow_html=True)

    st.markdown('<div class="home-title">Welcome to Multiple Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-description">Get early insights for Diabetes, Heart Disease, and Parkinson\'s by simply entering your health details. Secure & Accurate Predictions!</div>', unsafe_allow_html=True)

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    with col2:
        if st.button("Register", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close card
    st.markdown('</div>', unsafe_allow_html=True)  # Close container


def register():
    set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
    st.markdown('<div class="register-card">', unsafe_allow_html=True)
    st.markdown('<div class="register-title">Register</div>', unsafe_allow_html=True)

    username = st.text_input("Username", key="reg_user")
    password = st.text_input("Password", type="password", key="reg_pass")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        # Username format validation
        if not re.match(r'^[A-Za-z0-9_]{4,}$', username):
            st.error("Username must be at least 4 characters and can only contain letters, numbers, and underscores (_).")

        # Password strength validation
        elif len(password) < 6:
            st.error("Password must be at least 6 characters long.")
        elif not re.search(r'[A-Z]', password):
            st.error("Password must contain at least one uppercase letter.")
        elif not re.search(r'[a-z]', password):
            st.error("Password must contain at least one lowercase letter.")
        elif not re.search(r'[0-9]', password):
            st.error("Password must contain at least one digit.")
        elif password != confirm_password:
            st.error("Passwords do not match.")

        # Check if username already exists
        else:
            c.execute("SELECT * FROM users WHERE username = ?", (username,))
            existing_user = c.fetchone()

            if existing_user:
                st.error("Username already taken. Please choose another.")
            else:
                hashed_password = hash_password(password)
                with conn:
                    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                st.success("Registration successful! You can now log in.")
                st.session_state.page = 'login'
                st.rerun()

    if st.button("👉 Already have an account? Login here"):
        st.session_state.page = 'login'
        st.rerun()

    if st.button("🏠 Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def login():
    set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        # Username format check
        if not re.match(r'^[A-Za-z0-9_]{4,}$', username):
            st.error("Invalid username format.")
        else:
            hashed = hash_password(password)
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
            if c.fetchone():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.session_state.page = 'main'
                st.rerun()
            else:
                st.error("Invalid credentials.")

    if st.button("👉 Don't have an account? Register here"):
        st.session_state.page = 'register'
        st.rerun()

    if st.button("🏠 Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# Save result function
def save_result(disease, result):
    with conn:
        c.execute("INSERT INTO results (username, disease, result, timestamp) VALUES (?, ?, ?, ?)",
                  (st.session_state.username, disease, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# View history function
def view_history():
    st.subheader("Previous Test Results")
    c.execute("SELECT disease, result, timestamp FROM results WHERE username = ?", (st.session_state.username,))
    rows = c.fetchall()
    if rows:
        for r in rows:
            st.info(f"**{r[0]}** - {r[1]} at {r[2]}")
    else:
        st.write("No results found.")

# Prediction pages
def diabetes_page():
    st.subheader('Diabetes Prediction')

    st.markdown("**Please enter your medical details carefully. Normal ranges are mentioned as guidance.**")

    Pregnancies = st.number_input('Pregnancies (0-20)', min_value=0, max_value=20, help="Number of times pregnant. Typically 0-20.")
    Glucose = st.number_input('Glucose (mg/dL) (50-300)', min_value=50, max_value=300, help="Normal: 70-140 mg/dL. High: >140 mg/dL.")
    BloodPressure = st.number_input('Blood Pressure (mm Hg) (30-180)', min_value=30, max_value=180, help="Normal: around 80 mm Hg. Low: <60. High: >90.")
    SkinThickness = st.number_input('Skin Thickness (mm) (0-100)', min_value=0, max_value=100, help="Typically 10-50 mm. High values indicate obesity.")
    Insulin = st.number_input('Insulin (mu U/mL) (0-900)', min_value=0, max_value=900, help="Normal: 16-166 mu U/mL. 0 may indicate no measurement.")
    BMI = st.number_input('BMI (Body Mass Index) (10.0-70.0)', min_value=10.0, max_value=70.0, help="Normal: 18.5-24.9. Overweight >25. Obesity >30.")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function (0.0-2.5)', min_value=0.0, max_value=2.5, help="Higher values imply higher risk based on family history.")
    Age = st.number_input('Age (years) (1-120)', min_value=1, max_value=120, help="Age in years. Generally 1-120.")

    # Prediction
    diab_stage = symptoms = precautions = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            diab_prediction = diabetes_model.predict(user_input)

            if diab_prediction[0] == 1:
                if Glucose < 140:
                    diab_stage = 'Pre-diabetes'
                    symptoms = "Increased thirst, frequent urination, fatigue."
                    precautions = "Adopt a healthier diet, exercise regularly, and monitor blood sugar levels."
                elif Glucose < 200:
                    diab_stage = 'Type 2 Diabetes'
                    symptoms = "Excessive thirst, frequent urination, blurry vision."
                    precautions = "Adopt a low-carb, low-sugar diet, exercise, and manage stress."
                else:
                    diab_stage = 'Type 1 Diabetes'
                    symptoms = "Unexplained weight loss, extreme thirst, fatigue."
                    precautions = "Insulin therapy, regular blood sugar checks, healthy diet."
            else:
                diab_stage = 'No Diabetes'
                symptoms = "No symptoms of diabetes."
                precautions = "Maintain a healthy lifestyle with regular exercise and balanced nutrition."

            save_result("Diabetes Disease", diab_stage)
            
            st.success(f'Diabetes Prediction: {diab_stage}')
            st.info(f'Symptoms: {symptoms}')
            st.warning(f'Precautions: {precautions}')

        except ValueError:
            st.error('Please enter valid numerical values.')


def heart_page():
    st.subheader('Heart Disease Prediction')

    st.markdown("**Please enter your medical details carefully. Normal ranges are mentioned as guidance.**")

    # User input form with limits and helpful hints
    user_inputs = {}
    user_inputs['Age'] = st.number_input('Age (years) (1-120)', min_value=1, max_value=120, help="Age in years. Typical heart disease risk increases after 40.")
    user_inputs['Sex'] = st.selectbox('Sex', options=['Male', 'Female'], help="Select Male or Female.")
    user_inputs['Chest Pain'] = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], help="Type of chest pain experienced.")
    user_inputs['Resting BP'] = st.number_input('Resting Blood Pressure (mm Hg) (50-250)', min_value=50, max_value=250, help="Normal ~120 mm Hg. High >140 mm Hg indicates hypertension.")
    user_inputs['Cholesterol'] = st.number_input('Cholesterol (mg/dL) (100-600)', min_value=100, max_value=600, help="Normal <200 mg/dL. High >240 mg/dL.")
    user_inputs['Fasting Blood Sugar'] = st.selectbox('Fasting Blood Sugar > 120 mg/dL?', options=[0, 1], help="0 = No, 1 = Yes")
    user_inputs['ECG'] = st.selectbox('Resting ECG Results', options=['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], help="ECG interpretation.")
    user_inputs['Max Heart Rate'] = st.number_input('Maximum Heart Rate Achieved (60-220)', min_value=60, max_value=220, help="Normal varies with age. 220 - Age is typical max HR.")
    user_inputs['Exercise Angina'] = st.selectbox('Exercise-induced Angina', options=['Yes', 'No'], help="Yes = Angina occurs during exercise.")
    user_inputs['Oldpeak'] = st.number_input('Oldpeak (ST depression) (0.0-6.0)', min_value=0.0, max_value=6.0, help="ST depression induced by exercise relative to rest.")
    user_inputs['Slope'] = st.selectbox('Slope of Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'], help="Heart's response during exercise.")
    user_inputs['CA'] = st.number_input('Number of Major Vessels Colored by Fluoroscopy (0-3)', min_value=0, max_value=3, help="Number of major vessels.")
    user_inputs['Thal'] = st.selectbox('Thalassemia', options=['Normal', 'Fixed Defect', 'Reversible Defect'], help="Type of thalassemia.")

    # Preprocessing the categorical inputs
    sex_value = 1 if user_inputs['Sex'] == 'Male' else 0
    cp_value = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[user_inputs['Chest Pain']]
    restecg_value = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}[user_inputs['ECG']]
    exang_value = 1 if user_inputs['Exercise Angina'] == 'Yes' else 0
    slope_value = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[user_inputs['Slope']]
    thal_value = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}[user_inputs['Thal']]

    # Create DataFrame for prediction
    input_data = pd.DataFrame([{
        'age': float(user_inputs['Age']),
        'sex': float(sex_value),
        'cp': float(cp_value),
        'trestbps': float(user_inputs['Resting BP']),
        'chol': float(user_inputs['Cholesterol']),
        'fbs': float(user_inputs['Fasting Blood Sugar']),
        'restecg': float(restecg_value),
        'thalach': float(user_inputs['Max Heart Rate']),
        'exang': float(exang_value),
        'oldpeak': float(user_inputs['Oldpeak']),
        'slope': float(slope_value),
        'ca': float(user_inputs['CA']),
        'thal': float(thal_value)
    }])

    # Prediction and result display
    heart_result = symptoms = precautions = ''
    if st.button('Heart Disease Test Result'):
        try:
            # Prediction using the model
            heart_prediction = heart_disease_model.predict(input_data)

            # Risk evaluation based on prediction and cholesterol level
            cholesterol = float(user_inputs['Cholesterol'])
            if heart_prediction[0] == 1:
                if cholesterol < 200:
                    heart_result = 'Low Risk of Heart Disease'
                    symptoms = "Mild chest pain, discomfort during physical exertion."
                    precautions = "Maintain a healthy diet, avoid smoking, and exercise regularly."
                elif 200 <= cholesterol < 250:
                    heart_result = 'Moderate Risk of Heart Disease'
                    symptoms = "Frequent chest pain, shortness of breath."
                    precautions = "Monitor cholesterol, follow prescribed medication, and consult your doctor."
                else:
                    heart_result = 'High Risk of Heart Disease'
                    symptoms = "Severe chest pain, palpitations, excessive sweating."
                    precautions = "Seek immediate medical help. Surgery or stent placement may be necessary."
            else:
                heart_result = 'No Heart Disease'
                symptoms = "No signs of heart-related issues."
                precautions = "Maintain a healthy lifestyle with a balanced diet and regular check-ups."

            # Save the result
            save_result("Heart Disease", heart_result)

            # Display results
            st.success(f'Heart Disease Prediction: {heart_result}')
            st.info(f'Symptoms: {symptoms}')
            st.warning(f'Precautions: {precautions}')

        except Exception as e:
            st.error(f"An error occurred: {e}")




def parkinsons_page():
    st.subheader("Parkinson's Disease Prediction")
    
    st.markdown("**Please enter voice measurement values carefully. Guidance based on typical ranges is provided.**")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, help="Average vocal fundamental frequency. Normal ~150 Hz.")
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=75.0, max_value=600.0, help="Maximum vocal frequency. Typically higher than Fo.")
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, help="Minimum vocal frequency.")
        
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, help="Jitter percentage, measures voice frequency variation.")
        
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.01, help="Absolute jitter. Very small values.")
        
    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.1, help="Relative Average Perturbation.")
        
    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.1, help="Pitch Period Perturbation Quotient.")
        
    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.3, help="Derivative of RAP. Measures voice irregularity.")
        
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, help="Amplitude variation in voice signal.")
        
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=3.0, help="Shimmer in decibels.")
        
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, help="Three-point Amplitude Perturbation Quotient.")
        
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, help="Five-point Amplitude Perturbation Quotient.")
        
    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, help="Average Amplitude Perturbation Quotient.")
        
    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, help="Derivative of APQ3.")
        
    with col5:
        NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, help="Noise-to-Harmonics Ratio. High in Parkinson's cases.")
        
    with col1:
        HNR = st.number_input('HNR', min_value=0.0, max_value=50.0, help="Harmonics-to-Noise Ratio. Normal >20 is healthy.")
        
    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, help="Recurrence Period Density Entropy.")
        
    with col3:
        DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, help="Detrended Fluctuation Analysis.")
        
    with col4:
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=0.0, help="Spread1 measure. Negative values expected.")
        
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, max_value=0.5, help="Spread2 measure.")
        
    with col1:
        D2 = st.number_input('D2', min_value=1.0, max_value=3.0, help="Correlation dimension. Complexity of voice.")
        
    with col2:
        PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, help="Pitch Period Entropy.")
    
    # Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):                     
        prediction_proba = parkinsons_model.predict_proba([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,
                                                            DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR,
                                                            HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if prediction_proba[0][1] > 0.6:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            probability = prediction_proba[0][1]

            if probability < 0.7:
                parkinsons_stage = 'Early-Stage Parkinson’s Disease'
                symptoms = "Mild tremors, slight rigidity, difficulty with fine motor tasks."
                precautions = "Early intervention, physical therapy, medication for symptom management."
            elif 0.7 <= probability < 0.85:
                parkinsons_stage = 'Moderate Parkinson’s Disease'
                symptoms = "Increased tremors, muscle stiffness, slower movement, and postural instability."
                precautions = "Regular medication, physical therapy, and monitoring for cognitive issues."
            else:
                parkinsons_stage = 'Advanced Parkinson’s Disease'
                symptoms = "Severe tremors, balance problems, difficulty walking, significant cognitive decline."
                precautions = "Advanced medical care, possible deep brain stimulation, and assistance with daily tasks."
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            parkinsons_stage = 'No Parkinson’s Disease'
            symptoms = "No symptoms of Parkinson’s disease."
            precautions = "Maintain a healthy lifestyle with regular exercise."

        save_result("Parkinson Disease", parkinsons_stage)

        st.success(f'Parkinson’s Disease Prediction: {parkinsons_stage}')
        st.info(f'Symptoms: {symptoms}')
        st.warning(f'Precautions: {precautions}')

    


# Main app logic
def main_app():
    set_css()  # Set the background for the main app
    st.markdown('<div class="main-bg">', unsafe_allow_html=True)
    
    set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
    if st.session_state.logged_in:
        with st.sidebar:
            selected = option_menu(
                menu_title="Disease Prediction",
                options=["Diabetes", "Heart Disease", "Parkinson's", "View History", "Logout"],
                icons=["activity", "heart", "person", "clock", "box-arrow-left"],
                default_index=0
            )

        if selected == "Diabetes":
            diabetes_page()
        elif selected == "Heart Disease":
            heart_page()
        elif selected == "Parkinson's":
            parkinsons_page()
        elif selected == "View History":
            view_history()
        elif selected == "Logout":
            st.session_state.logged_in = False
            st.session_state.username = None
            st.success("You have been logged out.")
            st.rerun()
    else:
        home_page()

# Entry point
if __name__ == '__main__':
    set_css()  # Set the CSS for the initial page
    if not st.session_state.logged_in:
        # Default page decision (login or register)
        if 'page' not in st.session_state:
            st.session_state.page = 'home'  # Change this to 'home'

        if st.session_state.page == 'home':
            home_page()  # Call the home page function
        elif st.session_state.page == 'login':
            login()
        else:
            register()
    else:
        main_app()

footer = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet">
<footer>
    <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
        <p style="font-size:1.1rem;">
            Connect
            &nbsp;
            <a href="https://www.linkedin.com/in/">
                <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="Black" class="bi bi-linkedin" viewBox="0 0 16 16">
                    <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                </svg>
            </a>
            &nbsp;
            <a href="https://github.com/">
                <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="Black" class="bi bi-github" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            </a>
        </p>
    </div>
</footer>
"""
st.markdown(footer, unsafe_allow_html=True)
