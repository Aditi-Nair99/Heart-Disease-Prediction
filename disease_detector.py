import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("❤️ Heart Disease Prediction Application")
st.markdown("""
This app predicts the likelihood of heart disease based on patient health metrics.
Enter the required information below and click **Predict** to see the results.
""")

# Load or train model
@st.cache_resource
def load_model():
    # We'll use a pre-trained model or train one if not available
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        # If no saved model exists, we'll create a dummy one for demonstration
        # In a real application, you would train a proper model here
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=13, n_informative=8, 
                                  n_redundant=5, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        return model, scaler

model, scaler = load_model()

# Sidebar for user input
st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 45)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', 
                             ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', 
                                  ('Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', 
                                ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thalassemia', ('Normal', 'Fixed Defect', 'Reversible Defect'))
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_dict[cp]
    fbs = 1 if fbs == 'True' else 0
    restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    restecg = restecg_dict[restecg]
    exang = 1 if exang == 'Yes' else 0
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope = slope_dict[slope]
    thal_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = thal_dict[thal]
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.subheader("Patient Data Summary")
st.write(input_df)

# Preprocess the input
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display results
st.subheader("Prediction")
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(f"**Prediction:** {heart_disease[prediction][0]}")

# Probability meter
st.subheader("Prediction Probability")
prob = prediction_proba[0][1] * 100
st.markdown(f"**Probability of heart disease: {prob:.2f}%**")

# Create a visual gauge for risk
fig, ax = plt.subplots(figsize=(8, 2))
ax.barh([0], [prob], color=['green' if prob < 30 else 'orange' if prob < 70 else 'red'])
ax.set_xlim(0, 100)
ax.set_xlabel('Probability (%)')
ax.set_title('Heart Disease Risk')
ax.text(prob + 1, 0, f'{prob:.1f}%', va='center')
st.pyplot(fig)

# Risk interpretation
st.subheader("Risk Interpretation")
if prob < 30:
    st.success("Low risk of heart disease. Maintain your healthy lifestyle!")
elif prob < 70:
    st.warning("Moderate risk of heart disease. Consider consulting a healthcare provider and making lifestyle changes.")
else:
    st.error("High risk of heart disease. Please consult a healthcare provider for further evaluation.")

# Feature importance (for demonstration)
st.subheader("Feature Importance")
# For demonstration, we'll show some feature importance (in a real app, this would come from the model)
features = ['Age', 'Sex', 'Chest Pain', 'BP', 'Cholesterol', 'Blood Sugar', 
            'ECG', 'Max HR', 'Exercise Angina', 'ST Depression', 'Slope', 'Vessels', 'Thalassemia']

# Random importance values for demonstration - in a real app, use model.feature_importances_
importance = np.array([0.15, 0.08, 0.12, 0.09, 0.11, 0.05, 0.06, 0.13, 0.07, 0.08, 0.03, 0.02, 0.01])

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values('Importance', ascending=True)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feature_importance['Feature'], feature_importance['Importance'])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance in Prediction')
st.pyplot(fig)

# Add some health recommendations
st.sidebar.header("Health Recommendations")
st.sidebar.info("""
- Maintain a healthy weight
- Exercise regularly
- Eat a balanced diet
- Monitor blood pressure and cholesterol
- Avoid smoking
- Limit alcohol consumption
- Manage stress effectively
""")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This application is for educational purposes only and should not be used as a substitute for professional medical advice.")