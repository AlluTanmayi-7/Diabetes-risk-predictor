import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Load model
model = joblib.load("diabetes_model.pkl")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #2C3E50;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 3px;
}
.diabetic {
    background-color: #FDEDEC;
    color: #C0392B;
}
.not-diabetic {
    background-color: #E8F8F5;
    color: #117A65;
}
.confidence {
    background-color: #EBF5FB;
    padding: 14px;
    border-radius: 10px;
    font-size: 16px;
    color: #1A5276;
}
  
</style>
""", unsafe_allow_html=True)


# ---------- Main Title ----------
st.markdown('<div class="main-title">🩺 Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the details below to check diabetes risk</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.image(
        "icon.png",
        width=200

    )
    st.header("Input Features")

    pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
    glucose = st.slider("Glucose Level", 0, 200, 110)
    blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
    age = st.slider("Age", 10, 100, 30)

# ---------- Prediction ----------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Predict", use_container_width=True):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-box diabetic">⚠️ The person is DIABETIC</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box not-diabetic">✅ The person is NOT DIABETIC</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="confidence">Model confidence: <b>{probability:.2%}</b></div>',
            unsafe_allow_html=True
        )
