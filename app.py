import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="MediSense AI", page_icon="🏥", layout="wide")

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# Session State
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# Helper Functions
# ===============================
def get_image(symptoms):
    s = symptoms.lower()
    if "chest" in s:
        return "https://source.unsplash.com/600x400/?heart"
    elif "fever" in s:
        return "https://source.unsplash.com/600x400/?fever"
    elif "headache" in s:
        return "https://source.unsplash.com/600x400/?headache"
    else:
        return "https://source.unsplash.com/600x400/?hospital"

def calculate_severity(temp, hr):
    if temp > 39 or hr > 120:
        return "🔴 Emergency"
    elif temp > 38:
        return "🟠 High"
    elif temp > 37:
        return "🟡 Medium"
    else:
        return "🟢 Low"

def recommend_doctor(symptoms):
    s = symptoms.lower()
    if "chest" in s:
        return "Cardiologist"
    elif "skin" in s:
        return "Dermatologist"
    elif "fever" in s:
        return "General Physician"
    else:
        return "General Doctor"

def analyze(name, age, gender, symptoms):
    prompt = f"""
You are MediSense AI.

Patient:
{name}, {age}, {gender}

Symptoms:
{symptoms}

Give:
- Conditions
- Risk Level
- Advice
- Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(**inputs, max_length=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# UI HEADER
# ===============================
st.title("🏥 MediSense AI")
st.caption("Clinical Decision Support System")

st.warning("⚠️ Not a replacement for real doctors")

# ===============================
# SIDEBAR (Vitals)
# ===============================
st.sidebar.header("Patient Details")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.header("Vitals")
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
hr = st.sidebar.slider("Heart Rate", 50, 150, 80)
bp = st.sidebar.text_input("Blood Pressure (e.g. 120/80)")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🧠 Analysis", "📜 History", "ℹ️ About"])

# ===============================
# TAB 1: ANALYSIS
# ===============================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        symptoms = st.text_area("Enter Symptoms", height=150)
        run = st.button("Run Analysis")

    with col2:
        if run and symptoms:

            with st.spinner("Analyzing..."):
                result = analyze(name, age, gender, symptoms)

            severity = calculate_severity(temp, hr)
            doctor = recommend_doctor(symptoms)
            image = get_image(symptoms)

            # Emergency Alert
            if "🔴" in severity:
                st.error("🚨 EMERGENCY: Seek immediate medical help!")

            st.subheader("Risk Level")
            st.markdown(f"### {severity}")

            st.subheader("Recommended Doctor")
            st.info(doctor)

            st.image(image)

            st.subheader("Medical Report")
            st.write(result)

            # Download
            st.download_button(
                "Download Report",
                data=result,
                file_name="report.txt"
            )

            # Save history
            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "name": name,
                "symptoms": symptoms,
                "severity": severity
            })

# ===============================
# TAB 2: HISTORY
# ===============================
with tab2:

    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in st.session_state.history[::-1]:
            with st.expander(f"{h['name']} | {h['time']}"):
                st.write(f"Symptoms: {h['symptoms']}")
                st.write(f"Severity: {h['severity']}")

# ===============================
# TAB 3: ABOUT
# ===============================
with tab3:

    st.markdown("""
### MediSense AI

AI-powered clinical assistant using NLP.

**Features:**
- Symptom Analysis  
- Risk Detection  
- Doctor Recommendation  
- Vital Monitoring  
- History Tracking  
""")
