import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide"
)

# ===============================
# MODEL LOAD
# ===============================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# SYMPTOM → DISEASE MAPPING (LOGIC ENGINE)
# ===============================
def symptom_prediction(symptoms):
    s = symptoms.lower()

    if "cough" in s and "fever" in s:
        return "Flu / Viral Infection / Seasonal Cold"
    elif "cough" in s:
        return "Common Cold / Weather Infection"
    elif "fever" in s:
        return "Viral Fever / Infection"
    elif "headache" in s:
        return "Migraine / Stress / Dehydration"
    elif "chest" in s or "heart" in s:
        return "Heart-related issue / Anxiety"
    else:
        return "General Infection / Needs Medical Check"

# ===============================
# IMAGE SELECTOR
# ===============================
def get_image(symptoms):
    s = symptoms.lower()

    if "chest" in s:
        return "https://source.unsplash.com/600x400/?heart,medical"
    elif "fever" in s:
        return "https://source.unsplash.com/600x400/?fever,patient"
    elif "headache" in s:
        return "https://source.unsplash.com/600x400/?headache"
    elif "cough" in s:
        return "https://source.unsplash.com/600x400/?cough"
    else:
        return "https://source.unsplash.com/600x400/?hospital"

# ===============================
# RISK ENGINE
# ===============================
def risk_level(symptoms, temp, hr):
    score = 0

    if temp > 39:
        score += 3
    elif temp > 38:
        score += 2

    if hr > 120:
        score += 2

    if "chest" in symptoms.lower():
        score += 3

    if score >= 6:
        return "🔴 Emergency"
    elif score >= 4:
        return "🟠 High Risk"
    elif score >= 2:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# ===============================
# IMMUNITY ENGINE
# ===============================
def immunity_score(diet, sleep, exercise):
    score = 0

    if diet == "Good":
        score += 2
    elif diet == "Average":
        score += 1

    if sleep >= 7:
        score += 2
    elif sleep >= 5:
        score += 1

    if exercise == "Regular":
        score += 2
    elif exercise == "Sometimes":
        score += 1

    if score >= 5:
        return "🟢 Strong Immunity"
    elif score >= 3:
        return "🟡 Moderate Immunity"
    else:
        return "🔴 Weak Immunity"

# ===============================
# AI MODEL
# ===============================
def ai_analysis(name, age, gender, symptoms):

    prompt = f"""
You are MediSense AI Medical Assistant.

Patient:
Name: {name}
Age: {age}
Gender: {gender}

Symptoms:
{symptoms}

Provide:
- Possible Conditions
- Risk Level
- Doctor Recommendation
- Advice
- Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=250,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# UI HEADER
# ===============================
st.title("🏥 MediSense AI")
st.caption("Clinical Decision Support System + AI Health Intelligence")

st.warning("⚠️ This system is for educational purposes only.")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.header("Vitals")
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
hr = st.sidebar.slider("Heart Rate", 50, 150, 80)

st.sidebar.header("Lifestyle")
diet = st.sidebar.selectbox("Diet", ["Good", "Average", "Poor"])
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
exercise = st.sidebar.selectbox("Exercise", ["Regular", "Sometimes", "None"])

# ===============================
# MAIN UI
# ===============================
tab1, tab2, tab3 = st.tabs(["🧠 Analysis", "📜 History", "ℹ️ About"])

# ===============================
# TAB 1
# ===============================
with tab1:

    symptoms = st.text_area("Enter Symptoms", height=150)

    if st.button("Analyze Health") and symptoms:

        with st.spinner("Analyzing patient data..."):

            ai_result = ai_analysis(name, age, gender, symptoms)
            disease = symptom_prediction(symptoms)
            risk = risk_level(symptoms, temp, hr)
            immune = immunity_score(diet, sleep, exercise)
            img = get_image(symptoms)

        # ALERT SYSTEM
        if "🔴" in risk:
            st.error("🚨 Emergency detected! Seek medical attention immediately.")

        st.subheader("🧬 Possible Disease")
        st.info(disease)

        st.subheader("⚠️ Risk Level")
        st.markdown(f"### {risk}")

        st.subheader("🛡️ Immunity Status")
        st.markdown(f"### {immune}")

        st.subheader("👨‍⚕️ AI Medical Report")
        st.write(ai_result)

        st.image(img)

        st.download_button(
            "Download Report",
            ai_result,
            file_name="medisense_report.txt"
        )

        # SAVE HISTORY
        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "name": name,
            "symptoms": symptoms,
            "disease": disease,
            "risk": risk,
            "immune": immune
        })

# ===============================
# TAB 2 HISTORY
# ===============================
with tab2:

    if not st.session_state.history:
        st.info("No history available")
    else:
        for h in reversed(st.session_state.history):
            with st.expander(f"{h['name']} | {h['time']}"):
                st.write("Symptoms:", h["symptoms"])
                st.write("Disease:", h["disease"])
                st.write("Risk:", h["risk"])
                st.write("Immunity:", h["immune"])

# ===============================
# TAB 3 ABOUT
# ===============================
with tab3:

    st.markdown("""
### MediSense AI

AI-powered healthcare assistant combining:

- Symptom-based disease prediction  
- Risk analysis system  
- Immunity scoring engine  
- AI medical report generation  
- Lifestyle health tracking  

**Tech Stack:**
- Streamlit  
- Hugging Face Transformers  
- PyTorch  
""")
