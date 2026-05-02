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
# LOAD MODEL
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
# HELPERS
# ===============================
def get_image(symptoms):
    s = symptoms.lower()
    if "chest" in s or "heart" in s:
        return "https://source.unsplash.com/600x400/?heart,medical"
    elif "fever" in s:
        return "https://source.unsplash.com/600x400/?fever,patient"
    elif "headache" in s:
        return "https://source.unsplash.com/600x400/?headache"
    elif "cough" in s:
        return "https://source.unsplash.com/600x400/?cough"
    return "https://source.unsplash.com/600x400/?hospital"

def severity_score(symptoms, temp, hr):
    score = 0

    if temp > 39:
        score += 3
    elif temp > 38:
        score += 2

    if hr > 120:
        score += 2

    if "chest" in symptoms.lower():
        score += 3
    if "breath" in symptoms.lower():
        score += 2

    if score >= 6:
        return "🔴 Emergency"
    elif score >= 4:
        return "🟠 High Risk"
    elif score >= 2:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

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
    return "🔴 Weak Immunity"

def doctor_recommendation(symptoms):
    s = symptoms.lower()
    if "chest" in s:
        return "Cardiologist"
    elif "skin" in s:
        return "Dermatologist"
    elif "eye" in s:
        return "Ophthalmologist"
    elif "fever" in s:
        return "General Physician"
    return "General Physician"

def analyze_ai(name, age, gender, symptoms):
    prompt = f"""
You are MediSense AI Medical Assistant.

Patient:
Name: {name}
Age: {age}
Gender: {gender}

Symptoms:
{symptoms}

Give structured medical analysis:
- Possible Conditions (max 3)
- Risk Level
- Advice
- Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# UI HEADER
# ===============================
st.title("🏥 MediSense AI")
st.caption("Clinical Decision Support + Lifestyle Health Intelligence System")

st.warning("⚠️ Educational use only. Not a replacement for medical advice.")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.header("Vitals")
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
hr = st.sidebar.slider("Heart Rate", 50, 150, 80)

st.sidebar.header("Lifestyle")
diet = st.sidebar.selectbox("Diet", ["Good", "Average", "Poor"])
sleep = st.sidebar.slider("Sleep (hours)", 0, 12, 7)
exercise = st.sidebar.selectbox("Exercise", ["Regular", "Sometimes", "None"])

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🧠 Analysis", "💬 Chat AI", "📜 History"])

# ===============================
# TAB 1: ANALYSIS
# ===============================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        symptoms = st.text_area("Enter Symptoms", height=150)
        run = st.button("Analyze Health")

    with col2:

        if run and symptoms:

            with st.spinner("Analyzing patient data..."):

                ai_report = analyze_ai(name, age, gender, symptoms)
                risk = severity_score(symptoms, temp, hr)
                imm = immunity_score(diet, sleep, exercise)
                doc = doctor_recommendation(symptoms)
                img = get_image(symptoms)

            # ALERT SYSTEM
            if "🔴" in risk:
                st.error("🚨 Emergency detected! Seek immediate medical attention.")
            elif "🟠" in risk:
                st.warning("⚠️ High risk detected. Medical consultation recommended.")

            st.subheader("Risk Level")
            st.markdown(f"### {risk}")

            st.subheader("Immunity Status")
            st.markdown(f"### {imm}")

            st.subheader("Recommended Doctor")
            st.info(doc)

            st.image(img)

            st.subheader("AI Medical Report")
            st.write(ai_report)

            st.download_button(
                "Download Report",
                ai_report,
                file_name="medisense_report.txt"
            )

            # Save history
            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "name": name,
                "symptoms": symptoms,
                "risk": risk,
                "immunity": imm
            })

# ===============================
# TAB 2: CHAT
# ===============================
with tab2:

    chat_input = st.text_input("Ask MediSense AI Doctor")

    if st.button("Send"):
        if chat_input:
            response = analyze_ai(name, age, gender, chat_input)
            st.write("🤖 AI:", response)

# ===============================
# TAB 3: HISTORY
# ===============================
with tab3:

    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in reversed(st.session_state.history):
            with st.expander(f"{h['name']} | {h['time']}"):
                st.write("Symptoms:", h["symptoms"])
                st.write("Risk:", h["risk"])
                st.write("Immunity:", h["immunity"])
