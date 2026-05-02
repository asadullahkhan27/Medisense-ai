import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
from PIL import Image
import pytesseract
import speech_recognition as sr

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="MediSense AI Pro",
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
# AI CORE
# ===============================
def analyze_ai(name, age, gender, text):
    prompt = f"""
You are MediSense AI Medical Assistant.

Patient:
Name: {name}
Age: {age}
Gender: {gender}

Input:
{text}

Return:
- Possible Conditions (max 3)
- Risk Level
- Advice
- Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# RISK ANALYSIS
# ===============================
def severity_score(text):
    s = text.lower()
    score = 0

    if "fever" in s:
        score += 2
    if "chest" in s or "pain" in s:
        score += 3
    if "breath" in s:
        score += 2
    if "headache" in s:
        score += 1

    if score >= 5:
        return "🔴 Emergency"
    elif score >= 3:
        return "🟠 High Risk"
    elif score >= 1:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

# ===============================
# MEDICATION SUGGESTION
# ===============================
def suggest_medication(text):
    s = text.lower()
    meds = []

    if "fever" in s:
        meds.append("Paracetamol")
    if "headache" in s:
        meds.append("Ibuprofen")
    if "cough" in s:
        meds.append("Cough Syrup (Dextromethorphan)")
    if "cold" in s:
        meds.append("Antihistamine")

    return meds if meds else ["Consult doctor"]

# ===============================
# OCR IMAGE ANALYSIS
# ===============================
def extract_text(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# ===============================
# VOICE INPUT (SAFE VERSION)
# ===============================
def voice_to_text():
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            audio = r.listen(source, timeout=5)

        text = r.recognize_google(audio)
        return text

    except Exception as e:
        return f"Voice error: {str(e)}"

# ===============================
# UI HEADER
# ===============================
st.title("🏥 MediSense AI Pro")
st.warning("For educational use only. Not a medical diagnosis tool.")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Symptom Analysis",
    "💬 Chat AI",
    "📷 Image Report",
    "🎤 Voice Input",
    "📊 History"
])

# ===============================
# TAB 1 - SYMPTOMS
# ===============================
with tab1:
    symptoms = st.text_area("Enter Symptoms")

    if st.button("Analyze Symptoms"):
        result = analyze_ai(name, age, gender, symptoms)
        risk = severity_score(symptoms)
        meds = suggest_medication(symptoms)

        st.subheader("Risk Level")
        st.write(risk)

        st.subheader("Medication Suggestion")
        st.write(meds)

        st.subheader("AI Report")
        st.write(result)

        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "input": symptoms,
            "type": "text"
        })

# ===============================
# TAB 2 - CHAT
# ===============================
with tab2:
    chat = st.text_input("Ask AI Doctor")

    if st.button("Send"):
        reply = analyze_ai(name, age, gender, chat)
        st.write(reply)

# ===============================
# TAB 3 - IMAGE REPORT OCR
# ===============================
with tab3:
    st.subheader("Upload Medical / Excel Screenshot")

    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if file:
        st.image(file)

        text = extract_text(file)

        st.subheader("Extracted Text")
        st.write(text)

        if text:
            analysis = analyze_ai(name, age, gender, text)

            st.subheader("AI Analysis")
            st.write(analysis)

            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "input": text,
                "type": "image"
            })

# ===============================
# TAB 4 - VOICE INPUT
# ===============================
with tab4:
    st.subheader("Voice Symptom Input")

    if st.button("Start Voice Recording"):
        spoken = voice_to_text()

        st.write("You said:")
        st.success(spoken)

        if spoken:
            result = analyze_ai(name, age, gender, spoken)

            st.subheader("AI Analysis")
            st.write(result)

            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "input": spoken,
                "type": "voice"
            })

# ===============================
# TAB 5 - HISTORY
# ===============================
with tab5:
    st.subheader("Patient History")

    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in reversed(st.session_state.history):
            st.write(h)
