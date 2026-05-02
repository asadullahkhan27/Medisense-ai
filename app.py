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
# SESSION
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# IMAGE HELPERS
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

# ===============================
# RISK ANALYSIS
# ===============================
def severity_score(symptoms, temp, hr):
    score = 0
    s = symptoms.lower()

    if temp > 39:
        score += 3
    elif temp > 38:
        score += 2

    if hr > 120:
        score += 2

    if "chest" in s:
        score += 3
    if "breath" in s:
        score += 2
    if "pain" in s:
        score += 1

    if score >= 6:
        return "🔴 Emergency"
    elif score >= 4:
        return "🟠 High Risk"
    elif score >= 2:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

# ===============================
# MEDICATION SUGGESTION (NEW FEATURE)
# ===============================
def suggest_medication(symptoms):
    s = symptoms.lower()

    meds = []

    if "fever" in s:
        meds.append("Paracetamol (for fever relief)")
    if "headache" in s:
        meds.append("Ibuprofen (pain relief)")
    if "cough" in s:
        meds.append("Dextromethorphan syrup (dry cough)")
    if "cold" in s:
        meds.append("Antihistamine (like Loratadine)")
    if "body pain" in s:
        meds.append("Acetaminophen / Ibuprofen")

    if not meds:
        return ["Consult doctor for proper prescription"]

    return meds

# ===============================
# DOCTOR SUGGESTION
# ===============================
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

# ===============================
# AI REPORT
# ===============================
def analyze_ai(name, age, gender, symptoms):
    prompt = f"""
You are MediSense AI.

Patient:
Name: {name}
Age: {age}
Gender: {gender}

Symptoms:
{symptoms}

Give:
- Possible Conditions (max 3)
- Risk Level
- Advice
- Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# IMAGE ANALYSIS (SIMULATED CAMERA FEATURE)
# ===============================
def analyze_body_image(image_type):
    if image_type == "Eyes":
        return "Possible signs: fatigue, dehydration, or infection (non-diagnostic check)"
    elif image_type == "Tongue":
        return "Possible signs: dehydration, vitamin deficiency, or fever indicators"
    return "No clear analysis available"

# ===============================
# UI HEADER
# ===============================
st.title("🏥 MediSense AI")
st.warning("Educational tool only. Not a medical diagnosis system.")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.header("Vitals")
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
hr = st.sidebar.slider("Heart Rate", 50, 150, 80)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Analysis",
    "💬 Chat",
    "📷 Vision Check",
    "📜 History"
])

# ===============================
# TAB 1
# ===============================
with tab1:
    symptoms = st.text_area("Enter Symptoms")
    run = st.button("Analyze")

    if run and symptoms:

        ai_report = analyze_ai(name, age, gender, symptoms)
        risk = severity_score(symptoms, temp, hr)
        meds = suggest_medication(symptoms)
        doc = doctor_recommendation(symptoms)

        st.subheader("Risk Level")
        st.write(risk)

        st.subheader("Recommended Doctor")
        st.info(doc)

        st.subheader("Suggested Medication (Basic Guidance)")
        for m in meds:
            st.write("•", m)

        st.subheader("AI Report")
        st.write(ai_report)

        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "name": name,
            "symptoms": symptoms,
            "risk": risk
        })

# ===============================
# TAB 2
# ===============================
with tab2:
    chat_input = st.text_input("Ask AI Doctor")

    if st.button("Send"):
        response = analyze_ai(name, age, gender, chat_input)
        st.write(response)

# ===============================
# TAB 3 (VISION FEATURE)
# ===============================
with tab3:
    st.subheader("Upload Eye / Tongue Image")

    img_type = st.selectbox("Select Check Type", ["Eyes", "Tongue"])

    camera = st.camera_input("Capture Image")

    if camera:
        st.image(camera)

        result = analyze_body_image(img_type)

        st.subheader("AI Visual Analysis (Non-Diagnostic)")
        st.info(result)

# ===============================
# TAB 4
# ===============================
with tab4:
    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in reversed(st.session_state.history):
            st.write(h)
