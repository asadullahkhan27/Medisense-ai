import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="MediSense AI", page_icon="🏥", layout="wide")

# ===============================
# Load Model (cached)
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

if "chat" not in st.session_state:
    st.session_state.chat = []

# ===============================
# Helpers
# ===============================
def get_image(symptoms):
    s = symptoms.lower()
    if "chest" in s:
        return "https://source.unsplash.com/600x400/?heart"
    elif "fever" in s:
        return "https://source.unsplash.com/600x400/?fever"
    elif "headache" in s:
        return "https://source.unsplash.com/600x400/?headache"
    return "https://source.unsplash.com/600x400/?hospital"

def severity(temp, hr):
    if temp > 39 or hr > 120:
        return "🔴 Emergency"
    elif temp > 38:
        return "🟠 High"
    elif temp > 37:
        return "🟡 Medium"
    return "🟢 Low"

def doctor(symptoms):
    s = symptoms.lower()
    if "chest" in s:
        return "Cardiologist"
    elif "skin" in s:
        return "Dermatologist"
    elif "eye" in s:
        return "Ophthalmologist"
    return "General Physician"

def generate_report(name, age, gender, symptoms, lang="English"):
    prompt = f"""
You are MediSense AI.

Patient:
{name}, {age}, {gender}

Symptoms:
{symptoms}

Respond in {lang}.

Give:
1. Conditions (max 3)
2. Risk Level
3. Doctor Type
4. Advice
5. Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# UI Header
# ===============================
st.title("🏥 MediSense AI")
st.caption("AI Clinical Decision Support System")

st.warning("⚠️ This system is not a replacement for professional medical advice.")

# ===============================
# Sidebar
# ===============================
st.sidebar.header("Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.header("Vitals")
temp = st.sidebar.slider("Temperature (°C)", 35.0, 42.0, 37.0)
hr = st.sidebar.slider("Heart Rate", 50, 150, 80)

lang = st.sidebar.selectbox("Language", ["English", "Urdu"])

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Analysis", "💬 AI Doctor Chat", "📜 History", "ℹ️ About"])

# ===============================
# TAB 1 — Analysis
# ===============================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        symptoms = st.text_area("Enter Symptoms")
        run = st.button("Analyze")

    with col2:
        if run and symptoms:
            with st.spinner("Analyzing..."):
                report = generate_report(name, age, gender, symptoms, lang)

            sev = severity(temp, hr)
            doc = doctor(symptoms)
            img = get_image(symptoms)

            if "🔴" in sev:
                st.error("🚨 Emergency! Visit hospital immediately.")

            st.subheader("Risk Level")
            st.markdown(f"### {sev}")

            st.subheader("Recommended Doctor")
            st.info(doc)

            st.image(img)

            st.subheader("Medical Report")
            st.write(report)

            st.download_button("Download Report", report, file_name="report.txt")

            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "name": name,
                "symptoms": symptoms,
                "report": report
            })

# ===============================
# TAB 2 — Chatbot
# ===============================
with tab2:
    user_input = st.text_input("Ask MediSense AI Doctor")

    if st.button("Send"):
        if user_input:
            response = generate_report(name, age, gender, user_input, lang)
            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("AI", response))

    for sender, msg in st.session_state.chat:
        st.markdown(f"**{sender}:** {msg}")

# ===============================
# TAB 3 — History
# ===============================
with tab3:
    if not st.session_state.history:
        st.info("No history available")
    else:
        for h in reversed(st.session_state.history):
            with st.expander(f"{h['name']} | {h['time']}"):
                st.write(h["symptoms"])
                st.write(h["report"])

# ===============================
# TAB 4 — About
# ===============================
with tab4:
    st.markdown("""
### MediSense AI

AI-powered clinical assistant using Hugging Face.

**Features:**
- Symptom Analysis  
- AI Doctor Chat  
- Risk Detection  
- Doctor Recommendation  
- Multilingual Support  
- Patient History  
""")
