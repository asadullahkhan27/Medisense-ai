import streamlit as st
import sqlite3
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import speech_recognition as sr

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Fast Hospital AI", layout="wide")
st.title("🏥 Fast Hospital AI System")

# ===============================
# DATABASE
# ===============================
conn = sqlite3.connect("hospital_ai.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    created_at TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    input_type TEXT,
    input_text TEXT,
    result TEXT,
    time TEXT
)
""")

conn.commit()

# ===============================
# FAST MODEL LOAD
# ===============================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"   # ⚡ FAST MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# ⚡ FAST AI ENGINE
# ===============================
@st.cache_data
def analyze_ai(text):
    prompt = f"""
You are a hospital AI assistant.

Patient Input:
{text}

Return:

Possible Conditions:
- condition 1
- condition 2
- condition 3

Risk Level:
(low / medium / high / emergency)

Advice:
- short medical advice

Home Care:
- basic care steps
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=180,   # ⚡ FAST
        min_length=60,
        num_beams=2,      # ⚡ FAST
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# PATIENT SYSTEM
# ===============================
def add_patient(name, age, gender):
    c.execute(
        "INSERT INTO patients (name, age, gender, created_at) VALUES (?, ?, ?, ?)",
        (name, age, gender, str(datetime.datetime.now()))
    )
    conn.commit()
    return c.lastrowid

def get_patients():
    c.execute("SELECT * FROM patients ORDER BY id DESC")
    return c.fetchall()

def search_patient(name):
    c.execute("SELECT * FROM patients WHERE name LIKE ?", ('%' + name + '%',))
    return c.fetchall()

# ===============================
# OCR SAFE
# ===============================
def extract_text(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except:
        return ""

# ===============================
# VOICE SAFE
# ===============================
def voice_to_text(file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except:
        return "Voice input detected"

# ===============================
# TABS
# ===============================
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧑 Patient Info",
    "🧠 AI Diagnosis",
    "📷 Image Report",
    "🎤 Voice Input",
    "📷 Camera Scan",
    "📜 History",
    "📊 Patients"
])

# ===============================
# 🧑 PATIENT INFO
# ===============================
with tab0:
    st.subheader("Patient Registration")

    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Register Patient"):
        pid = add_patient(name, age, gender)
        st.success(f"Patient Registered! ID: {pid}")

    st.divider()

    st.subheader("Search Patient")
    search = st.text_input("Search Name")

    if st.button("Search"):
        results = search_patient(search)
        for r in results:
            st.write(f"ID: {r[0]} | {r[1]} | Age: {r[2]} | {r[3]}")

# ===============================
# 🧠 AI DIAGNOSIS (FAST)
# ===============================
with tab1:
    symptoms = st.text_area("Enter Symptoms")

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result = analyze_ai(symptoms)

        st.subheader("AI Report")
        st.write(result)

        c.execute(
            "INSERT INTO history (patient_name, input_type, input_text, result, time) VALUES (?, ?, ?, ?, ?)",
            (name, "text", symptoms, result, str(datetime.datetime.now()))
        )
        conn.commit()

# ===============================
# 📷 IMAGE REPORT
# ===============================
with tab2:
    file = st.file_uploader("Upload Image")

    if file:
        st.image(file)

        text = extract_text(file)

        if not text or len(text.strip()) < 5:
            text = "Medical image analysis required"

        with st.spinner("Analyzing image..."):
            result = analyze_ai(text)

        st.write(result)

# ===============================
# 🎤 VOICE INPUT
# ===============================
with tab3:
    audio = st.file_uploader("Upload WAV")

    if audio:
        st.audio(audio)

        if st.button("Convert"):
            text = voice_to_text(audio)
            result = analyze_ai(text)

            st.write(result)

# ===============================
# 📷 CAMERA SCAN
# ===============================
with tab4:
    cam = st.camera_input("Take Picture")

    if cam:
        st.image(cam)

        text = extract_text(cam)

        if not text or len(text.strip()) < 5:
            text = "Camera medical scan requires analysis"

        with st.spinner("Analyzing camera..."):
            result = analyze_ai(text)

        st.write(result)

# ===============================
# 📜 HISTORY
# ===============================
with tab5:
    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()

    for d in data:
        st.write(d)

# ===============================
# 📊 PATIENTS
# ===============================
with tab6:
    patients = get_patients()

    for p in patients:
        st.write(f"ID: {p[0]} | {p[1]} | Age: {p[2]} | Gender: {p[3]}")
