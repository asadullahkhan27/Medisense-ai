import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import speech_recognition as sr
import sqlite3
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Hospital AI CDSS", layout="wide")
st.title("🏥 Clinical Decision Support AI System")

# =========================
# DATABASE (PRODUCTION READY)
# =========================
conn = sqlite3.connect("hospital_ai.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    input_type TEXT,
    data TEXT,
    result TEXT
)
""")
conn.commit()

# =========================
# LOAD MEDICAL AI MODEL
# =========================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# MEDICAL ENGINE (CDSS CORE)
# =========================
def medical_engine(text):
    prompt = f"""
You are a Clinical Decision Support System used in hospitals.

Analyze:
{text}

Return STRICT JSON format:
{{
  "conditions": [],
  "risk_level": "",
  "advice": "",
  "next_step": ""
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=300)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# OCR ENGINE
# =========================
def ocr(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# =========================
# VOICE ENGINE (FILE BASED)
# =========================
def voice_to_text(file):
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

# =========================
# SAVE TO DB
# =========================
def save(input_type, data, result):
    c.execute(
        "INSERT INTO reports (time, input_type, data, result) VALUES (?, ?, ?, ?)",
        (str(datetime.datetime.now()), input_type, data, result)
    )
    conn.commit()

# =========================
# UI TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 AI Diagnosis",
    "📷 Image Report",
    "🎤 Voice Input",
    "📷 Camera Scan",
    "📊 Records"
])

# =========================
# TAB 1 - TEXT AI
# =========================
with tab1:
    text = st.text_area("Enter Symptoms")

    if st.button("Analyze"):
        result = medical_engine(text)
        st.json(result)
        save("text", text, result)

# =========================
# TAB 2 - IMAGE REPORT
# =========================
with tab2:
    img = st.file_uploader("Upload Report Image")

    if img:
        st.image(img)
        text = ocr(img)
        result = medical_engine(text)

        st.write(text)
        st.json(result)

        save("image", text, result)

# =========================
# TAB 3 - VOICE
# =========================
with tab3:
    audio = st.file_uploader("Upload Voice (WAV)")

    if audio:
        st.audio(audio)
        text = voice_to_text(audio)
        result = medical_engine(text)

        st.write(text)
        st.json(result)

        save("voice", text, result)

# =========================
# TAB 4 - CAMERA
# =========================
with tab4:
    cam = st.camera_input("Capture Patient Image")

    if cam:
        st.image(cam)
        text = ocr(cam)
        result = medical_engine(text)

        st.write(result)
        save("camera", "camera_image", result)

# =========================
# TAB 5 - DATABASE RECORDS
# =========================
with tab5:
    st.subheader("Patient Records")

    c.execute("SELECT * FROM reports ORDER BY id DESC")
    data = c.fetchall()

    for row in data:
        st.write(row)
