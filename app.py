import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BlipProcessor,
    BlipForConditionalGeneration
)
from PIL import Image
import pytesseract
import speech_recognition as sr
import datetime
import torch

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Hospital AI System", layout="wide")
st.title("🏥 Hospital AI Assistant (Full System)")

# ===============================
# TEXT MODEL (Medical AI)
# ===============================
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, text_model = load_text_model()

# ===============================
# VISION MODEL (BLIP)
# ===============================
@st.cache_resource
def load_vision_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, vision_model = load_vision_model()

# ===============================
# HISTORY
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# 🧠 MEDICAL AI
# ===============================
def medical_ai(text):
    prompt = f"""
You are a hospital-level medical AI assistant.

Patient input:
{text}

Return:
1. Possible Conditions
2. Risk Level
3. Advice
4. Home Care
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = text_model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# 📷 VISION AI
# ===============================
def vision_ai(image):
    inputs = processor(image, return_tensors="pt")
    out = vision_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ===============================
# 👁 EYE ANALYSIS
# ===============================
def eye_analysis(text):
    t = text.lower()
    if "red" in t:
        return "Possible Eye Infection"
    if "yellow" in t:
        return "Possible Liver/Jaundice Sign"
    return "No major eye issue detected"

# ===============================
# 👅 TONGUE ANALYSIS
# ===============================
def tongue_analysis(text):
    t = text.lower()
    if "white" in t:
        return "Possible dehydration or infection"
    if "red" in t:
        return "Possible fever/inflammation"
    return "Normal"

# ===============================
# 💊 MEDICATION SUGGESTION
# ===============================
def meds(text):
    t = text.lower()
    m = []

    if "fever" in t:
        m.append("Paracetamol")
    if "headache" in t:
        m.append("Ibuprofen")
    if "cough" in t:
        m.append("Cough Syrup")

    return m if m else ["Consult doctor"]

# ===============================
# 📊 OCR IMAGE REPORT
# ===============================
def extract_text(image_file):
    try:
        image = Image.open(image_file)
        return pytesseract.image_to_string(image)
    except:
        return ""

# ===============================
# 🎤 VOICE INPUT (SAFE)
# ===============================
def voice_to_text(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception as e:
        return f"Voice Error: {str(e)}"

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧠 Medical AI",
    "📷 Vision AI",
    "📊 Image Report",
    "🎤 Voice Input",
    "📷 Camera Scan",
    "📜 History"
])

# ===============================
# 🧠 TAB 1 - MEDICAL AI
# ===============================
with tab1:
    text = st.text_area("Enter Symptoms")

    if st.button("Analyze Symptoms"):
        result = medical_ai(text)
        st.write(result)

        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "input": text,
            "type": "text"
        })

# ===============================
# 📷 TAB 2 - VISION AI
# ===============================
with tab2:
    img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img:
        image = Image.open(img)
        st.image(image)

        caption = vision_ai(image)

        st.subheader("AI Vision Output")
        st.write(caption)

        st.subheader("Eye Analysis")
        st.write(eye_analysis(caption))

        st.subheader("Tongue Analysis")
        st.write(tongue_analysis(caption))

# ===============================
# 📊 TAB 3 - IMAGE REPORT OCR
# ===============================
with tab3:
    file = st.file_uploader("Upload Medical Report Image", type=["png", "jpg", "jpeg"])

    if file:
        st.image(file)

        text = extract_text(file)

        if not text:
            text = "Medical image uploaded"

        st.subheader("Extracted Text")
        st.write(text)

        result = medical_ai(text)
        st.subheader("AI Report")
        st.write(result)

        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "input": text,
            "type": "image"
        })

# ===============================
# 🎤 TAB 4 - VOICE INPUT
# ===============================
with tab4:
    audio = st.file_uploader("Upload Voice (WAV)", type=["wav"])

    if audio:
        st.audio(audio)

        if st.button("Convert Voice"):
            text = voice_to_text(audio)

            st.success("Converted Text")
            st.write(text)

            result = medical_ai(text)
            st.subheader("AI Analysis")
            st.write(result)

            st.session_state.history.append({
                "time": str(datetime.datetime.now()),
                "input": text,
                "type": "voice"
            })

# ===============================
# 📷 TAB 5 - CAMERA SCAN
# ===============================
with tab5:
    cam = st.camera_input("Take Picture")

    if cam:
        st.image(cam)

        image = Image.open(cam)

        text = pytesseract.image_to_string(image)

        if not text:
            text = "Camera medical scan image"

        result = medical_ai(text)

        st.subheader("AI Camera Analysis")
        st.write(result)

        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "input": "camera scan",
            "type": "camera"
        })

# ===============================
# 📜 TAB 6 - HISTORY
# ===============================
with tab6:
    st.subheader("User History")

    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in reversed(st.session_state.history):
            st.write(h)
