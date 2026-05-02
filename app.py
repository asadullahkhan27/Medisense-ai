import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BlipProcessor,
    BlipForConditionalGeneration
)
from PIL import Image
import torch

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Hospital AI System", layout="wide")

st.title("🏥 Hospital-Grade AI Assistant System")

# ===============================
# LOAD TEXT MODEL (Bio/Medical style)
# ===============================
@st.cache_resource
def load_text_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, text_model = load_text_model()

# ===============================
# LOAD VISION MODEL (BLIP)
# ===============================
@st.cache_resource
def load_vision_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, vision_model = load_vision_model()

# ===============================
# 🧠 MEDICAL TEXT AI (BioGPT STYLE SIMULATION)
# ===============================
def medical_ai(text):
    prompt = f"""
You are a medical AI trained on clinical knowledge (PubMed style reasoning).

Patient input:
{text}

Return:
1. Possible Conditions
2. Severity Level
3. Recommended Action
4. Medical Advice
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = text_model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# 📷 VISION AI (BLIP)
# ===============================
def vision_ai(image):
    inputs = processor(image, return_tensors="pt")
    out = vision_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ===============================
# 👁 EYE DISEASE DETECTION (SIMULATED)
# ===============================
def eye_analysis(description):
    text = description.lower()

    if "red" in text or "swollen" in text:
        return "Possible Eye Infection / Conjunctivitis"
    if "yellow" in text:
        return "Possible Jaundice Indicator"
    return "No clear eye disease detected"

# ===============================
# 👅 TONGUE ANALYSIS (SIMULATED CLINICAL RULES)
# ===============================
def tongue_analysis(description):
    text = description.lower()

    if "white" in text:
        return "Possible dehydration or fungal infection"
    if "red" in text:
        return "Possible fever or inflammation"
    return "Normal / No abnormal signs detected"

# ===============================
# UI TABS
# ===============================
tab1, tab2, tab3 = st.tabs([
    "🧠 Medical AI",
    "📷 Vision AI",
    "🏥 Full Health Scan"
])

# ===============================
# 🧠 TAB 1 - MEDICAL AI
# ===============================
with tab1:
    text = st.text_area("Enter Symptoms")

    if st.button("Analyze"):
        result = medical_ai(text)
        st.write(result)

# ===============================
# 📷 TAB 2 - VISION AI
# ===============================
with tab2:
    image_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

    if image_file:
        image = Image.open(image_file)
        st.image(image)

        caption = vision_ai(image)

        st.subheader("AI Vision Description")
        st.write(caption)

        st.subheader("Eye Analysis")
        st.write(eye_analysis(caption))

        st.subheader("Tongue Analysis")
        st.write(tongue_analysis(caption))

# ===============================
# 🏥 TAB 3 - FULL SYSTEM
# ===============================
with tab3:
    st.subheader("Complete Hospital AI Scan")

    text_input = st.text_area("Symptoms Input")
    image_input = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if st.button("Run Full Analysis"):

        # TEXT AI
        text_result = medical_ai(text_input)

        st.subheader("🧠 Medical AI Result")
        st.write(text_result)

        # IMAGE AI
        if image_input:
            image = Image.open(image_input)
            caption = vision_ai(image)

            st.subheader("📷 Vision AI Result")
            st.write(caption)

            st.subheader("👁 Eye Analysis")
            st.write(eye_analysis(caption))

            st.subheader("👅 Tongue Analysis")
            st.write(tongue_analysis(caption))
