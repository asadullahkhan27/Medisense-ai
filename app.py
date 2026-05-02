import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide"
)

# ===============================
# Model Load (Cache)
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
# Helpers
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
        return "https://source.unsplash.com/600x400/?cough,doctor"
    else:
        return "https://source.unsplash.com/600x400/?hospital"

def get_risk_badge(text):
    t = text.lower()
    if "emergency" in t:
        return "🔴 Emergency"
    elif "high" in t:
        return "🟠 High Risk"
    elif "medium" in t:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def analyze(name, age, gender, symptoms):

    prompt = f"""
You are MediSense AI, a Clinical Decision Support System.

Patient:
Name: {name}
Age: {age}
Gender: {gender}

Symptoms:
{symptoms}

Provide:
- Possible Conditions (max 3)
- Risk Level
- Specialist
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

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result

# ===============================
# UI
# ===============================
st.title("🏥 MediSense AI")
st.subheader("Clinical Decision Support System (CDSS)")

st.warning("⚠️ This tool is for educational purposes only.")

# Sidebar
st.sidebar.header("Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.markdown("---")
st.sidebar.info("Enter symptoms in main panel")

# Main Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Enter Symptoms")
    symptoms = st.text_area("Describe symptoms", height=150)

    analyze_btn = st.button("Analyze Symptoms")

with col2:
    st.subheader("📊 Results")

    if analyze_btn and symptoms:

        with st.spinner("Analyzing..."):
            result = analyze(name, age, gender, symptoms)

        # Risk Badge
        risk = get_risk_badge(result)
        st.markdown(f"### Risk Level: {risk}")

        # Image
        image_url = get_image(symptoms)
        st.image(image_url, use_column_width=True)

        # Report
        st.text_area("Medical Report", result, height=250)

        # Download
        st.download_button(
            label="Download Report",
            data=result,
            file_name="medical_report.txt",
            mime="text/plain"
        )

        # Save history
        st.session_state.history.append({
            "time": str(datetime.datetime.now()),
            "name": name,
            "symptoms": symptoms,
            "report": result
        })

# ===============================
# History Section
# ===============================
st.markdown("---")
st.subheader("📜 Patient History")

if st.button("Load History"):
    if not st.session_state.history:
        st.info("No history yet")
    else:
        for h in st.session_state.history[-5:]:
            st.markdown(f"""
**Time:** {h['time']}  
**Name:** {h['name']}  
**Symptoms:** {h['symptoms']}  
---  
""")

# ===============================
# About
# ===============================
st.markdown("---")
st.subheader("ℹ️ About")

st.markdown("""
MediSense AI is an AI-powered healthcare assistant built using Hugging Face Transformers.

**Features:**
- Symptom Analysis  
- Risk Classification  
- Medical Visualization  
- Downloadable Reports  
- Patient History  

**Tech Stack:**
- Python  
- Streamlit  
- Hugging Face  
- PyTorch  
""")