import joblib
import streamlit as st
import os
import time


BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "sentiment_pipeline.pkl")

with open(model_path, "rb") as f:
    model = joblib.load(f)
    
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

# 🎨 Custom CSS
st.markdown("""
<style>

/* 🔴 Background gradient (Netflix vibe) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #0f0f0f, #1a0000);
    color: white;
}

/* 🔴 Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #e50914, #ff4d4d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 🔴 Subtitle */
.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}

/* 🔴 Input box */
textarea {
    background-color: #111 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #333 !important;
    padding: 12px !important;
}

/* 🔴 Button */
.stButton>button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white;
    border: none;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 15px rgba(229, 9, 20, 0.6);
}

/* 🔴 Result cards */
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}

/* Positive glow */
.positive {
    background: linear-gradient(90deg, #00c853, #2e7d32);
    box-shadow: 0px 0px 15px rgba(0,255,0,0.4);
}

/* Negative glow */
.negative {
    background: linear-gradient(90deg, #b71c1c, #e53935);
    box-shadow: 0px 0px 15px rgba(255,0,0,0.4);
}

/* 🔴 Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #e50914, #ff4d4d);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* 🌫️ Page fade-in */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stAppViewContainer"] {
    animation: fadeIn 0.8s ease-in-out;
}

/* 🎬 Result animation */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-box {
    animation: slideUp 0.6s ease;
    transition: all 0.3s ease;
}

/* ✨ Glow pulse */
@keyframes glow {
    0% { box-shadow: 0 0 5px rgba(229,9,20,0.3); }
    50% { box-shadow: 0 0 20px rgba(229,9,20,0.7); }
    100% { box-shadow: 0 0 5px rgba(229,9,20,0.3); }
}

/* 🔥 Button animation */
.stButton>button {
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.03);
    animation: glow 1.2s infinite;
}

/* ⌨️ Textarea focus animation */
textarea:focus {
    border: 1px solid #e50914 !important;
    box-shadow: 0 0 10px rgba(229,9,20,0.5) !important;
    transition: 0.3s;
}

/* 📊 Smooth progress */
.stProgress > div > div {
    transition: width 0.6s ease-in-out;
}

</style>
""", unsafe_allow_html=True)
# 🧱 Layout
st.markdown('<div class="title">🎬 Movie Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze movie reviews with AI</div>', unsafe_allow_html=True)

with st.container():
    text = st.text_area("", placeholder="✍️ Type your movie review here...", height=150)

    if st.button("Analyze Review 🚀", key="premium_btn"):
        if text.strip() == "":
            st.warning("⚠️ Please enter a review")
        else:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(1)

                prediction = model.predict([text])[0]
                proba = model.predict_proba([text])[0]
                confidence = max(proba)

            # 🎯 Result UI
           
            if prediction == "positive":
               st.markdown( '<div class="result-box positive">😊 Positive Sentiment</div>',unsafe_allow_html=True)
                 
            else:
              st.markdown('<div class="result-box negative">😡 Negative Sentiment</div>',unsafe_allow_html=True)
                  

            # 📊 Confidence
            st.progress(float(confidence))
            st.write(f"Confidence: **{round(confidence * 100, 2)}%**")

    
