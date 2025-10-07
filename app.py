import streamlit as st
import os
from detector import load_model, predict_text, explain_text
import config
import joblib
from utils import clean_text

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("Paste an article or a headline and the model will tell you if it's likely **FAKE** or **REAL**.")

@st.cache_resource
def get_pipeline():
    try:
        pipeline = load_model(config.MODEL_PATH)
        return pipeline
    except Exception as e:
        return None

pipeline = get_pipeline()

with st.sidebar:
    st.header("Settings")
    st.write("Model path:")
    st.text(config.MODEL_PATH)
    if pipeline is None:
        st.warning("Model not found. Run `python train_model.py` then reload this app.")
    else:
        st.success("Model loaded ✅")
    st.write("---")
    st.markdown("Optional: Set `OPENAI_API_KEY` environment variable for AI explanations.")

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Paste article / headline here", height=300)
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    if uploaded_file is not None and not text_input.strip():
        text_input = uploaded_file.read().decode("utf-8")

    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Please paste text or upload a file")
        elif pipeline is None:
            st.error("Model not loaded. Run training script first.")
        else:
            label, conf = predict_text(text_input, pipeline)
            st.markdown(f"### Prediction: **{label}**")
            st.write(f"Confidence: **{conf*100:.2f}%**")
            # nice visual
            if label == "FAKE":
                st.error("The model predicts this article may be **FAKE**.")
            else:
                st.success("The model predicts this article is likely **REAL**.")
            # explanation
            st.write("---")
            st.subheader("Top contributing words")
            explanation = explain_text(text_input, pipeline, top_n=8)
            pos = explanation.get("top_positive", [])
            neg = explanation.get("top_negative", [])
            st.write("**Words that pushed towards FAKE:**")
            if pos:
                for w, s in pos:
                    st.write(f"- {w} (score {s:.4f})")
            else:
                st.write("No contributing words found.")
            st.write("**Words that pushed towards REAL:**")
            if neg:
                for w, s in neg:
                    st.write(f"- {w} (score {s:.4f})")
            else:
                st.write("No contributing words found.")

with col2:
    st.info("Tips & examples")
    st.write("- Try short headlines or long articles.")
    st.write("- If results look odd, retrain with more data / try tuning vectorizer.")
    st.write("---")
    st.write("Example headlines to try:")
    st.write("- `New study shows chocolate cures cancer`")
    st.write("- `Government announces new tax changes`")