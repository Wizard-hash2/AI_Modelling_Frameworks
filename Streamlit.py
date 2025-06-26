import streamlit as st
from transformers import pipeline

# 1. Load Hugging Face sentiment pipeline
sentiment = pipeline("sentiment-analysis")

# 2. Streamlit UI components
st.title("Sentiment Analysis App")
user_text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    # 3. Run sentiment model
    results = sentiment(user_text)
    # 4. Display results
    for res in results:
        label = res["label"]
        score = res["score"]
        st.write(f"**{label}** with confidence {score:.2f}")
