import streamlit as st
import joblib

st.set_page_config(page_title="sentiment AI", page_icon="", layout="centered")

st.title("Review Sentiment Analyzer")
st.write("an end to end machine learning model trained on 1000 product, moview, and resturant reivews")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.joblib")

pipeline = load_model()

user_input = st.text_area(
    "enter a customer review to analyze: ",
    placeholder=("eg. the customer support was surprisingly quick and helpful"),
    height=100
)

if st.button("analize sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("please enter some text first!")
    else:
        probs = pipeline.predict_proba([user_input])[0]
        neg_prob = probs[0] * 100
        pos_prob = probs[1] * 100

        if pos_prob >= 50:
            st.success(f"positive ({pos_prob:.1f}% confidence)")
        else: 
            st.success(f"negative ({neg_prob:.1f}% confidence)")

        st.write("confidence breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("positive signal", f"{pos_prob:.1f}%")
            st.progress(pos_prob/100)
        with col2:
            st.metric("negitive signal", f"{neg_prob:.1f}%")
            st.progress(neg_prob/100)

    