import os
import PyPDF2
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
import streamlit as st

nltk.download('vader_lexicon')

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = "Positive +" if sentiment_scores['compound'] >= 0 else "Negative -"
    sentiment_html = f"<b>Sentiment:</b> {sentiment}<br><br>"
    scores_html = f"<b>Sentiment Scores:</b><br>Positive: {sentiment_scores['pos']:.2f}<br>Negative: {sentiment_scores['neg']:.2f}<br>Neutral: {sentiment_scores['neu']:.2f}<br>Compound: {sentiment_scores['compound']:.2f}"
    return sentiment_html, scores_html

# export GOOGLE_API_KEY="key here"

def generate_hiring_prediction(resume_text, job_description):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(f"Prediction for hiring: {resume_text} {job_description}")
    hiring_prediction = response.text.strip()
    return hiring_prediction

def main():
    st.image('test.png')

    st.title("Resume Analysis and Hiring Prediction")

    st.subheader("Upload Resume PDF")
    uploaded_file = st.file_uploader("Upload a resume PDF", type="pdf")
    
    job_description = st.text_area("Enter Job Description", height=200)
    
    if st.button("Analyze Resume") and (uploaded_file is not None or job_description.strip() != ""):
        if uploaded_file is not None:
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = job_description
        
        sentiment_html, scores_html = perform_sentiment_analysis(resume_text)
        hiring_prediction = generate_hiring_prediction(resume_text, job_description)

        st.subheader("Sentiment Analysis")
        st.markdown(sentiment_html, unsafe_allow_html=True)
        st.markdown(scores_html, unsafe_allow_html=True)
        st.write(hiring_prediction)

if __name__ == "__main__":
    main()
