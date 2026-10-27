import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).parent / "faqs.json"
SIMILARITY_THRESHOLD = 0.18

@st.cache_data
def load_faqs():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

faqs = load_faqs()
questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

# Convert FAQ questions into numerical TF-IDF vectors.
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
question_vectors = vectorizer.fit_transform(questions)

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 FAQ Chatbot")
st.write("Ask a question about college, internships, or programming.")
st.caption("This demo uses TF-IDF and cosine similarity to find the closest FAQ.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_question}
    )

    user_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(user_vector, question_vectors).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score >= SIMILARITY_THRESHOLD:
        response = answers[best_index]
        response += f"\n\n*Matched FAQ: {questions[best_index]}*"
    else:
        response = (
            "Sorry, I couldn't find a reliable answer in my FAQ database. "
            "Please try rephrasing your question or add this question to faqs.json."
        )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
    st.rerun()

with st.sidebar:
    st.header("Example questions")
    for question in questions[:8]:
        st.write(f"• {question}")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
