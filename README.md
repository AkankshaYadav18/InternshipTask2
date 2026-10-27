# FAQ Chatbot

A simple FAQ chatbot built with Python, Streamlit, and scikit-learn.

## Features
- Chat-style web interface
- FAQ dataset stored in `faqs.json`
- TF-IDF text vectorization
- Cosine similarity to select the closest matching question
- Fallback response when no FAQ matches well

## Setup

Python 3.10 or newer is recommended.

1. Open a terminal in this folder.
2. (Optional) Create and activate a virtual environment.
3. Install libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Add more FAQs

Edit `faqs.json`. Add entries using this format:

```json
{
  "question": "Your question here?",
  "answer": "Your answer here."
}
```

Keep the whole file as a valid JSON list. Use your own tested questions and answers for your internship submission.

## How it works

1. Loads questions and answers from `faqs.json`.
2. Converts FAQ questions into TF-IDF vectors.
3. Converts the user's question into a vector.
4. Calculates cosine similarity between the user's question and each FAQ question.
5. Returns the answer for the closest match if it passes a similarity threshold.

This is a small demonstration, not a production-ready support system. The threshold may need adjustment for your dataset.
