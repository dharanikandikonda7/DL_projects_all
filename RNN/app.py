import streamlit as st
from collections import defaultdict, Counter
import re

st.set_page_config(
    page_title="RNN Next Word Predictor",
    page_icon="🔤"
)

st.title("🔤 RNN Next Word Predictor")

st.write(
    """
    This project demonstrates the concept behind Recurrent Neural Networks (RNNs):
    predicting the next word from a sequence of previous words.
    """
)

corpus = """
machine learning is transforming the world
machine learning helps solve complex problems
deep learning is a subset of machine learning
artificial intelligence powers modern applications
artificial intelligence and machine learning are related
data science uses machine learning techniques
neural networks are inspired by the human brain
recurrent neural networks process sequential data
rnn models are useful for language processing
language models predict the next word
streamlit makes deployment easy
python is widely used in data science
"""

words = re.findall(r'\w+', corpus.lower())

word_pairs = defaultdict(list)

for i in range(len(words) - 2):
    key = (words[i], words[i + 1])
    word_pairs[key].append(words[i + 2])

st.subheader("Enter Two Words")

text = st.text_input(
    "Example: machine learning"
)

if st.button("Predict Next Word"):

    tokens = re.findall(r'\w+', text.lower())

    if len(tokens) < 2:
        st.warning("Please enter at least two words.")
    else:

        key = (tokens[-2], tokens[-1])

        if key in word_pairs:

            predictions = Counter(word_pairs[key])

            st.success("Possible Next Words")

            for word, count in predictions.most_common(5):
                st.write(
                    f"• {word} ({count} occurrence)"
                )

        else:
            st.error(
                "No prediction available for this sequence."
            )

st.markdown("---")

st.subheader("How This Relates to RNN")

st.write("""
- RNNs process text one word at a time.
- They maintain information from previous words.
- They are commonly used for next-word prediction.
- Modern language models evolved from these ideas.
""")