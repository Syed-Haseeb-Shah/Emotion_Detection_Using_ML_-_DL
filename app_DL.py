import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
from keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ========================loading the saved files==================================================

# Load the trained model
model = load_model('model1.h5')

# Load the LabelEncoder
with open('lb1.pkl', 'rb') as f:
    lb = pickle.load(f)

# Load vocabulary size and max length
with open('vocab_info.pkl', 'rb') as f:
    vocab_info = pickle.load(f)

vocab_size = vocab_info['vocab_size']
max_len = vocab_info['max_len']

# =========================defining functions======================================================

def sentence_cleaning(sentence):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    one_hot_word = [one_hot(input_text=text, n=vocab_size)]
    pad = pad_sequences(sequences=one_hot_word, maxlen=max_len, padding='pre')
    return pad

def predict_emotion(input_text):
    sentence = sentence_cleaning(input_text)
    result = lb.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba = np.max(model.predict(sentence))
    return result, proba

# ================================creating app=====================================================
# App
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy', 'Fear', 'Anger', 'Love', 'Sadness', 'Surprise']")
st.write("=================================================")

# Taking input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    predicted_emotion, label = predict_emotion(user_input)
    st.write("Predicted Emotion:", predicted_emotion)
    st.write("Probability:", label)
