import streamlit as st
import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
loaded_model = pickle.load(open('C:\\Users\\user1\\Documents\\Sugandh\\machine learning\\trained_model (1).sav', 'rb'))

tfidf_vectorizer = pickle.load(open('C:\\Users\\user1\\Documents\\Sugandh\\machine learning\\tfidf_vectorizer.pkl', 'rb'))

# Streamlit app title
st.title('Twitter Sentiment Analysis')

# Text input for the user
user_input = st.text_input('Enter text for analysis')

# Button to trigger prediction
if st.button('Predict'):
    # Preprocess the user input
    stemmed_content = re.sub('[^a-zA-Z]', ' ', user_input)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    port_stem = PorterStemmer()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    input_vectorized = tfidf_vectorizer.transform([stemmed_content])

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_vectorized)

    # Display the prediction result
    if prediction[0] == 0:
        st.write('Negative Tweet')
    elif prediction[0] == 1:
        st.write('Positive Tweet')
    else:
        st.write('Unable to determine sentiment.')
