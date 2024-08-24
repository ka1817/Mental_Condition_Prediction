import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your model, tokenizer, and label encoder
model = load_model('best_model.keras')
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the maximum sequence length
max_sequence_length = 100  # Same as used during training

# Define the Streamlit app
st.title('Mental Health Text Classification')
st.write('Enter a sentence to classify it into one of the mental health conditions.')

# User input
user_input = st.text_area("Enter your text:", "")

if st.button('Classify'):
    if user_input:
        # Tokenize and pad the user input
        user_sequences = tokenizer.texts_to_sequences([user_input])
        user_padded = pad_sequences(user_sequences, maxlen=max_sequence_length)
        
        # Predict
        predictions = model.predict(user_padded)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Show result
        st.write(f'Prediction: {predicted_class}')
    else:
        st.write("Please enter a sentence to classify.")
