import streamlit as st
from src.data_processing import extract_features
from src.model import hybrid_model

def app():
    st.title('Song Genre Classifier')
    audio_file = st.file_uploader('Upload a song')
    if audio_file is not None:
        features = extract_features(audio_file)
        prediction = hybrid_model.predict(features)
        st.write(f'The predicted genre is: {prediction}')
