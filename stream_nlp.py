import pickle
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
with open('model_email.sav', 'rb') as f:
    model_email = pickle.load(f)

tfidf = TfidfVectorizer
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_features_tf-idf.sav", "rb"))))

#judl hakaman

st.title('prediksi email penipuan')

clean_text = st.text_input('masukan teks email')

fraud_detection = ''

if st.button('hasil deteksi'):
    predict_fraud = model_email.predict(loaded_vec.fit_transform([clean_text]))

    if predict_fraud == 'spam':
        fraud_detection = 'Email Spam'
    else:
        fraud_detection = 'Email Normal'


st.success(fraud_detection)