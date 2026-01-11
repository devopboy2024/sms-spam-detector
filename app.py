import nltk
import psutil
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Download required nltk data (run once)
nltk.download('punkt')
nltk.download('stopwords')

ps=PorterStemmer()

# ================= TEXT PREPROCESSING =================
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)#tokenization
    # removing punctutaion by isalnum() method
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    # removing stopwords with punctuation
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # stemming : it convert word form into root form
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ================= STREAMLIT UI =================
st.title('Email/SMS Spam Classifier')

input_sms=st.text_input('Enter the message')

if st.button('predict'):
    # 1.preprocess
    transform_sms = transform_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.header("Spam Detected")
    else:
        st.header("Not Spam")





