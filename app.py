# import nltk
# import streamlit as st
# import pickle
# import string

# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize

# ps=PorterStemmer()
# STOP_WORDS=set(stopwords.words('english'))

# # ================= TEXT PREPROCESSING =================
# def transform_text(text):
#     text=text.lower()
#     text=nltk.word_tokenize(text)#tokenization
#     # removing punctutaion by isalnum() method
#     y=[]
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     # removing stopwords with punctuation
#     text=y[:]
#     y.clear()
#     for i in text:
#         if i not in STOP_WORDS('english') and i not in string.punctuation:
#             y.append(i)
#     # stemming : it convert word form into root form
#     text=y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)

# tfidf=pickle.load(open('tfidf.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# # ================= STREAMLIT UI =================
# st.title('Email/SMS Spam Classifier')

# input_sms=st.text_input('Enter the message')

# if st.button('predict'):
#     # 1.preprocess
#     transform_sms = transform_text(input_sms)
#     # 2.vectorize
#     vector_input = tfidf.transform([transform_sms])
#     # 3.predict
#     result = model.predict(vector_input)[0]
#     # 4.Display
#     if result == 1:
#         st.header("Spam Detected")
#     else:
#         st.header("Not Spam")



import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# ===================== NLTK DOWNLOAD (CLOUD SAFE) =====================
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# ===================== GLOBAL OBJECTS =====================
ps = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))

# ===================== TEXT PREPROCESSING =====================
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove punctuation & non-alphanumeric
    tokens = [i for i in tokens if i.isalnum()]

    # Remove stopwords
    tokens = [i for i in tokens if i not in STOP_WORDS and i not in string.punctuation]

    # Stemming
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# ===================== LOAD MODEL & TFIDF =====================
@st.cache_resource
def load_models():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_models()

# ===================== STREAMLIT UI =====================
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Not Spam")

