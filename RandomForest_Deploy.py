import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):
    prediction = classifier.predict([[sepal_length1, sepal_width1, petal_length1, petal_width1]])
    return prediction


def main():
    st.title("Iris Flower Prediction")

    html_temp = """  
       <div style="background-color: #FFFF00; padding: 16px">  
       <h1 style="color: #000000; text-align: center;">Streamlit Iris Flower Classifier ML App</h1>  
       </div>  
       """

    st.markdown(html_temp, unsafe_allow_html=True)

    sepal_length1 = st.text_input("Sepal Length", "Type Here")
    sepal_width1 = st.text_input("Sepal Width", "Type Here")
    petal_length1 = st.text_input("Petal Length", "Type Here")
    petal_width1 = st.text_input("Petal Width", "Type Here")
    result = ""
    ans = ""

    if st.button("Predict"):
        result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)

        if result == 1:
            ans = "Iris Setosa"
        if result == 2:
            ans = "Iris Versicolor"
        if result == 3:
            ans = "Iris Virginica"

    st.write('The output of the above is', ans)


if __name__ == '__main__':
    main()
