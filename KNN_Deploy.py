import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

# Load the trained KNN model
with open('knn.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)


def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):
    try:
        sepal_length1 = float(sepal_length1)
        sepal_width1 = float(sepal_width1)
        petal_length1 = float(petal_length1)
        petal_width1 = float(petal_width1)
    except ValueError as e:
        raise ValueError("All input values must be numeric.") from e

    prediction = classifier.predict([[sepal_length1, sepal_width1, petal_length1, petal_width1]])
    return prediction


def main():
    st.title("Iris Flower Prediction with KNN")

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
        try:
            result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)
            if result[0] == 0:
                ans = "Iris Setosa"
            elif result[0] == 1:
                ans = "Iris Versicolor"
            elif result[0] == 2:
                ans = "Iris Virginica"
            else:
                ans = "Unknown Iris Species"
        except ValueError as e:
            st.error(f"Error: {e}")

    st.write('The output of the above is:', ans)


if __name__ == '__main__':
    main()
