import streamlit as st
import pandas as pd

# Load Data Testing
@st.cache
def load_testing_data():
    datasetTest = pd.read_csv('padi_produktifitas_testing12 (1).csv', delimiter=';', encoding='utf-8')
    datasetTest.drop(columns=['no', 'kabupaten/kota'], inplace=True)
    return datasetTest

st.title("Data Testing")
st.write("Halaman ini menampilkan data testing dan hasil prediksi produktivitas padi.")

datasetTest = load_testing_data()
st.dataframe(datasetTest)
