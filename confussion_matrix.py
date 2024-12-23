import pandas as pd
import streamlit as st
from dataTesting import datasetTest

# Menampilkan Confusion Matrix
matrix = confusion_matrix(datasetTest['produktivitas_padi'], datasetTest['Prediksi Produktivitas Padi'])
print(matrix)

# Menghitung Akurasi prediksi (Persentase)
def hitung_akurasi(datasetTest):
    # Bandingkan prediksi dengan label asli
    benar = sum(datasetTest['produktivitas_padi'] == datasetTest['Prediksi Produktivitas Padi'])
    salah = sum(datasetTest['produktivitas_padi'] != datasetTest['Prediksi Produktivitas Padi'])
    akurasi = benar / (benar + salah)
    return akurasi

# Hitung akurasi dalam bentuk persentase dan tampilkan
akurasi = hitung_akurasi(datasetTest) * 100
print(f"Akurasi model Naive Bayes: {akurasi:.2f}%")
