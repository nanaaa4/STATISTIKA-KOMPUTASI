import pandas as pd
import streamlit as st
from dataTesting import datasetTest

# Fungsi untuk menghitung Confusion Matrix
def confusion_matrix(label_true, label_pred):
    """
    Menghitung confusion matrix untuk prediksi.
    """
    matrix = {
        'TP': 0,  # True Positive
        'TN': 0,  # True Negative
        'FP': 0,  # False Positive
        'FN': 0   # False Negative
    }

    # Iterasi data untuk menghitung TP, TN, FP, FN
    for i in range(len(label_true)):
        if label_true[i] == 0 and label_pred[i] == 0:
            matrix['TN'] += 1
        elif label_true[i] == 0 and label_pred[i] == 1:
            matrix['FP'] += 1
        elif label_true[i] == 1 and label_pred[i] == 0:
            matrix['FN'] += 1
        elif label_true[i] == 1 and label_pred[i] == 1:
            matrix['TP'] += 1

    return matrix

# Fungsi untuk menghitung akurasi
def hitung_akurasi(datasetTest):
    """
    Menghitung akurasi model.
    """
    benar = sum(datasetTest['produktivitas_padi'] == datasetTest['Prediksi Produktivitas Padi'])
    salah = sum(datasetTest['produktivitas_padi'] != datasetTest['Prediksi Produktivitas Padi'])
    akurasi = benar / (benar + salah)
    return akurasi

# Menampilkan judul
st.title("Confusion Matrix dan Akurasi Model")

# Menampilkan dataset testing
st.subheader("Dataset Testing")
st.write(datasetTest)

# Menghitung Confusion Matrix
matrix = confusion_matrix(
    datasetTest['produktivitas_padi'].tolist(),
    datasetTest['Prediksi Produktivitas Padi'].tolist()
)

# Menampilkan Confusion Matrix
st.subheader("Confusion Matrix")
st.write(f"True Positive (TP): {matrix['TP']}")
st.write(f"True Negative (TN): {matrix['TN']}")
st.write(f"False Positive (FP): {matrix['FP']}")
st.write(f"False Negative (FN): {matrix['FN']}")

# Menghitung Akurasi
akurasi = hitung_akurasi(datasetTest) * 100

# Menampilkan Akurasi
st.subheader("Akurasi Model")
st.write(f"Akurasi model Naive Bayes: {akurasi:.2f}%")
