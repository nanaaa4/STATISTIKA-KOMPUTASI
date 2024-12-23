import pandas as pd
import streamlit as st
from dataTesting import datasetTest

def confusion_matrix(label_true, label_pred):
    """
        label_true: List atau array NumPy berisi label sebenarnya.
        label_pred: List atau array NumPy berisi label prediksi.

        TP (True Positive)
        TN (True Negative)
        FP (False Positive)
        FN (False Negative)
    """

    # Inisialisasi confusion matrix label produktivitas padi
    matrix = {
        'TP': 0,  # Tepat memprediksi “Tinggi” (produktivitas tinggi)
        'TN': 0,  # Tepat memprediksi “Rendah” (produktivitas rendah)
        'FP': 0,  # Salah memprediksi "Tinggi" padahal sebenarnya "Rendah".
        'FN': 0   # Salah memprediksi "Rendah" padahal sebenarnya "Tinggi".
    }

    # Iterasi melalui data
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
