import pandas as pd
import streamlit as st

# """Menghitung Probabolitas Prior dan Probabilitas Likelihood"""

# # Menghitung Probabilitas Prior
# prior = datasetTrain['produktivitas_padi'].value_counts(normalize=True)
# print("Probabilitas Prior:")
# print(prior)

# # Menghitung Likelihood
# print("\nProbabilitas Likelihood (P(fitur | Produktivitas Padi)):")
# features = ['luas_panen', 'produksi_padi', 'hari_hujan', 'curah_hujan', 'luas_lahan', 'tenaga_kerja', 'jumlah_penduduk']

# likelihoods = {}
# for feature in features:
#     likelihoods[feature] = {}
#     for label in datasetTrain['produktivitas_padi'].unique():
#         likelihood = datasetTrain[datasetTrain['produktivitas_padi'] == label][feature].value_counts(normalize=True)
#         likelihoods[feature][label] = likelihood

#         print(f"\nP(fitur = {feature} | Produktivitas Padi = {label}):")
#         print(likelihood)

# """Membuat Prediksi Naive Bayes"""

# Fungsi Prediksi menggunakan Naive Bayes
def predict_naive_bayes(data_baru, prior, likelihoods, features):
    posteriors = {} # Dictionary untuk menyimpan probabilitas posterior untuk setiap kelas

    # Loop melalui setiap kelas (misalnya 0 dan 1)
    for label in prior.index:
        posterior = prior[label]  # Mulai dengan prior
        for feature in features:
            value = data_baru.get(feature, None)  # Ambil nilai fitur dari data_baru
            if value is not None and value in likelihoods[feature][label]:
                posterior *= likelihoods[feature][label][value]
            else:
                posterior *= 0.0001  # Handling nilai yang tidak muncul
        posteriors[label] = posterior
    return posteriors
