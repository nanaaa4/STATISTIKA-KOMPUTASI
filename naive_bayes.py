import pandas as pd
import streamlit as st

# Fungsi Prediksi menggunakan Naive Bayes
def predict_naive_bayes(data_baru, prior, likelihoods, features):
    posteriors = {}  # Dictionary untuk menyimpan probabilitas posterior untuk setiap kelas

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
    
