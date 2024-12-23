import pandas as pd
import streamlit as st
from dataTraining import (
    klasifikasi_luas_panen, klasifikasi_produksi_padi, klasifikasi_hari_hujan, 
    klasifikasi_curah_hujan, klasifikasi_luas_lahan, klasifikasi_tenaga_kerja, klasifikasi_jumlah_penduduk,
    encode_luas, encode_produksi, encode_hari_hujan, encode_curah_hujan, 
    encode_luas_lahan, encode_tenaga_kerja, encode_jumlah_penduduk,
    datasetTrain  # Pastikan datasetTrain diimpor dari dataTraining
)
from naive_bayes import predict_naive_bayes

# Streamlit User Input and Prediction
st.title('Prediksi Produktivitas Padi dengan Naive Bayes')

# List of features
features = [
    'luas_panen (ha)', 'produksi_padi (ton/ha)', 'hari_hujan (hari)', 
    'curah_hujan (mm)', 'luas_lahan (ha)', 'tenaga_kerja (orang)', 'jumlah_penduduk (orang)'
]

# Input data dari pengguna
st.subheader("Masukkan Data untuk Prediksi")
user_input = {}
for feature in features:
    value = st.number_input(f"Masukkan {feature}:", min_value=0.0)
    user_input[feature.split(' ')[0]] = value  # Menyimpan hanya nama fitur tanpa satuan

# Proses klasifikasi
user_input['luas_panen'] = klasifikasi_luas_panen(user_input['luas_panen'])
user_input['produksi_padi'] = klasifikasi_produksi_padi(user_input['produksi_padi'])
user_input['hari_hujan'] = klasifikasi_hari_hujan(user_input['hari_hujan'])
user_input['curah_hujan'] = klasifikasi_curah_hujan(user_input['curah_hujan'])
user_input['luas_lahan'] = klasifikasi_luas_lahan(user_input['luas_lahan'])
user_input['tenaga_kerja'] = klasifikasi_tenaga_kerja(user_input['tenaga_kerja'])
user_input['jumlah_penduduk'] = klasifikasi_jumlah_penduduk(user_input['jumlah_penduduk'])

# Encode data
user_input['luas_panen'] = encode_luas(user_input['luas_panen'])
user_input['produksi_padi'] = encode_produksi(user_input['produksi_padi'])
user_input['hari_hujan'] = encode_hari_hujan(user_input['hari_hujan'])
user_input['curah_hujan'] = encode_curah_hujan(user_input['curah_hujan'])
user_input['luas_lahan'] = encode_luas_lahan(user_input['luas_lahan'])
user_input['tenaga_kerja'] = encode_tenaga_kerja(user_input['tenaga_kerja'])
user_input['jumlah_penduduk'] = encode_jumlah_penduduk(user_input['jumlah_penduduk'])

# Mengubah input pengguna menjadi DataFrame
user_df = pd.DataFrame([user_input])

# Hitung prior dan likelihoods dari dataset training
prior = datasetTrain['produktivitas_padi'].value_counts(normalize=True)
likelihoods = {}
for feature in user_input.keys():
    likelihoods[feature] = {}
    for label in datasetTrain['produktivitas_padi'].unique():
        likelihood = datasetTrain[datasetTrain['produktivitas_padi'] == label][feature].value_counts(normalize=True)
        likelihoods[feature][label] = likelihood

# Prediksi produktivitas
posteriors = predict_naive_bayes(user_input, prior, likelihoods, list(user_input.keys()))
prediksi = max(posteriors, key=posteriors.get)

# Menampilkan hasil prediksi
prediksi_label = 'Tinggi' if prediksi == 1 else 'Rendah'
st.subheader(f"Prediksi Produktivitas Padi: {prediksi_label}")

# Menampilkan DataFrame input pengguna
st.write("Data Input:")
st.dataframe(user_df)
