import pandas as pd
import streamlit as st
from dataTraining import klasifikasi_luas_panen, klasifikasi_produksi_padi, klasifikasi_hari_hujan, klasifikasi_curah_hujan, klasifikasi_luas_lahan, klasifikasi_tenaga_kerja, klasifikasi_jumlah_penduduk 
from dataTraining import encode_luas, encode_produksi, encode_hari_hujan, encode_curah_hujan, encode_luas_lahan, encode_tenaga_kerja, encode_jumlah_penduduk
from dataTraining import prior, likelihoods
from naive_bayes import predict_naive_bayes

# Streamlit User Input and Prediction
st.title('Prediksi Produktivitas Padi dengan Naive Bayes')

user_input = {}
features = [
    'luas_panen (ha)', 'produksi_padi (ton/ha)', 'hari_hujan (hari)', 
    'curah_hujan (mm)', 'luas_lahan (ha)', 'tenaga_kerja (orang)', 'jumlah_penduduk (orang)'
]

st.subheader("Masukkan Data untuk Prediksi")
for feature in features:
    value = st.number_input(f"Masukkan {feature}:", min_value=0.0)
    user_input[feature.split(' ')[0]] = value  # Only store feature name without unit

# Klasifikasi data input user
user_input['luas_panen'] = klasifikasi_luas_panen(user_input['luas_panen'])
user_input['produksi_padi'] = klasifikasi_produksi_padi(user_input['produksi_padi'])
user_input['hari_hujan'] = klasifikasi_hari_hujan(user_input['hari_hujan'])
user_input['curah_hujan'] = klasifikasi_curah_hujan(user_input['curah_hujan'])
user_input['luas_lahan'] = klasifikasi_luas_lahan(user_input['luas_lahan'])
user_input['tenaga_kerja'] = klasifikasi_tenaga_kerja(user_input['tenaga_kerja'])
user_input['jumlah_penduduk'] = klasifikasi_jumlah_penduduk(user_input['jumlah_penduduk'])

# Encode input data
user_input['luas_panen'] = encode_luas(user_input['luas_panen'])
user_input['produksi_padi'] = encode_produksi(user_input['produksi_padi'])
user_input['hari_hujan'] = encode_hari_hujan(user_input['hari_hujan'])
user_input['curah_hujan'] = encode_curah_hujan(user_input['curah_hujan'])
user_input['luas_lahan'] = encode_luas_lahan(user_input['luas_lahan'])
user_input['tenaga_kerja'] = encode_tenaga_kerja(user_input['tenaga_kerja'])
user_input['jumlah_penduduk'] = encode_jumlah_penduduk(user_input['jumlah_penduduk'])

# Calculate prior and likelihoods (from training data)
prior = datasetTrain['produktivitas_padi'].value_counts(normalize=True)
likelihoods = {}
for feature in features:
    likelihoods[feature] = {}
    for label in datasetTrain['produktivitas_padi'].unique():
        likelihood = datasetTrain[datasetTrain['produktivitas_padi'] == label][feature].value_counts(normalize=True)
        likelihoods[feature][label] = likelihood

# Predict the productivity
posteriors = predict_naive_bayes(user_input, prior, likelihoods, features)
prediksi = max(posteriors, key=posteriors.get)

# Display the result
prediksi_label = 'Tinggi' if prediksi == 1 else 'Rendah'
st.subheader(f"Prediksi Produktivitas Padi: {prediksi_label}")
