import pandas as pd
import streamlit as st

# Reading training and testing datasets
datasetTrain = pd.read_csv('padi_produktifitas_training20 (1).csv', delimiter=';', encoding='utf-8')
datasetTest = pd.read_csv('padi_produktifitas_testing12 (1).csv', delimiter=';', encoding='utf-8')

# Cleaning function for both training and testing data
def clean_column(column):
    if column.dtype == 'object':
        column = column.str.replace('.', '', regex=False)
        column = column.str.replace(',', '.', regex=False)
        column = column.str.replace(' ', '', regex=False)
        column = pd.to_numeric(column, errors='coerce')
    return column

# Clean both datasets
for dataset in [datasetTrain, datasetTest]:
    dataset['luas_panen'] = clean_column(dataset['luas_panen'])
    dataset['produksi_padi'] = clean_column(dataset['produksi_padi'])
    dataset['hari_hujan'] = clean_column(dataset['hari_hujan'])
    dataset['curah_hujan'] = clean_column(dataset['curah_hujan'])
    dataset['luas_lahan'] = clean_column(dataset['luas_lahan'])
    dataset['tenaga_kerja'] = clean_column(dataset['tenaga_kerja'])
    dataset['jumlah_penduduk'] = clean_column(dataset['jumlah_penduduk'])

# Klasifikasi fungsi
def klasifikasi_luas_panen(nilai):
    if nilai > 25180.91:
        return 'Besar'
    elif 12318.25 <= nilai <= 25180.91:
        return 'Sedang'
    else:
        return 'Kecil'

def klasifikasi_produksi_padi(nilai):
    if nilai > 139929.40:
        return 'Banyak'
    elif 64838.4 <= nilai <= 139929.40:
        return 'Sedang'
    else:
        return 'Sedikit'

def klasifikasi_hari_hujan(nilai):
    return 'Tinggi' if nilai >= 16.1 else 'Rendah'

def klasifikasi_curah_hujan(nilai):
    return 'Tinggi' if nilai >= 246.8 else 'Rendah'

def klasifikasi_luas_lahan(nilai):
    if nilai > 20745:
        return 'Luas'
    elif 10866 <= nilai < 20745:
        return 'Sedang'
    else:
        return 'Kecil'

def klasifikasi_tenaga_kerja(nilai):
    if nilai > 131801:
        return 'Banyak'
    elif 78135 <= nilai <= 131801:
        return 'Sedang'
    else:
        return 'Sedikit'

def klasifikasi_jumlah_penduduk(nilai):
    if nilai > 463936:
        return 'Padat'
    else:
        return 'Tidak Padat'

# Encode Data function
def encode_luas(panen):
    if panen == 'Kecil':
        return 1
    elif panen == 'Sedang':
        return 2
    elif panen == 'Besar':
        return 3

def encode_produksi(produksi):
    if produksi == 'Sedikit':
        return 1
    elif produksi == 'Sedang':
        return 2
    elif produksi == 'Banyak':
        return 3

def encode_hari_hujan(hari_hujan):
    if hari_hujan == 'Rendah':
        return 1
    elif hari_hujan == 'Tinggi':
        return 2

def encode_curah_hujan(curah_hujan):
    if curah_hujan == 'Rendah':
        return 1
    elif curah_hujan == 'Tinggi':
        return 2

def encode_luas_lahan(lahan):
    if lahan == 'Kecil':
        return 1
    elif lahan == 'Sedang':
        return 2
    elif lahan == 'Luas':
        return 3

def encode_tenaga_kerja(tenaga_kerja):
    if tenaga_kerja == 'Sedikit':
        return 1
    elif tenaga_kerja == 'Sedang':
        return 2
    elif tenaga_kerja == 'Banyak':
        return 3

def encode_jumlah_penduduk(penduduk):
    if penduduk == 'Tidak Padat':
        return 1
    elif penduduk == 'Padat':
        return 2

def encode_label(label):
    if label == 'Rendah':
        return 0
    elif label == 'Tinggi':
        return 1

# Naive Bayes Prediction function
def predict_naive_bayes(data_baru, prior, likelihoods, features):
    posteriors = {}
    for label in prior.index:
        posterior = prior[label]
        for feature in features:
            value = data_baru.get(feature, None)
            if value is not None and value in likelihoods[feature][label]:
                posterior *= likelihoods[feature][label][value]
            else:
                posterior *= 0.0001
        posteriors[label] = posterior
    return posteriors

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
