import pandas as pd
import streamlit as st
# from naive_bayes import data_baru, prior, likelihoods, features

datasetTrain = pd.read_csv('padi_produktifitas_training20 (1).csv', delimiter=';', encoding='utf-8')

st.title('Informasi Dataset')
st.write("### Info Dataset")
st.write(datasetTrain.info())

st.write("### 5 Baris Pertama Dataset")
st.write(datasetTrain.head())

# Fungsi untuk membersihkan kolom
def clean_columnTrain(column):
    if column.dtype == 'object':
        # Menghilangkan tanda titik, tanda koma, dan spasi
        column = column.str.replace('.', '', regex=False)
        column = column.str.replace(',', '.', regex=False)
        column = column.str.replace(' ', '', regex=False)
        # Mengonversi ke numerik, menangani kesalahan dengan mengatur ke NaN
        column = pd.to_numeric(column, errors='coerce')
    return column

# Streamlit UI
st.title('Proses Pembersihan Data')
st.write("### Dataset Training Awal")
st.write(datasetTrain.head())

# Menghapus kolom yang tidak diperlukan
datasetTrain.drop(columns=['no', 'kabupaten/kota'], inplace=True)

st.write("### Dataset Training Setelah Penghapusan Kolom")
st.write(datasetTrain.head())

# Cleaning data di kolom tertentu
datasetTrain['luas_panen'] = clean_columnTrain(datasetTrain['luas_panen'])
datasetTrain['produksi_padi'] = clean_columnTrain(datasetTrain['produksi_padi'])
datasetTrain['hari_hujan'] = clean_columnTrain(datasetTrain['hari_hujan'])
datasetTrain['curah_hujan'] = clean_columnTrain(datasetTrain['curah_hujan'])
datasetTrain['luas_lahan'] = clean_columnTrain(datasetTrain['luas_lahan'])
datasetTrain['tenaga_kerja'] = clean_columnTrain(datasetTrain['tenaga_kerja'])
datasetTrain['jumlah_penduduk'] = clean_columnTrain(datasetTrain['jumlah_penduduk'])

# Menampilkan tipe data dan 5 baris pertama setelah pembersihan
st.write("### Data Training Setelah Pembersihan")
st.write(datasetTrain.dtypes)
st.write(datasetTrain.head())


# Fungsi klasifikasi untuk data
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

# Fungsi encoding untuk data
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

# Streamlit UI
st.title('Klasifikasi dan Encoding Data (Hanya datasetTrain)')
st.write("### Dataset Training Awal")
st.write(datasetTrain.head())

# Klasifikasi data pada datasetTrain
datasetTrain['luas_panen'] = datasetTrain['luas_panen'].apply(klasifikasi_luas_panen)
datasetTrain['produksi_padi'] = datasetTrain['produksi_padi'].apply(klasifikasi_produksi_padi)
datasetTrain['hari_hujan'] = datasetTrain['hari_hujan'].apply(klasifikasi_hari_hujan)
datasetTrain['curah_hujan'] = datasetTrain['curah_hujan'].apply(klasifikasi_curah_hujan)
datasetTrain['luas_lahan'] = datasetTrain['luas_lahan'].apply(klasifikasi_luas_lahan)
datasetTrain['tenaga_kerja'] = datasetTrain['tenaga_kerja'].apply(klasifikasi_tenaga_kerja)
datasetTrain['jumlah_penduduk'] = datasetTrain['jumlah_penduduk'].apply(klasifikasi_jumlah_penduduk)

# Menampilkan hasil klasifikasi
st.write("### Data Training Setelah Klasifikasi")
st.write(datasetTrain.head())

# Encoding data pada datasetTrain
datasetTrain['luas_panen'] = datasetTrain['luas_panen'].apply(encode_luas)
datasetTrain['produksi_padi'] = datasetTrain['produksi_padi'].apply(encode_produksi)
datasetTrain['hari_hujan'] = datasetTrain['hari_hujan'].apply(encode_hari_hujan)
datasetTrain['curah_hujan'] = datasetTrain['curah_hujan'].apply(encode_curah_hujan)
datasetTrain['luas_lahan'] = datasetTrain['luas_lahan'].apply(encode_luas_lahan)
datasetTrain['tenaga_kerja'] = datasetTrain['tenaga_kerja'].apply(encode_tenaga_kerja)
datasetTrain['jumlah_penduduk'] = datasetTrain['jumlah_penduduk'].apply(encode_jumlah_penduduk)
datasetTrain['produktivitas_padi'] = datasetTrain['produktivitas_padi'].apply(encode_label)

# Menampilkan hasil encoding
st.write("### Data Training Setelah Encoding")
st.write(datasetTrain.head())

# Mengecek apakah ada data kosong
st.write("### Mengecek Data Kosong di Dataset Training")
st.write(datasetTrain.isnull().sum())

# Menghitung Probabilitas Prior
prior = datasetTrain['produktivitas_padi'].value_counts(normalize=True)

# Menghitung Likelihood
features = ['luas_panen', 'produksi_padi', 'hari_hujan', 'curah_hujan', 'luas_lahan', 'tenaga_kerja', 'jumlah_penduduk']

likelihoods = {}
for feature in features:
    likelihoods[feature] = {}
    for label in datasetTrain['produktivitas_padi'].unique():
        likelihood = datasetTrain[datasetTrain['produktivitas_padi']] == [label][feature].value_counts(normalize=True)
        likelihoods[feature][label] = likelihood.get(label, "Data tidak tersedia")

# Streamlit UI
st.title('Probabilitas Prior dan Likelihood (Naive Bayes)')

# Menampilkan Probabilitas Prior
st.write("### Probabilitas Prior:")
st.write(prior)

# Menampilkan Probabilitas Likelihood
st.write("### Probabilitas Likelihood (P(fitur | Produktivitas Padi)):")

for feature in features:
    st.write(f"**Feature: {feature}**")
    for label in datasetTrain['produktivitas_padi'].unique():
        st.write(f"\n**P({feature} | Produktivitas Padi = {label}):**")
        st.write(likelihoods[feature].get(label, "Data tidak tersedia"))

# Menampilkan button untuk mengecek hasil
if st.button('Tampilkan Hasil Probabilitas'):
    st.write("### Hasil Probabilitas sudah ditampilkan di atas.")
