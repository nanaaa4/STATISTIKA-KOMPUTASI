import pandas as pd
import streamlit as st
from dataTraining import prior, likelihoods
from naive_bayes import predict_naive_bayes

datasetTest = pd.read_csv('padi_produktifitas_testing12 (1).csv', delimiter=';', encoding='utf-8')

st.title('Informasi Dataset')
st.write("### 5 Baris Pertama Dataset")
st.write(datasetTest.head())

# Fungsi untuk membersihkan kolom
def clean_columnTest(column):
    if column.dtype == 'object':
        # Menghilangkan tanda titik, tanda koma, dan spasi
        column = column.str.replace('.', '', regex=False)
        column = column.str.replace(',', '.', regex=False)
        column = column.str.replace(' ', '', regex=False)
        # Mengonversi ke numerik, menangani kesalahan dengan mengatur ke NaN
        column = pd.to_numeric(column, errors='coerce')
    return column

st.title('Proses Pembersihan Data')
st.write("### Dataset Testing Awal")
st.write(datasetTest.head())

# Menghapus kolom yang tidak diperlukan
datasetTest.drop(columns=['no', 'kabupaten/kota'], inplace=True)

st.write("### Dataset Testing Setelah Penghapusan Kolom")
st.write(datasetTest.head())

# Cleaning data di kolom tertentu
datasetTest['luas_panen'] = clean_columnTest(datasetTest['luas_panen'])
datasetTest['produksi_padi'] = clean_columnTest(datasetTest['produksi_padi'])
datasetTest['hari_hujan'] = clean_columnTest(datasetTest['hari_hujan'])
datasetTest['curah_hujan'] = clean_columnTest(datasetTest['curah_hujan'])
datasetTest['luas_lahan'] = clean_columnTest(datasetTest['luas_lahan'])
datasetTest['tenaga_kerja'] = clean_columnTest(datasetTest['tenaga_kerja'])
datasetTest['jumlah_penduduk'] = clean_columnTest(datasetTest['jumlah_penduduk'])

# Menampilkan tipe data dan 5 baris pertama setelah pembersihan
st.write("### Data Testing Setelah Pembersihan")
st.write(datasetTest.dtypes)
st.write(datasetTest.head())

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


st.title('Klasifikasi dan Encoding Data (Hanya datasetTest)')
st.write("### Dataset Testing Awal")
st.write(datasetTest.head())

# Klasifikasi data pada datasetTest
datasetTest['luas_panen'] = datasetTest['luas_panen'].apply(klasifikasi_luas_panen)
datasetTest['produksi_padi'] = datasetTest['produksi_padi'].apply(klasifikasi_produksi_padi)
datasetTest['hari_hujan'] = datasetTest['hari_hujan'].apply(klasifikasi_hari_hujan)
datasetTest['curah_hujan'] = datasetTest['curah_hujan'].apply(klasifikasi_curah_hujan)
datasetTest['luas_lahan'] = datasetTest['luas_lahan'].apply(klasifikasi_luas_lahan)
datasetTest['tenaga_kerja'] = datasetTest['tenaga_kerja'].apply(klasifikasi_tenaga_kerja)
datasetTest['jumlah_penduduk'] = datasetTest['jumlah_penduduk'].apply(klasifikasi_jumlah_penduduk)

# Menampilkan hasil klasifikasi
st.write("### Data Testing Setelah Klasifikasi")
st.write(datasetTest.head())

# Encoding data pada datasetTest
datasetTest['luas_panen'] = datasetTest['luas_panen'].apply(encode_luas)
datasetTest['produksi_padi'] = datasetTest['produksi_padi'].apply(encode_produksi)
datasetTest['hari_hujan'] = datasetTest['hari_hujan'].apply(encode_hari_hujan)
datasetTest['curah_hujan'] = datasetTest['curah_hujan'].apply(encode_curah_hujan)
datasetTest['luas_lahan'] = datasetTest['luas_lahan'].apply(encode_luas_lahan)
datasetTest['tenaga_kerja'] = datasetTest['tenaga_kerja'].apply(encode_tenaga_kerja)
datasetTest['jumlah_penduduk'] = datasetTest['jumlah_penduduk'].apply(encode_jumlah_penduduk)
datasetTest['produktivitas_padi'] = datasetTest['produktivitas_padi'].apply(encode_label)

# Menampilkan hasil encoding
st.write("### Data Testing Setelah Encoding")
st.write(datasetTest.head())

# Mengecek apakah ada data kosong
st.write("### Mengecek Data Kosong di Dataset Testing")
st.write(datasetTest.isnull().sum())

# Fitur yang digunakan dalam prediksi
features = ['luas_panen', 'produksi_padi', 'hari_hujan', 'curah_hujan', 'luas_lahan', 'tenaga_kerja', 'jumlah_penduduk']

# Prediksi untuk seluruh datasetTesting
prediksi_all = []
for i, row in datasetTest.iterrows():
    # Ambil data baris sebagai dictionary
    data_baru = row[features].to_dict()

    # Panggil fungsi prediksi untuk data_baru
    posteriors = predict_naive_bayes(data_baru, prior, likelihoods, features)

    # Ambil label dengan probabilitas tertinggi sebagai prediksi
    prediksi = max(posteriors, key=posteriors.get)

    # Simpan hasil prediksi
    prediksi_all.append(prediksi)

# Menambahkan hasil prediksi ke datasetTesting
datasetTest['Prediksi Produktivitas Padi'] = prediksi_all

# Menampilkan hasil prediksi
st.dataframe(datasetTest.head(12))
