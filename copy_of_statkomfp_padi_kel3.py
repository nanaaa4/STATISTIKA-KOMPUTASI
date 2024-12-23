import streamlit as st
import pandas as pd

# Page navigation setup
PAGES = {
    "Data Loading & Cleaning": "page_1",
    "Classification & Encoding": "page_2",
    "Model Training": "page_3",
    "Prediction & Evaluation": "page_4",
    "User Input Prediction": "page_5",
}

# Display the sidebar to select pages
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Data Loading & Cleaning (Page 1)
if selection == "Data Loading & Cleaning":
    st.title("Data Loading & Cleaning")
    st.write("### Step 1: Load the Dataset")

    # Load dataset
    datasetTrain = pd.read_csv('padi_produktifitas_training20 (1).csv', delimiter=';', encoding='utf-8')
    datasetTest = pd.read_csv('padi_produktifitas_testing12 (1).csv', delimiter=';', encoding='utf-8')

    st.write("#### Training Data")
    st.dataframe(datasetTrain.head())

    st.write("#### Testing Data")
    st.dataframe(datasetTest.head())

    # Clean the dataset
    datasetTrain.drop(columns=['no', 'kabupaten/kota'], inplace=True)
    datasetTest.drop(columns=['no', 'kabupaten/kota'], inplace=True)

    st.write("#### Cleaned Training Data")
    st.dataframe(datasetTrain.head())

    st.write("#### Cleaned Testing Data")
    st.dataframe(datasetTest.head())

# Classification & Encoding (Page 2)
# Classification & Encoding (Page 2)
elif selection == "Classification & Encoding":
    st.title("Classification & Encoding")
    st.write("### Step 2: Classify Data into Categories")

    # Classification and encoding functions (define them as in your code)
    def klasifikasi_luas_panen(nilai):
        if nilai > 25180.91:
            return 'Besar'
        elif 12318.25 <= nilai <= 25180.91:
            return 'Sedang'
        else:
            return 'Kecil'

    # Apply classification on both training and testing datasets
    datasetTrain['luas_panen'] = datasetTrain['luas_panen'].apply(klasifikasi_luas_panen)
    datasetTest['luas_panen'] = datasetTest['luas_panen'].apply(klasifikasi_luas_panen)

    st.write("#### Classified Training Data")
    st.dataframe(datasetTrain.head())

    st.write("#### Classified Testing Data")
    st.dataframe(datasetTest.head())

    # Encode the data (like 'encode_luas')
    def encode_luas(panen):
        if panen == 'Kecil':
            return 1
        elif panen == 'Sedang':
            return 2
        elif panen == 'Besar':
            return 3
        return 0

    datasetTrain['luas_panen'] = datasetTrain['luas_panen'].apply(encode_luas)
    datasetTest['luas_panen'] = datasetTest['luas_panen'].apply(encode_luas)

    st.write("#### Encoded Training Data")
    st.dataframe(datasetTrain.head())

    st.write("#### Encoded Testing Data")
    st.dataframe(datasetTest.head())

# Model Training (Page 3)
# Model Training (Page 3)
elif selection == "Model Training":
    st.title("Model Training")
    st.write("### Step 3: Train the Naive Bayes Model")

    # Calculate prior probabilities
    prior = datasetTrain['produktivitas_padi'].value_counts(normalize=True)
    st.write("#### Prior Probabilities")
    st.write(prior)

    # Calculate likelihoods for each feature
    features = ['luas_panen', 'produksi_padi', 'hari_hujan', 'curah_hujan', 'luas_lahan', 'tenaga_kerja', 'jumlah_penduduk']
    likelihoods = {feature: {} for feature in features}

    for feature in features:
        for label in datasetTrain['produktivitas_padi'].unique():
            likelihood = datasetTrain[datasetTrain['produktivitas_padi'] == label][feature].value_counts(normalize=True)
            likelihoods[feature][label] = likelihood
            st.write(f"#### P({feature} | Produktivitas Padi = {label})")
            st.write(likelihood)

    st.write("#### Likelihoods Calculated Successfully")

# Prediction & Evaluation (Page 4)
# Prediction & Evaluation (Page 4)
elif selection == "Prediction & Evaluation":
    st.title("Prediction & Evaluation")
    st.write("### Step 4: Make Predictions & Evaluate the Model")

    # Prediction function
    def predict_naive_bayes(data_baru, prior, likelihoods, features):
        posteriors = {}  # Dictionary for storing posterior probabilities

        for label in prior.index:
            posterior = prior[label]  # Start with prior probability
            for feature in features:
                value = data_baru.get(feature, None)
                if value is not None and value in likelihoods[feature][label]:
                    posterior *= likelihoods[feature][label][value]
                else:
                    posterior *= 0.0001  # Handling unseen values
            posteriors[label] = posterior
        return posteriors

    # Example prediction for the testing data
    prediksi_all = []
    for i, row in datasetTest.iterrows():
        data_baru = row[features].to_dict()
        posteriors = predict_naive_bayes(data_baru, prior, likelihoods, features)
        prediksi = max(posteriors, key=posteriors.get)
        prediksi_all.append(prediksi)

    # Adding predictions to the dataset
    datasetTest['Prediksi Produktivitas Padi'] = prediksi_all

    st.write("#### Predictions for Testing Data")
    st.dataframe(datasetTest[['luas_panen', 'produksi_padi', 'Prediksi Produktivitas Padi']].head())

    # Evaluate the accuracy
    correct_predictions = sum(datasetTest['produktivitas_padi'] == datasetTest['Prediksi Produktivitas Padi'])
    accuracy = correct_predictions / len(datasetTest)
    st.write(f"#### Accuracy: {accuracy * 100:.2f}%")

# User Input Prediction (Page 5)
elif selection == "User Input Prediction":
    st.title("User Input Prediction")
    st.write("### Step 5: Predict Using User Input")

    # Input fields for prediction
    luas_panen = st.number_input("Luas Panen (ha):", min_value=0.0, step=0.1)
    produksi_padi = st.number_input("Produksi Padi (ton/ha):", min_value=0.0, step=0.1)
    hari_hujan = st.number_input("Hari Hujan (hari):", min_value=0.0, step=0.1)
    curah_hujan = st.number_input("Curah Hujan (mm):", min_value=0.0, step=0.1)
    luas_lahan = st.number_input("Luas Lahan (ha):", min_value=0.0, step=0.1)
    tenaga_kerja = st.number_input("Tenaga Kerja (orang):", min_value=0, step=1)
    jumlah_penduduk = st.number_input("Jumlah Penduduk (orang):", min_value=0, step=1)

    # Button to trigger prediction
    if st.button("Prediksi"):
        user_input = {
            'luas_panen': luas_panen,
            'produksi_padi': produksi_padi,
            'hari_hujan': hari_hujan,
            'curah_hujan': curah_hujan,
            'luas_lahan': luas_lahan,
            'tenaga_kerja': tenaga_kerja,
            'jumlah_penduduk': jumlah_penduduk,
        }

        # Classify and encode input values
        # (Use the classification and encoding functions as before)
        user_input['luas_panen'] = encode_luas(user_input['luas_panen'])
        user_input['produksi_padi'] = encode_produksi(user_input['produksi_padi'])
        user_input['hari_hujan'] = encode_hari_hujan(user_input['hari_hujan'])
        user_input['curah_hujan'] = encode_curah_hujan(user_input['curah_hujan'])
        user_input['luas_lahan'] = encode_luas_lahan(user_input['luas_lahan'])
        user_input['tenaga_kerja'] = encode_tenaga_kerja(user_input['tenaga_kerja'])
        user_input['jumlah_penduduk'] = encode_jumlah_penduduk(user_input['jumlah_penduduk'])

        # Make prediction
        posteriors = predict_naive_bayes(user_input, prior, likelihoods, features)
        prediction = max(posteriors, key=posteriors.get)

        st.write(f"### Predicted Produktivitas Padi: {'Tinggi' if prediction == 1 else 'Rendah'}")

