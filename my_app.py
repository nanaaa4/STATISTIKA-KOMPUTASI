import streamlit as st

# load page
training_page = st.Page("./dataTraining.py", title="Data Training")
testing_page = st.Page("./dataTesting.py", title="Data Testing")
confussion_matrix = st.Page("./confussion_matrix.py", title="Akurasi Naive Bayes")
form_predict = st.Page("./form_predict.py", title="Prediksi Produktivitas Padi")

# Initialize the page
pg = st.navigation([training_page, testing_page, confussion_matrix, form_predict])
st.set_page_config(page_title="PREDIKSI PRODUKTIVITAS PADI")

pg.run() 
