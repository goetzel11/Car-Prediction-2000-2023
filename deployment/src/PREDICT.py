import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import joblib

Svm_model = joblib.load('./model/SVM_Model.pkl')

def run():
    # Membuat form
    with st.form('Data Ajuan'):
        CAR_ID = st.number_input('CAR ID:', value=0)
        Brand = st.selectbox('BRAND', ['Tesla', 'BMW', 'Audi', 'Ford', 'Honda', 'Mercedes', 'Toyota'])
        Year = st.number_input("Tahun Produksi", min_value=0, value=0)
        Engine_Size = st.number_input('Engine Size', min_value=0.0, max_value=20.0, value=1.0)
        Fuel_Type = st.selectbox('Fuel Type', ['Petrol', 'Electric', 'Diesel', 'Hybrid'])
        Transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
        Mileage = st.number_input('Millage', min_value=0, value=0)
        Condition = st.selectbox('Condition', ['New', 'Used', 'Like New'])
        Model = st.selectbox('Model', ['Model X', '5 Series', 'A4', 'Model Y', 'Mustang', 'Q7', 'Q5', 'Civic', 'Explorer', 'Model 3', 'Fiesta', 'X3', 'GLA', 'A3', 'X5','C-Class', 'E-Class', 'CR-V', 'Camry', 'Accord', 'GLC', 'Corolla', 'Fit', 'Model S', 'Prius', '3 Series', 'RAV4', 'Focus'])
        predict = st.form_submit_button('Predict')

    # Only predict when the form is submitted
    if predict:
        # Create a DataFrame with the input data
        data_inf = pd.DataFrame({
            'CAR ID': [CAR_ID],
            'Brand': [Brand],
            'Year': [Year],
            'Fuel Type': [Fuel_Type],
            'Engine Size': [Engine_Size],
            'Transmission': [Transmission],
            'Mileage': [Mileage],
            'Condition': [Condition],
            'Model': [Model]
        })
        
        # Make prediction

        predictions = Svm_model.predict(data_inf)
        st.success(f" Harga Mobil: {predictions[0]:.2f} USD")

if __name__ == '__main__':
    run()