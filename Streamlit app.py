import pandas as pd
import streamlit as st 
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#load the model 
model = load_model('model.h5')

with open('onehot_encoder.pkl','rb') as file:
    onehot_encoder= pickle.load(file) 

with open('label_encoder.pkl','rb') as file:
    label_encoder= pickle.load(file) 

with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file) 


#Streamlit app 
st.title("churn prediction App")

#user input 
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Add geography columns
# One-hot encode geography
geo_encoded = onehot_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Concatenate the geo-encoded data with the rest of the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_value = prediction[0][0]
if prediction_value >0.5:
    st.markdown("<h3 style='color: red;'>1: Will exit the bank</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: green;'>2: Will not exit the bank</h3>", unsafe_allow_html=True)



