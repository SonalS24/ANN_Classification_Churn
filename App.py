import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model.h5')

# Load encoders and scaler
with open('one_hot_encode_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('data_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# Streamlit App Title
st.title('Customer Churn Prediction')

# Input fields
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is Active member', [0, 1])

# Encode categorical variables
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())

# Ensure column names match exactly with scaler
input_data = pd.DataFrame({
    'CreditScore': [credit_score], 
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Merge encoded geography data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure feature order matches the scaler
input_data = input_data[scalar.feature_names_in_]

# Scale the input data
input_data_scaled = scalar.transform(input_data)

# Predict churn probability
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display result
if prediction_prob > 0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('The customer is not likely to Churn')
