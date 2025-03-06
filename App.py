#First Import all the required Libraries
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


#Second Load the trained model that is model.h5
model = tf.keras.models.load_model('model.h5')

#third load all the scalar, one hot encoder and label encoder files
model = load_model('model.h5')

with open ('one_hot_encode_geo.pkl','rb') as file:
    label_encoder_geo = pickle.load(file)

with open ('data_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open ('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

#Going to Streamlit app (avoiding HTML etc)
st.title('Customer Churn Prediction')