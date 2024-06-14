import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
model = joblib.load('trained_model.joblib')

# Создание заголовка
st.title('Классификация месяцев года')

# Вводим температуру и владность воздуха:
temp = st.number_input('Температура: ')
air = st.number_input('Влажность: ')

input_data = pd.DataFrame([ (temp, air) ])

input_data_scaled = scaler.transform(input_data)

predict_data_enc = model.predict(input_data_scaled)

output = label_encoder.inverse_transform(predict_data_enc)

# Отображаем предсказанный месяц:
st.write('Месяц: ', output)
