import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title = 'MLpruebaWebApp', layout = 'wide')

pipeline_path = 'artefacts/preprocessor/preprocessor.pkl'
model_path = 'artefacts/model/svc.pkl'
encoder_path = 'artefacts/preprocessor/encoder.pkl'

with open(pipeline_path, 'rb') as file1:
    print(file1.read(100))

try:
    pipeline = joblib.load(pipeline_path)
    print('Pipeline cargada')
except Exception as e:
    print(f'Error cargando el pipeline {e}')

with open(model_path, 'rb') as file2:
    print(file2.read(100))

try:
    model = joblib.load(model_path)
    print('Modelo cargado')
except Exception as e:
    print(f'Error cargando el Modelo {e}')

with open(encoder_path, 'rb') as file3:
    print(file3.read(100))

try:
    encoder = joblib.load(encoder_path)
    print('Encoder cargado')
except Exception as e:
    print(f'Error cargando el Encoder {e}')

#######################################################

st.title('WebpruebaApp de Machine Learning')
st.header('Ingreso de los datos')

col1, col2, col3 = st.columns(3)

with col1:
    battery_power = st.slider('poder de la bateria (mAh)', min_value = 500, max_value = 2000, value = 800)
    clock_speed = st.slider('Velocidad del CPU', min_value = 0.5, max_value = 3.0, value = 1.5)
    fc = st.slider('Camara frontal (MP)', min_value = 0, max_value= 19, step = 1)
    int_memory = st.slider('Memoria interna (GB)', min_value = 2, max_value= 64, step = 34)
    px_heigth = st.slider('Resolucion de la pantalla (Altura en px)', min_value = 100, max_value = 2000 ) 

with col2:
    m_dep = st.slider('Grosor del telefono (cm)', min_value = 0.1, max_value = 1.0)
    mobile_wt = st.slider('Peso del dispositivo (g)', min_value = 100, max_value = 2000)

with col3:
    ram = st.slider('Memoria Ram (MB)', min_value = 0, max_value = 100)

    blue = st.selectbox('Tiene Bluetooth?', [0,1])