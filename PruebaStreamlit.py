import streamlit as st
import pandas as pd
import joblib



st.set_page_config(page_title = 'MLpruebaWebApp', layout = 'wide')

pipeline_path = 'artefacts/preprocessor/preprocessor.pkl'
model_path = 'artefacts/model/svm.pkl'
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
    n_cores = st.slider('Numero de nucleos del CPU', min_value = 1, max_value = 10)
    pc = st.slider('camara trasera ( MP)', min_value = 0, max_value = 19, step = 1)
    px = st.slider('Resolucion de la pantalla (Ancho en px)', min_value = 100, max_value = 2000)

with col3:
    ram = st.slider('Memoria Ram (MB)', min_value = 256, max_value = 4000)
    sc_h = st.slider('Altura de la pantalla (cm)', min_value = 5, max_value = 19)
    sc_w = st.slider('Ancho de la pantalla (cm)', min_value = 0, max_value = 18)
    talk_time = st.slider('Duracion de la bateria bajo uso constante (Hrs)', min_value = 2, max_value = 20)

st.divider() #Agrega una linea divisoria
col4, col5, col6 = st.columns(3)

with col4:
    blue = st.selectbox('Tiene Bluetooth?', options=[0,1])
    three_g = st.selectbox('Tiene 3G?', options=[0,1])

with col5:
    dual_sim = st.selectbox('Tiene Dual SIM?', options= [0,1])
    touch_screen = st.selectbox('Tiene pantalla tactil?', options=[0,1])

with col6:
    wifi = st.selectbox('Tiene Wifi?', options= [0,1])
    four_g = st.selectbox('4G', options= [0,1])

st.divider()  # Agrega una linea divisoria

if st.button('Predecir'):
    input_data = pd.DataFrame(
        {
            'battery_power': [battery_power],
            'blue': [blue],
            'clock_speed': [clock_speed],
            'dual_sim': [dual_sim],
            'fc': [fc],
            'four_g': [four_g],
            'int_memory': [int_memory],
            'm_dep': [m_dep],
            'mobile_wt': [mobile_wt],
            'n_cores': [n_cores],
            'pc': [pc],
            'px_height': [px_heigth],
            'px_width': [px],
            'ram': [ram],
            'mobile_wt': [mobile_wt],
            'sc_h': [sc_h],
            'sc_w': [sc_w],
            'talk_time': [talk_time],
            'three_g' : [three_g],
            'touch_screen': [touch_screen],
            'wifi': [wifi],
            
       }
    )

    st.dataframe(input_data)
    pipelined_data = pipeline.transform(input_data)
    prediction = model.predict(pipelined_data)

    if prediction[0] == 0:
        st.success('Bajo costo')
    elif prediction[0] == 1:
        st.success('Costo medio')
    elif prediction[0] == 2:
        st.success('Alto costo')
    elif prediction[0] == 3:
        st.success('Costo muy alto')
