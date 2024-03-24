import streamlit as st
import requests
import pyowm
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
owm = pyowm.OWM('11081b639d8ada3e97fc695bcf6ddb20')
from PIL import Image
import time
navigation_menu = ["Home", "Weather","Crop Recommendation","Fertilizer"]
selected = st.sidebar.selectbox("Navigation", navigation_menu)
if selected=='Home':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:#33ccff;'>Agriculture</h1>", unsafe_allow_html=True)
if selected=='Weather':
        st.markdown(f"<h1 style='text-align: center; color:skyblue;'>Weather</h1>", unsafe_allow_html=True)
        id = st.text_input("Enter City")
        try:
            mgr = owm.weather_manager()
            observation = mgr.weather_at_place(id)
            weather = observation.weather
            t1 = weather.temperature('celsius')['temp']
            h1 = weather.humidity
            w1 = weather.wind()
            p1=weather.pressure['press']
            num_weekdays = 5
            count_weekdays = 0
            weekday_names = []
            now = time.time()
            now1 = time.localtime()
            us_date = time.strftime("%m/%d/%Y", now1)
            while count_weekdays < num_weekdays:
                now += 86400
                local_time = time.localtime(now)
                weekday = local_time.tm_wday
                wn = time.strftime("%a", local_time)
                if count_weekdays!=5:
                    count_weekdays += 1
                    weekday_names.append(time.strftime("%a", local_time))
            col1, col2,col3,col4= st.columns([2,8,5,3])
            col1.image('download (1).png', width=75)
            with col4:
                pass
            with col2:
                st.markdown(f"<h4 style='color:red;'>{t1}°C</h4>", unsafe_allow_html=True)
            col1,col2,col3,col4= st.columns([5,8,5,5])
            with col1:
                st.markdown(f"<p>{'Humidity :  '}{h1}%</p>", unsafe_allow_html=True)
            with col1:
                st.markdown(f"<p>{'Pressure :  '}{' '}{p1}hPa</p>", unsafe_allow_html=True)
            with col1:
                st.markdown(f"<p>{'Wind Speed:  '}{w1['speed']}hPa</p>", unsafe_allow_html=True)  
            col1, col2,col3,col4,col5= st.columns([4,4,4,4,4])
            forecaster = mgr.forecast_at_place(id, '3h', limit=40)

            c=0
            l=[]
            for weather in forecaster.forecast:
                temperature = weather.temperature('celsius')['temp']
                c+=1
                if c==8 or c==16 or c==24 or c==32 or c==40:
                    l.append(temperature)
            with col1:
                st.markdown(f"<h4 style='color:#EE82EE	';>{weekday_names[0]}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{l[0]}°C</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<h4 style='color:blue';>{weekday_names[1]}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{l[1]}°C</p>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<h4 style='color:green';>{weekday_names[2]}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{l[2]}°C</p>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<h4 style='color:orange';>{weekday_names[3]}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{l[3]}°C</p>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"<h4 style='color:red';>{weekday_names[4]}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{l[4]}°C</p>", unsafe_allow_html=True)
        except:
            pass
if selected=='Crop Recommendation':
    st.markdown(f"<h1 style='text-align: center; color:red;'>Crop Recomendation</h1>", unsafe_allow_html=True)
    df=pd.read_csv("Crop_recommendation.csv")
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    labels = df['label']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    RF = RandomForestClassifier(n_estimators=25, random_state=42)
    RF.fit(Xtrain,Ytrain)
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        a=st.number_input('Enter N')
    with col2:
        b=st.number_input('Enter P')
    with col3:
        c1=st.number_input('Enter K')
    col1, col2,col3,col4= st.columns([5,5,5,5])
    with col1:
        d=st.number_input('Temperature °C')
    with col2:
        e=st.number_input('Humidity %')
    with col3:
        f=st.number_input('pH')
    with col4:
        g=st.number_input('Rainfall mm')
    data = np.array([[a,b,c1,d,e,f,g]])
    prediction = RF.predict(data)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        if prediction[0]=='apple':
            st.image("images/apple.jpg")
        if prediction[0]=='banana':
            st.image("images/banana.jpg")
        if prediction[0]=='blackgram':
            st.image("images/blackgram.jpg")
        if prediction[0]=='chickpea':
            st.image("images/chickpea.jpg")
        if prediction[0]=='coconut':
            st.image("images/coconut.jpg")
        if prediction[0]=='coffee':
            st.image("images/coffee.jpg")
        if prediction[0]=='cotton':
            st.image("images/cotton.jpg")
        if prediction[0]=='grapes':
            st.image("images/grapes.jpg")
        if prediction[0]=='jute':
            st.image("images/jute.jpg")
        if prediction[0]=='kidneybeans':
            st.image("images/kidneybeans.jpg")
        if prediction[0]=='lentil':
            st.image("images/lentil.jpg")
        if prediction[0]=='maize':
            st.image("images/maize.jpg")
        if prediction[0]=='mango':
            st.image("images/mango.jpg")
        if prediction[0]=='mothbeans':
            st.image("images/mothbeans.jpg")
        if prediction[0]=='mungbean':
            st.image("images/mungbeans.jpg")
        if prediction[0]=='muskmelon':
            st.image("images/muskmelon.jpg")
        if prediction[0]=='orange':
            st.image("images/orange.jpg")
        if prediction[0]=='papaya':
            st.image("images/papaya.jpg")
        if prediction[0]=='pomegranate':
            st.image("images/pomegranate.jpg")
        if prediction[0]=='pigeonpeas':
            st.image("images/pigeonpeas.jpg")
        if prediction[0]=='rice':
            st.image("images/rice.jpg")
        if prediction[0]=='watermelon':
            st.image("images/watermelon.jpg")
    with col3:
        st.write(' ')
if selected=='Fertilizer':
    st.markdown(f"<h1 style='text-align: center; color:blue;'>Fertilizer Prediction</h1>", unsafe_allow_html=True)
    data = pd.read_csv('Fertilizer Prediction.csv')
    data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'},inplace=True)
    from sklearn.preprocessing import LabelEncoder
    encode_soil = LabelEncoder()
    data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)
    Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
    Soil_Type = Soil_Type.set_index('Original')
    encode_crop = LabelEncoder()
    data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)
    Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
    Crop_Type = Crop_Type.set_index('Original')
    encode_ferti = LabelEncoder()
    data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)
    Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
    Fertilizer = Fertilizer.set_index('Original')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer',axis=1),data.Fertilizer,test_size=0.2,random_state=1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    soil=['Black','Clayey','Loamy','Red','Sandy']
    crop=['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat']
    fert=['10-26-26','14-35-14','17-17-17','20-20','28-28','DAP','Urea']
    rand = RandomForestClassifier(n_estimators=30,random_state=42)
    pred_rand = rand.fit(x_train,y_train).predict(x_test)
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        a=st.number_input('Temperature °C')
    with col2:
        b=st.number_input('Humidity %')
    with col3:
        c=st.number_input('Moisture')
    col1,col2= st.columns([5,5])
    with col1:
        d=st.selectbox('Soil Type',('Black','Clayey','Loamy','Red','Sandy'))
    with col2:
        e=st.selectbox('Crop Type',('Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat'))
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        f=st.number_input('Enter N')
    with col2:
        g=st.number_input('Enter P')
    with col3:
        h=st.number_input('Enter K')
    data = np.array([[a,b,c,soil.index(d),crop.index(e),f,g,h]])
    prediction = rand.predict(data)
    res=fert[prediction[0]]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        if res=='10-26-26':
            st.image("images/10-26-26.jpg")
        if res=='14-35-14':
            st.image("images/14-35-14.jpg")
        if res=='17-17-17':
            st.image("images/17-17-17.jpg")
        if res=='20-20':
            st.image("images/20-20.jpg")
        if res=='28-28':
            st.image("images/28-28.jpg")
        if res=='DAP':
            st.image("images/DAP.jpg")
        if res=='Urea':
            st.image("images/Urea.jpg")
    with col3:
        st.write(' ')

    