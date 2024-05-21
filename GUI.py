import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

liver_disease = pickle.load(open('liverModel.sav','rb'))


liver_data = pd.read_csv("Liver Patient Dataset (LPD)_train.csv", encoding='unicode_escape')

data1 = liver_data.head(10)

with st.sidebar:
    
    selected = option_menu('Liver Prediction System',
                          
                          [ 'Home',
                            'Liver Disease Prediction'
                           ],
                          icons=['house','person'],
                          default_index=0)


if (selected == 'Home'):

   st.title("Liver Disease Prediction Using Deep Learning Model")

   st.image("medicalProfessional.png")
   
   st.write("- Liver diseases are two of the leading causes of death worldwide. Early diagnosis and intervention can improve the chances of survival for patients with these diseases.")
   st.write("- The aim of this research is to develop a novel machine learning algorithm that can accurately predict the likelihood of both heart disease and liver disease in individuals.")
   st.write("- The primary challenge is to create an effective hybrid model that can efficiently process medical data and provide reliable predictions for these two distinct health conditions.")
   
   
        
# Liver Disease Prediction Page
if (selected == 'Liver Disease Prediction'):
    
    # page title
    st.title('Liver Disease Prediction using Deep Learning Model')
    
    st.write("We highly reccomend you to first get to know about the insights of the information asked here.")     
    if st.checkbox("Insights"):
        st.write("The chosen database includes the following structure, having different measures for different types of attributes.")
        st.write("- Total Bilirubin: total bilirubin in adults is 0.1 to 1.2 milligrams per deciliter (mg/dL).")
        st.write("- Alkphos Alkaline Phosphotase: The normal range is 44 to 147 international units per liter (IU/L) or 0.73 to 2.45 microkatal per liter (µkat/L).")
        st.write("- Sgpt Alamine Aminotransferase: SGPT (serum glutamic-pyruvic transaminase) or ALT (alanine aminotransferase) levels in the blood is typically 7–56 units per liter (U/L).")
        st.write("- Sgot Aspartate Aminotransferase: The normal range of serum glutamic-oxaloacetic transaminase (SGOT) aspartate, also known as aspartate aminotransferase (AST), is typically between 8 and 45 units per liter")
        st.write("- Total Protien: The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")
        st.write("- ALB Albumin: The normal range for albumin in an adult's blood is 3.4–5.4 grams per deciliter (g/dL), or 34–54 grams per liter (g/L).")
        st.write("- A/G Ratio Albumin and Globulin Ratio: The normal range for albumin-to-globulin (A/G) ratio is between 1.1 and 2.5, but this can vary by lab.")
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', 30.00, 75.00, step=1.00)
        
    with col2:
        bilirubin = st.number_input('Total Bilirubin', 0.01, 5.20, step=0.01)
        
    with col3:
        Alkphos = st.number_input('Alkphos Alkaline Phosphotase', 22.00, 347.00, step= 1.00)
        
    with col1:
        sgpt = st.number_input('Sgpt Alamine Aminotransferase', 3.00, 76.00, step=1.00)
        
    with col2:
        sgpot = st.number_input('Sgot Aspartate Aminotransferase', 3.00, 85.00, step=1.00)
        
    with col3:
        protien = st.number_input('Total Protiens', 2.00, 15.30, step=0.01)
        
    with col1:
        ALB = st.number_input('ALB Albumin', 14.00, 84.00, step=1.00)                
            
    with col2:
        ratio = st.number_input('A/G Ratio Albumin and Globulin Ratio', 0.10, 5.50, step=0.01)
        
    with col3:
        female = 0.00
        male  = 0.00  
        
        gender = st.selectbox("Gender", ["male", "female"])
        
        if gender == 'female':
            female = 1.00
        elif gender == 'male':
            male = 1.00 
            
        
    # Create a button and store the click status in a variable
    clicked = st.button("Predict")

    # If the button is clicked, perform some action (replace with your logic)
    if clicked:
      prediction = liver_disease.predict([[age,bilirubin,Alkphos,sgpt,sgpot,protien,ALB,ratio,female,male]])
      
      if prediction == 1:
          st.write("You have chances of having a liver disease")
      else:
          st.write("You do not posses chances of having any liver problem.")
          st.write("Althogh we suggest you to maintain a stress free and healthy lifestyle.")
    st.write(" ")      
    st.write(" ")  
    st.write(" ")  
    
    st.image("liver.jpg")       
    
