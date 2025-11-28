import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import re




model = load("boost.joblib")

df = pd.read_csv("cleaned_hotel_data.csv") 
df = df.drop("is_canceled", axis=1)

st.title("Hotel Bookin Cancellation Predictor")

columns =list(df)
input= {}

for x in columns:


    is_numeric = np.issubdtype(df[x].dtype, np.number)
    

    placeholder = "Number" if is_numeric else "Text"
    
    val = st.text_input(x.replace("_", " ").title(), placeholder=placeholder)
    
    if val:
        if is_numeric:
            try:
                input[x] = float(val)
            except ValueError:
                st.error(f"{x.replace("_", " ").title()} must be a number.")
        else:
            # Expect text, but reject pure numbers
            if re.fullmatch(r"[A-Za-zÀ-ÿ\s]+", val):
                input[x] = val
                
            else:
                st.error(f"{x.replace("_", " ").title()} must be text.") 

with st.form(key='my_form_to_submit'):

    predict_button = st.form_submit_button(label='Predict')

if predict_button:

    input_df = pd.DataFrame([input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)   
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    result = "Canceled" if prediction == 1 else "Not Canceled"
    st.subheader(f"Prediction: {result} with a probabilty of {probs[prediction]*100}%")
