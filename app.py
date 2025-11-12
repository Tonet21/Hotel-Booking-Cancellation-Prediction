import streamlit as st
import pandas as pd
from joblib import load
import xgboost


##model = load(xgboost.joblib)
df = pd.read_csv("cleaned_hotel_data.csv") 
df = df.drop("is_canceled", axis=1)

st.title("Hotel Bookin Cancellation Predictor")

columns =list(df)
input= {}

for x in columns:
    input[x] = st.text_input(x.replace("_"," ").title(), placeholder= df[x].dtype)

##tell if type of input is incorrect and change name of model I guess
