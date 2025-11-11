import streamlit as st
import pandas as pd
from joblib import load
import xgboost


model = load(xgboost.joblib)

st.title("Hotel Bookin Cancellation Predictor")
