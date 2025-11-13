import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

##model = load(xgboost.joblib)
df = pd.read_csv("cleaned_hotel_data.csv") 
df = df.drop("is_canceled", axis=1)

st.title("Hotel Bookin Cancellation Predictor")

columns =list(df)
input= {}

import numpy as np
import streamlit as st

input_data = {}

for x in columns:
    # Determine if column is numeric

    is_numeric = np.issubdtype(df[x].dtype, np.number)
    
    # Choose appropriate placeholder
    placeholder = "Number" if is_numeric else "Text"
    
    # Get user input as text (Streamlit always returns strings from text_input)
    val = st.text_input(x.replace("_", " ").title(), placeholder=placeholder)
    
    # Validate and convert
    if val:
        if is_numeric:
            try:
                input_data[x] = float(val)
            except ValueError:
                st.error(f"❌ {x.replace("_", " ").title()} must be a number.")
        else:
            # Expect text, but reject pure numbers
            if val:
                input_data[x] = val
                
            else:
                st.error(f"❌ {val} must be text, not a number.")    

##check when the input must be text but it isn't
