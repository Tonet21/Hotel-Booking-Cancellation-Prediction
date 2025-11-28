import shap
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

model = load("boost.joblib")

df = pd.read_csv("cleaned_hotel_data.csv") 
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

X = pd.get_dummies(X, drop_first=True)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")
plt.savefig("summary_plot_bar.png")

