from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump


df = pd.read_csv("cleaned_hotel_data.csv")

X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    reg_alpha=0.3, ## l2
    reg_lambda=1.0, ## l1
    eval_metric=["error", "auc", "logloss"],
    use_label_encoder=False,
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)


y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



dump(model, "boost.joblib")
print("\nModel saved as boost.joblib")