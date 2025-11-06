from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# 1️⃣ Load your already cleaned dataset
df = pd.read_csv("cleaned_hotel_data.csv")
df = df.drop(columns=['reservation_status', 'reservation_status_date'])


# 2️⃣ Define target and features
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

# 3️⃣ One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 4️⃣ Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    eval_metric="error",
    use_label_encoder=False,
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
