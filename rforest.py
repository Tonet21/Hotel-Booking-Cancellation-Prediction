import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load your already cleaned dataset
df = pd.read_csv("cleaned_hotel_data.csv")

# 2️⃣ Define target and features
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

# 3️⃣ One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 4️⃣ Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# 1️⃣ Train the Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# 2️⃣ Evaluate performance
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]



print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 3️⃣ Save the trained model
dump(rf, "rforest.joblib")
print("\n✅ Model saved as rforest.joblib")

# Model Performance Summary at 200:
# --------------------------
# Accuracy: 0.84
# ROC AUC: 0.90
#
# Class 0 (Not Canceled):
#   Precision: 0.86  -> Of all predicted non-cancellations, 86% were correct.
#   Recall:    0.93  -> The model correctly identified 93% of real non-cancellations.
#   F1-Score:  0.90
#
# Class 1 (Canceled):
#   Precision: 0.77  -> Of all predicted cancellations, 77% were correct.
#   Recall:    0.61  -> The model correctly identified 61% of actual cancellations.
#   F1-Score:  0.68
#
# Interpretation:
# The model performs well overall (84% accuracy, high ROC AUC of 0.90),
# showing strong discrimination between classes. It predicts non-cancellations
# more accurately than cancellations (higher recall for Class 0),
# suggesting it might be slightly conservative in flagging bookings as canceled.


## same at 300