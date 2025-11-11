from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

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
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    reg_alpha=0.1, ## l2
    reg_lambda=1.0, ## l1
    eval_metric=["error", "auc", "logloss"],
    use_label_encoder=False,
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



dump(model, "xgboost.joblib")
print("\n✅ Model saved as xgboost.joblib")

# ✅ XGBoost Model Performance Summary
# Training Error: 0.147   | Validation Error: 0.162
# Training AUC: 0.918     | Validation AUC: 0.901
# Training Log Loss: 0.322 | Validation Log Loss: 0.345
#
# F1 Score (overall): 0.68
#
# Classification Report:
#   Class 0 (Not Canceled): Precision = 0.87 | Recall = 0.92 | F1 = 0.89
#   Class 1 (Canceled):     Precision = 0.74 | Recall = 0.63 | F1 = 0.68
#
# Overall Accuracy: 0.84
# Macro Avg F1: 0.79 | Weighted Avg F1: 0.83
# ROC AUC (Validation): 0.90
#
# 🔍 Interpretation:
# The model performs strongly overall (84% accuracy) with high AUC (0.90),
# meaning it distinguishes well between canceled and non-canceled bookings.
# Precision is good for both classes, but recall for cancellations (0.63)
# suggests some canceled bookings are missed — a tradeoff that could be
# adjusted by tuning the decision threshold or using class weighting.

## worse with learning rate 0.01

# ✅ XGBoost Model Performance Summary (Improved) better with learning rate 0.5
# Training Error: 0.091   | Validation Error: 0.157
# Training AUC: 0.967     | Validation AUC: 0.905
# Training Log Loss: 0.222 | Validation Log Loss: 0.342
#
# Classification Report:
#   Class 0 (Not Canceled): Precision = 0.88 | Recall = 0.91 | F1 = 0.89
#   Class 1 (Canceled):     Precision = 0.74 | Recall = 0.66 | F1 = 0.70
#
# Overall Accuracy: 0.84
# Macro Avg F1: 0.80 | Weighted Avg F1: 0.84
# ROC AUC (Validation): 0.905
#
# 🔍 Interpretation:
# The model maintains strong overall performance (84% accuracy) and an excellent
# ROC AUC of ~0.91, showing good discrimination between canceled and non-canceled
# bookings. Compared to the previous version:
#   - Slightly higher AUC (0.905 vs 0.901) → better separation power
#   - Improved recall for cancellations (0.66 vs 0.63) → fewer missed cancellations
#   - F1 for cancellations rose from 0.68 to 0.70 → better balance overall
# This indicates **a small but meaningful improvement** in generalization and
# sensitivity to canceled bookings, with no sign of overfitting.

# XGBoost Model Summary with 400 and lrate 0.1
# ---------------------
# Training error: 0.1317
# Validation error: 0.1555
# Train AUC: 0.9347
# Test AUC: 0.9071
# Classification Report:
#   - Precision (class 1 - canceled): 0.75
#   - Recall (class 1 - canceled): 0.65
#   - F1-score (class 1 - canceled): 0.70
# Overall accuracy: 0.84
#
# Interpretation:
# Model shows good generalization — slightly higher train error and lower train AUC 
# indicate reduced overfitting compared to previous runs. Test AUC and F1 remained stable, 
# meaning the model kept predictive strength while becoming more robust.

# XGBoost Model Summary same as before with l1 and l2
# ---------------------
# Training error: 0.1316
# Validation error: 0.1556
# Train AUC: 0.9343
# Test AUC: 0.9073
# Accuracy: 0.84
# Precision (canceled): 0.75
# Recall (canceled): 0.65
# F1-score (canceled): 0.70
#
# Interpretation:
# Model demonstrates strong generalization with minimal overfitting.
# High AUC indicates excellent ability to distinguish cancellations.
# Balanced performance — slightly conservative but robust.
 