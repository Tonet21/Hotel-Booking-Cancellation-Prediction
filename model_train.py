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
    oob_score=True,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 2️⃣ Evaluate performance
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

oob_scores = []

# Train incrementally to monitor progress
for n in range(10, 201, 10):
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    print(f"{n} trees — OOB Score: {rf.oob_score_:.4f}")



print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 3️⃣ Save the trained model
dump(rf, "rforest.joblib")
print("\n✅ Model saved as rforest.joblib")
