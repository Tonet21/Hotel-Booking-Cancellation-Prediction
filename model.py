import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("cleaned_hotel_data.csv")


X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("\nTop 10 models by accuracy:\n")
print(models.sort_values(by="Accuracy", ascending=False).head(10))

models.to_csv("lazy_results.csv", index=True)
print("\nResults saved to lazy_results.csv")


