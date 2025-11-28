# Hotel-Booking-Cancellation-Prediction

A Machine Learning project using EDA, LazyPredict, and XGBoost

Link: https://hotel-booking-cancellation-prediction-k6ueusw85czhademi3uscn.streamlit.app/

## Overview

This project aims to predict hotel booking cancellations using structured booking data.
After performing exploratory data analysis (EDA), several preprocessing steps, model experiments, and hyperparameter optimizations were conducted.
The final model is an XGBoost classifier, chosen for its robustness, performance, and ability to handle heterogeneous data.

### 1. Exploratory Data Analysis (EDA)
Understanding Missing Data

During the initial EDA, the dataset was inspected for missing values, distributions, leakage risks, and correlation structures.
Two columns stood out due to extremely high percentages of missing data:

- agent
- country

The agent column contained so many null values that it lacked statistical utility and introduced unnecessary noise.
The country column, while conceptually interesting, also contained a large proportion of missing values and added sparse categorical levels that complicated modeling without improving performance.

After computing the percentage of nulls and running exploratory tests, both variables were dropped.
Additional columns were removed if they contributed little to predictive power or created issues of multicollinearity or leakage (e.g., columns encoding outcomes rather than predictors).

This simplification improved model stability and reduced preprocessing overhead while keeping the dataset interpretable.

### 2. Model Benchmarking with LazyPredict

Before committing to a specific model, LazyPredict was used to establish a performance baseline.
LazyPredict quickly trains multiple off-the-shelf models without requiring manual tuning.
This gave a fast overview of:

Which algorithms performed reasonably well out of the box

How linear models compared to tree-based methods

Whether the problem was better suited for complex nonlinear models

Tree-based models (including Gradient Boosting and Random Forests) consistently outperformed linear ones, especially in terms of AUC and recall for the cancellation class.
This informed the decision to focus on XGBoost for final training and optimization.

### 3. Training the XGBoost Model

XGBoost was selected due to:

Its strong performance on tabular datasets

Built-in regularization (L1 and L2)

Ability to model nonlinear interactions

Robustness to moderate multicollinearity

Excellent performance during initial experiments

Hyperparameter tuning included testing different numbers of estimators, learning rates, and regularization parameters.

### 4. Final Model Performance
XGBoost (400 estimators, learning rate 0.1)

Training error: 0.1317
Validation error: 0.1555
Train AUC: 0.9347
Test AUC: 0.9071
Overall accuracy: 0.84

Class 1 (canceled) metrics:

Precision: 0.75

Recall: 0.65

F1-score: 0.70

Interpretation:
The model shows strong generalization.
The slight increase in error from training to validation suggests reduced overfitting compared to previous attempts.
AUC above 0.90 demonstrates excellent ability to distinguish cancellations from non-cancellations.
Precision and recall are balanced, indicating the model is conservative but reliable.

XGBoost with L1 and L2 Regularization

Training error: 0.1316
Validation error: 0.1556
Train AUC: 0.9343
Test AUC: 0.9073
Accuracy: 0.84

Cancellation class:

Precision: 0.75

Recall: 0.65

F1-score: 0.70

Interpretation:
Regularization slightly stabilizes training behavior without sacrificing performance.
Generalization remains strong, confirming that XGBoost handles this dataset well even with regularization penalties applied.

### 5. Deployment

The model and Streamlit interface can be deployed via Streamlit Cloud or any hosting service capable of running Python applications.
A clean repository structure with clear preprocessing steps, model files, and a requirements.txt ensures smooth deployment.

## Nota Bene

This project is part of an ongoing learning process.
While the code and methodology are functional and yield strong model performance, there is room for refinement in areas such as modularization, feature engineering, cross-validation strategy, and interpretability techniques.
The project reflects a commitment to learning and improving while producing meaningful results.

## Model Explainability with SHAP

To understand why the XGBoost model makes its predictions—and which features drive hotel reservation cancellations—the project uses SHAP (SHapley Additive exPlanations). SHAP provides local and global interpretability by assigning each feature a contribution value for each prediction.

How SHAP Was Applied

The same cleaned and encoded dataset used for training (cleaned_hotel_data.csv) was loaded, and the model (boost.joblib) was explained using SHAP’s TreeExplainer, which is optimized for gradient boosting models like XGBoost.

The workflow was:

- Load the trained model

- Load the cleaned dataset

- Encode categorical features using the same one-hot encoding

- Compute SHAP values

- Generate global and local explanation plots

- Save all visualizations as .png files for reproducibility

### Global Interpretability

### SHAP Feature Importance (Bar Plot)

Ranks features by their average impact on predictions, making it easy to identify the most influential ones.

See summary_plot_bar.png.
