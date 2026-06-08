# Naive Bayes for Diabetes Prediction

from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('datasets/diabetes.csv')

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split into train+val (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split train+val into train (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Handle zero values (treating as missing) - calculate medians ONLY from training set
columns_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

for col in columns_with_zero:
    median_val = X_train[col].replace(0, np.nan).median()
    X_train[col] = X_train[col].replace(0, median_val)
    X_val[col] = X_val[col].replace(0, median_val)
    X_test[col] = X_test[col].replace(0, median_val)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Hyperparameter tuning on training set using GridSearchCV with cross-validation
print("=" * 60)
print("TUNING HYPERPARAMETERS ON TRAINING SET (5-Fold Cross-Validation)")
print("=" * 60)

param_grid = {
    'var_smoothing': np.logspace(1e-9, 1e-6, 10)
}

nb_clf = GaussianNB()
grid_search = GridSearchCV(
    estimator=nb_clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
best_nb_clf = grid_search.best_estimator_

# Evaluate on validation set
print("\n" + "=" * 60)
print("VALIDATION SET PERFORMANCE")
print("=" * 60)

y_val_pred = best_nb_clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
y_val_proba = best_nb_clf.predict_proba(X_val)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_val, y_val_proba))

# Evaluate on test set
print("\n" + "=" * 60)
print("TEST SET PERFORMANCE")
print("=" * 60)

y_test_pred = best_nb_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
y_test_proba = best_nb_clf.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


