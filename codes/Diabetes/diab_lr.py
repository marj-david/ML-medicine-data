#Logistic Regression for Diabetes Prediction

from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

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
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0, 0.5, 1],
    'solver': ['saga']
}

lr_base = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
grid_search = GridSearchCV(
    lr_base, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
# Get the best model

best_model = grid_search.best_estimator_


# Validation results
print("\n" + "=" * 60)
print("VALIDATION SET RESULTS")
print("=" * 60)
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Test results
print("\n" + "=" * 60)
print("TEST SET RESULTS (FINAL EVALUATION)")
print("=" * 60)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

y_probs = best_model.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_probs)

print(f"ROC AUC Score: {roc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()


# Confusion Matrix and Feature Importance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






