#Random forest for ECG heartbeat dataset

import locale
locale.setlocale(locale.LC_ALL, '')

from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv("datasets/mitbih_train.csv", header=None)
test_df = pd.read_csv("datasets/mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Hyperparameter tuning on training set using RandomizedSearchCV with cross-validation
print("=" * 60)
print("TUNING HYPERPARAMETERS ON TRAINING SET (3-FOLD CROSS-VALIDATION, 20 ITERATIONS)")
print("=" * 60)

param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [8, 10, 12, 15, 20],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3]
}

rf_clf = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=rf_clf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

grid_search.fit(X_train, y_train)
best_rf_clf = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Score:", f"{grid_search.best_score_:.4f}")

# Evaluate on validation set
y_val_pred = best_rf_clf.predict(X_val)
y_val_proba = best_rf_clf.predict_proba(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
print("\n" + "=" * 60)
print("VALIDATION SET PERFORMANCE")
print("=" * 60)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation ROC AUC Score: {val_roc_auc:.4f}")
print(f"Validation Weighted F1: {val_f1_weighted:.4f}")
print(f"Validation Macro F1: {val_f1_macro:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = best_rf_clf.predict(X_test)
y_test_proba = best_rf_clf.predict_proba(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
print("\n" + "=" * 60)
print("TEST SET PERFORMANCE")
print("=" * 60)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test ROC AUC Score: {test_roc_auc:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Multi-class ROC curve visualization (one-vs-rest for each class)
n_classes = len(np.unique(y_test))
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Random Forest (One-vs-Rest)")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
