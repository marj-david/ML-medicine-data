#Logistic Regression for ECG Heartbeat Classification

import locale
locale.setlocale(locale.LC_ALL, '')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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

# Hyperparameter tuning on training set using GridSearchCV with cross-validation
print("=" * 60)
print("TUNING HYPERPARAMETERS ON TRAINING SET (3-FOLD CROSS-VALIDATION)")
print("=" * 60)

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'l1_ratio': [0, 1],  # 0 for L2, 1 for L1
    'solver': ['saga']
}

lr_clf = LogisticRegression(random_state=42, max_iter=3000)
grid_search = GridSearchCV(
    estimator=lr_clf,
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

best_lr_clf = grid_search.best_estimator_

# Evaluate on validation set

y_val_pred = best_lr_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Evaluate on test set
y_test_pred = best_lr_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

#Test results
print("\n" + "=" * 60)
print("TEST SET RESULTS (FINAL EVALUATION)")
print("=" * 60)
y_pred = best_lr_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

y_probs = best_lr_clf.predict_proba(X_test)
roc = roc_auc_score(y_test, y_probs, multi_class='ovr')
print(f"Test ROC AUC (OvR): {roc:.4f}")

# Multiclass ROC Curves
n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_probs[:, i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves - Logistic Regression (ECG Heartbeat)')
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()