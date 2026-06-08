#Logistic Regression for ECG Heartbeat Classification - ROC Visualization

import locale
locale.setlocale(locale.LC_ALL, '')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

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

# Train with best hyperparameters (no GridSearchCV)
print("=" * 60)
print("TRAINING WITH BEST HYPERPARAMETERS (FAST)")
print("=" * 60)

# Use the best hyperparameters from GridSearchCV
# If different, modify these values based on your GridSearchCV results
best_params = {
    'C': 10,           # Modify based on your results
    'l1_ratio': 1,    # 0 for L2, 1 for L1
    'solver': 'saga'
}

lr_clf = LogisticRegression(
    random_state=42, 
    max_iter=3000,
    C=best_params['C'],
    l1_ratio=best_params['l1_ratio'],
    solver=best_params['solver']
)

print("Training model...")
lr_clf.fit(X_train, y_train)
print("Training complete!")

# Evaluate on test set
y_test_pred = lr_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# ROC AUC for multiclass
y_probs = lr_clf.predict_proba(X_test)
roc = roc_auc_score(y_test, y_probs, multi_class='ovr')
print(f"Test ROC AUC (One-vs-Rest): {roc:.4f}")

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

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression (ECG Heartbeat)")
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

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
