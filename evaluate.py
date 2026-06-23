import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, matthews_corrcoef)

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

model_names = [
    "Logistic_Regression", "K_Neighbors_Classifier", "Decision_Tree_Classifier",
    "Random_Forests", "Bagging", "Boosting", "Stacking"
]

results = []

for name in model_names:
    with open(f"{name}.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    TP, FN, FP, TN = cm[1,1], cm[1,0], cm[0,1], cm[0,0]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    tpr  = TP / (TP + FN)          # Sensitivity / Recall
    fnr  = FN / (TP + FN)
    fpr  = FP / (FP + TN)
    tnr  = TN / (TN + FP)          # Specificity
    fdr  = FP / (TP + FP) if (TP + FP) > 0 else 0
    mcc  = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Classifier": name.replace("_", " "),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "TPR": round(tpr, 4),
        "FNR": round(fnr, 4),
        "FPR": round(fpr, 4),
        "TNR": round(tnr, 4),
        "FDR": round(fdr, 4),
        "MCC": round(mcc, 4),
    })

df_results = pd.DataFrame(results)
print("\n=== TABLE I ===")
print(df_results[["Classifier","Accuracy","Precision","Recall","F1 Score","TPR"]].to_string(index=False))

print("\n=== TABLE II ===")
print(df_results[["Classifier","FNR","FPR","TNR","FDR","MCC"]].to_string(index=False))

# ── Plot Accuracy Comparison Bar Chart ──────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.barplot(x="Classifier", y="Accuracy", data=df_results, palette="Blues_d")
plt.title("Accuracy Comparison of ML Classifiers")
plt.xticks(rotation=30, ha='right')
plt.ylim(0.75, 0.85)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()

# ── Confusion Matrix for best model (Boosting / Stacking) ───────────────────
for name in ["Boosting", "Stacking"]:
    with open(f"{name}.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Treatment','Treatment'],
                yticklabels=['No Treatment','Treatment'])
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"cm_{name}.png", dpi=150)
    plt.show()

print("\n✅ Evaluation complete. Charts saved.")