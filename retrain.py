# retrain.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                               AdaBoostClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, classification_report

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv("survey.csv")
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# ── 2. CLEAN ─────────────────────────────────────────────────────────────────
# Drop useless columns
drop_cols = ['Timestamp', 'comments', 'state', 'Country']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Fix Age
df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]

# Standardize Gender
df['Gender'] = df['Gender'].str.lower().str.strip()
male   = ['male','m','man','cis male','male (cis)','make','maile','mal','male ']
female = ['female','f','woman','cis female','female (cis)','femail','femake','cis-female/femme','woman']
df['Gender'] = df['Gender'].apply(
    lambda x: 'Male' if x in male else ('Female' if x in female else 'Other')
)

# Fill missing
df['self_employed']  = df['self_employed'].fillna('No')
df['work_interfere'] = df['work_interfere'].fillna('Never')
df.fillna(df.mode().iloc[0], inplace=True)

# ── 3. ENCODE ────────────────────────────────────────────────────────────────
# Target first
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

# Encode all object columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nClass distribution:")
print(df['treatment'].value_counts())

# ── 4. SPLIT ─────────────────────────────────────────────────────────────────
X = df.drop('treatment', axis=1)
y = df['treatment']

print("\nFeature columns in order:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test set
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# ── 5. TRAIN ALL MODELS ──────────────────────────────────────────────────────
lr    = LogisticRegression(max_iter=1000, random_state=42)
knn   = KNeighborsClassifier(n_neighbors=5)
dt    = DecisionTreeClassifier(random_state=42)
rf    = RandomForestClassifier(n_estimators=100, random_state=42)
bag   = BaggingClassifier(n_estimators=100, random_state=42)
boost = AdaBoostClassifier(n_estimators=100, random_state=42)
stack = StackingClassifier(
    estimators=[('lr', lr), ('knn', knn), ('dt', dt)],
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
)

models = {
    "Logistic_Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "K_Neighbors_Classifier":   KNeighborsClassifier(n_neighbors=5),
    "Decision_Tree_Classifier": DecisionTreeClassifier(random_state=42),
    "Random_Forests":           RandomForestClassifier(n_estimators=100, random_state=42),
    "Bagging":                  BaggingClassifier(n_estimators=100, random_state=42),
    "Boosting":                 AdaBoostClassifier(n_estimators=100, random_state=42),
    "Stacking":                 stack,
}

print("\n── Training Results ──────────────────────────────")
best_model = None
best_acc   = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:30s}  Accuracy: {acc:.4f}")

    # Save every model
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    if acc > best_acc:
        best_acc   = acc
        best_model = name

print(f"\n🏆 Best model: {best_model} ({best_acc:.4f})")

# ── 6. DETAILED REPORT FOR BEST MODEL ────────────────────────────────────────
with open(f"{best_model}.pkl", "rb") as f:
    bm = pickle.load(f)

y_pred = bm.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Treatment","Needs Treatment"]))

# ── 7. SAVE BEST AS Stacking.pkl FOR THE FLASK APP ───────────────────────────
with open(f"{best_model}.pkl", "rb") as f:
    best = pickle.load(f)
with open("Stacking.pkl", "wb") as f:
    pickle.dump(best, f)

print(f"\n✅ Saved best model ({best_model}) as Stacking.pkl for Flask app")