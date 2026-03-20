import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

df = pd.read_csv("online_shoppers_intention.csv")
print("Dataset shape:", df.shape)
print(df.head())

# DATA CLEANING
df = df.drop_duplicates()
print("\nMissing values:\n", df.isnull().sum())
df["Revenue"] = df["Revenue"].astype(int)
le = LabelEncoder()
df["Month"] = le.fit_transform(df["Month"])
df["VisitorType"] = le.fit_transform(df["VisitorType"])
df["Weekend"] = df["Weekend"].astype(int)


# ===============================
# FEATURE ENGINEERING
# ===============================

df["TotalDuration"] = (
    df["Administrative_Duration"]
    + df["Informational_Duration"]
    + df["ProductRelated_Duration"]
)

df["TotalPages"] = (
    df["Administrative"]
    + df["Informational"]
    + df["ProductRelated"]
)

df["BounceExitDiff"] = df["ExitRates"] - df["BounceRates"]


# ===============================
# FEATURE / TARGET SPLIT
# ===============================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("\nOriginal Class Distribution:")
print(y.value_counts())


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# FEATURE SCALING
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# SMOTE BALANCING
# ===============================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nBalanced Class Distribution:")
print(pd.Series(y_train).value_counts())

# MODEL EVALUATION FUNCTION
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Print classification report
    print(f"\n{model_name} Classification Report\n")
    print(classification_report(y_test, y_pred))

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy:", acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return acc, fpr, tpr, auc


# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(random_state=42),

    "SVM": SVC(kernel="rbf", probability=True),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Gradient Boosting": GradientBoostingClassifier(),

    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = {}

for name, model in models.items():
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)


# =========================
# RANDOM FOREST HYPERPARAMETER TUNING
# =========================

print("\nRunning Random Forest Hyperparameter Tuning...\n")

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("Best Random Forest Parameters:", grid.best_params_)

results["Tuned Random Forest"] = evaluate_model(
    best_rf, X_train, X_test, y_train, y_test, "Tuned Random Forest"
)

# ===============================
# MODEL COMPARISON TABLE
# ===============================

model_comparison = pd.DataFrame({

    "Model": results.keys(),
    "Accuracy": [results[m][0] for m in results],
    "ROC_AUC": [results[m][3] for m in results]

})

print("\nModel Comparison\n")
print(model_comparison)


# ===============================
# ACCURACY BAR GRAPH
# ===============================

plt.figure(figsize=(10,6))

sns.barplot(x="Model", y="Accuracy", data=model_comparison)

plt.xticks(rotation=45)

plt.title("Model Accuracy Comparison")

plt.show()


# ===============================
# ROC CURVE
# ===============================

plt.figure(figsize=(8,6))

for name, (_, fpr, tpr, auc) in results.items():

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()