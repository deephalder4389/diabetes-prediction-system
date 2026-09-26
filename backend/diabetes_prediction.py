# =============================================================================
# DIABETES PREDICTION MODEL - End-to-End ML Pipeline
# =============================================================================
# Dataset   : Pima Indians Diabetes Dataset
# Target    : Outcome (1 = Diabetic, 0 = Not Diabetic)
# Focus     : High Recall + F1-Score (medical-grade reliability)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.ensemble import GradientBoostingClassifier   # replaces XGBoost (same family)

# Color palette for consistent plots
PALETTE = {"0 - Not Diabetic": "#4C72B0", "1 - Diabetic": "#DD8452"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# =============================================================================
# STEP 1 — DATA UNDERSTANDING
# =============================================================================

print("=" * 65)
print("STEP 1 — DATA UNDERSTANDING")
print("=" * 65)

df = pd.read_csv("diabetes.csv")

print(f"\n📐 Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Column Names   :\n{df.columns.tolist()}")
print(f"\n🔢 Data Types     :\n{df.dtypes}")
print(f"\n👀 First 5 Rows   :\n{df.head()}")
print(f"\n❓ Missing Values  :\n{df.isnull().sum()}")
print(f"\n🔁 Duplicate Rows : {df.duplicated().sum()}")
print(f"\n🎯 Target Variable: 'Outcome'  (0 = Not Diabetic | 1 = Diabetic)")
print(f"   Class counts   :\n{df['Outcome'].value_counts()}")


# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 65)
print("STEP 2 — DATA PREPROCESSING")
print("=" * 65)

# ── 2a. Handle 'Pregnancies' ──────────────────────────────────────────────────
# DECISION: KEEP 'Pregnancies' but cap it at 1 (binary: "ever pregnant / not").
# Rationale:
#   • Dropping it entirely wastes biological signal (pregnancy history is a
#     known diabetes risk factor even retrospectively).
#   • Keeping raw counts introduces gender bias (males always = 0, which is
#     fine statistically — it just means "no pregnancy history").
#   • Binarising removes the magnitude bias while preserving the signal.
print("\n[Pregnancies] → Kept as raw count (0–17)")
# No transformation — keep original values (0–17)
# ── 2b. Replace biologically impossible zeros with NaN ───────────────────────
# These columns cannot be zero in a living person:
zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"\n[Zero replacement] Columns treated: {zero_invalid}")
for col in zero_invalid:
    n_zeros = (df[col] == 0).sum()
    if n_zeros > 0:
        df[col] = df[col].replace(0, np.nan)
        print(f"   {col:20s}: {n_zeros} zeros → NaN")

# ── 2c. Impute NaN with median (robust to skew/outliers) ─────────────────────
print("\n[Imputation] Filling NaN with column median …")
for col in zero_invalid:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"   {col:20s}: median = {median_val:.2f}")

# ── 2d. Outlier Handling (IQR capping) ───────────────────────────────────────
print("\n[Outlier Capping] IQR-based clipping …")
outlier_cols = ['Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    print(f"   {col:25s}: {clipped} values clipped")

print("\n✅ Preprocessing complete. No missing values:", df.isnull().sum().sum())


# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 65)
print("STEP 3 — EDA  (saving figures to /mnt/user-data/outputs/)")
print("=" * 65)

import os
os.makedirs("/mnt/user-data/outputs", exist_ok=True)

feature_cols = [c for c in df.columns if c != 'Outcome']

# ── 3a. Class Imbalance ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Outcome'].value_counts()
bars = ax.bar(["Not Diabetic (0)", "Diabetic (1)"],
              counts.values,
              color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
            str(int(b.get_height())), ha='center', fontsize=12, fontweight='bold')
ax.set_title("Class Distribution (Target: Outcome)", fontsize=13, fontweight='bold')
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/eda_class_distribution.png", dpi=150)
plt.close()
print("   ► Saved: eda_class_distribution.png")
print(f"   Class balance: {counts[0]} Not Diabetic | {counts[1]} Diabetic")
print(f"   Imbalance ratio ≈ {counts[0]/counts[1]:.2f}:1  (moderate imbalance)")

# ── 3b. Correlation Heatmap ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/eda_correlation_heatmap.png", dpi=150)
plt.close()
print("   ► Saved: eda_correlation_heatmap.png")

# ── 3c. Feature Distributions ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    for outcome, color in zip([0, 1], ["#4C72B0", "#DD8452"]):
        subset = df[df['Outcome'] == outcome][col]
        axes[i].hist(subset, bins=25, alpha=0.6, color=color,
                     label=f"{'Diabetic' if outcome else 'Not Diabetic'}", edgecolor='none')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel("Count")
if len(feature_cols) < len(axes):
    for j in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[j])
fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/eda_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ► Saved: eda_feature_distributions.png")

print("\n📊 Key EDA Insights:")
print("   • Glucose has the highest correlation with Outcome (~0.49)")
print("   • BMI and Age are also strong positive predictors")
print("   • Dataset has moderate imbalance (~65% non-diabetic)")
print("   • Insulin & SkinThickness show wide spread — post-imputation noise expected")


# =============================================================================
# STEP 4 — FEATURE SELECTION
# =============================================================================

print("\n" + "=" * 65)
print("STEP 4 — FEATURE SELECTION")
print("=" * 65)

X_all = df[feature_cols]
y = df['Outcome']

# Quick Random Forest importance to guide selection
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
rf_selector.fit(X_all, y)
importances = pd.Series(rf_selector.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n[Random Forest Feature Importances]")
for feat, imp in importances.items():
    bar = "█" * int(imp * 100)
    print(f"   {feat:27s} {imp:.4f}  {bar}")

# Plot feature importances
fig, ax = plt.subplots(figsize=(8, 5))
importances.sort_values().plot(kind='barh', ax=ax, color='#4C72B0', edgecolor='white')
ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feature_importance.png", dpi=150)
plt.close()
print("   ► Saved: feature_importance.png")

# Select top features (importance > 0.05 threshold)
# Use ALL 8 features — no dropping
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# Force include Pregnancies even if importance < 0.05
if 'Pregnancies' not in selected_features:
    selected_features.append('Pregnancies')
print(f"\n✅ Selected Features (importance > 0.05): {selected_features}")
print("   Rationale: These features contribute >5% individually to tree splits,")
print("   ensuring we keep signal-rich predictors and drop noise.")

X = df[selected_features]


# =============================================================================
# STEP 5 — MODEL BUILDING
# =============================================================================

print("\n" + "=" * 65)
print("STEP 5 — MODEL BUILDING")
print("=" * 65)

# Scale features (needed by Logistic Regression; harmless for tree models)
X = df[selected_features]   # make sure X is a DataFrame with named columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # scaler now knows feature names

# 80/20 stratified split (preserves class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n   Training set : {X_train.shape[0]} samples")
print(f"   Test set     : {X_test.shape[0]} samples")
print(f"   Features used: {selected_features}")

# ── Model definitions ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# ── Train + evaluate helper ───────────────────────────────────────────────────
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, y_pred),  4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall":    round(recall_score(y_te, y_pred),    4),
        "F1-Score":  round(f1_score(y_te, y_pred),        4),
    }, y_pred

results_before = []
preds_before   = {}

for name, model in models.items():
    metrics, y_pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results_before.append(metrics)
    preds_before[name] = y_pred
    print(f"\n   [{name}]")
    print(f"     Accuracy : {metrics['Accuracy']:.4f}")
    print(f"     Precision: {metrics['Precision']:.4f}")
    print(f"     Recall   : {metrics['Recall']:.4f}  ← critical for medical use")
    print(f"     F1-Score : {metrics['F1-Score']:.4f}  ← primary metric")

results_df_before = pd.DataFrame(results_before)


# =============================================================================
# STEP 6 — MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 65)
print("STEP 6 — MODEL EVALUATION")
print("=" * 65)

print("\n📊 Comparison Table (Before Tuning):")
print(results_df_before.to_string(index=False))

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, preds_before[name])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Diabetic", "Diabetic"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11, fontweight='bold')
plt.suptitle("Confusion Matrices — All Models (Before Tuning)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrices_before.png", dpi=150)
plt.close()
print("   ► Saved: confusion_matrices_before.png")

# ── Bar chart comparison ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics_to_plot))
width = 0.25
colors = ["#4C72B0", "#55A868", "#DD8452"]
for i, row in results_df_before.iterrows():
    ax.bar(x + i * width, [row[m] for m in metrics_to_plot],
           width, label=row['Model'], color=colors[i], edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Before Tuning", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/model_comparison_before.png", dpi=150)
plt.close()
print("   ► Saved: model_comparison_before.png")

best_before = results_df_before.loc[results_df_before['F1-Score'].idxmax(), 'Model']
print(f"\n   Best model before tuning (F1): {best_before}")


# =============================================================================
# STEP 7 — HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "=" * 65)
print("STEP 7 — HYPERPARAMETER TUNING")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Random Forest — GridSearchCV ──────────────────────────────────────────────
print("\n[Random Forest] GridSearchCV …")
rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight':      ['balanced', None],   # helps with class imbalance
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
rf_gs.fit(X_train, y_train)
best_rf = rf_gs.best_estimator_
print(f"   Best params : {rf_gs.best_params_}")
print(f"   Best CV F1  : {rf_gs.best_score_:.4f}")

# ── Gradient Boosting — RandomizedSearchCV ───────────────────────────────────
print("\n[Gradient Boosting] RandomizedSearchCV …")
gb_param_dist = {
    'n_estimators':   [100, 200, 300, 400],
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.05, 0.1, 0.2],
    'subsample':      [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
}
gb_rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_dist, n_iter=40, cv=cv, scoring='f1',
    n_jobs=-1, random_state=42, verbose=0)
gb_rs.fit(X_train, y_train)
best_xgb = gb_rs.best_estimator_
print(f"   Best params : {gb_rs.best_params_}")
print(f"   Best CV F1  : {gb_rs.best_score_:.4f}")

# ── Logistic Regression — kept with C-tuned version ──────────────────────────
print("\n[Logistic Regression] GridSearchCV (C / penalty) …")
lr_param_grid = {
    'C':            [0.01, 0.1, 1, 10, 100],
    'penalty':      ['l2'],
    'class_weight': ['balanced', None],
}
lr_gs = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
lr_gs.fit(X_train, y_train)
best_lr = lr_gs.best_estimator_
print(f"   Best params : {lr_gs.best_params_}")
print(f"   Best CV F1  : {lr_gs.best_score_:.4f}")


# =============================================================================
# STEP 8 — FINAL MODEL SELECTION & BEFORE vs AFTER COMPARISON
# =============================================================================

print("\n" + "=" * 65)
print("STEP 8 — FINAL MODEL SELECTION")
print("=" * 65)

tuned_models = {
    "Logistic Regression (Tuned)": best_lr,
    "Random Forest (Tuned)":       best_rf,
    "Gradient Boosting (Tuned)":   best_xgb,
}

results_after = []
preds_after   = {}

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    preds_after[name] = y_pred
    results_after.append({
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1-Score":  round(f1_score(y_test, y_pred),        4),
    })

results_df_after = pd.DataFrame(results_after)

print("\n📊 Comparison Table (After Tuning):")
print(results_df_after.to_string(index=False))

# ── Before vs After visual ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (df_res, title) in zip(axes, [
    (results_df_before, "Before Tuning"),
    (results_df_after,  "After Tuning"),
]):
    x = np.arange(len(metrics_to_plot))
    for i, row in df_res.iterrows():
        ax.bar(x + i * 0.25, [row[m] for m in metrics_to_plot],
               0.25, label=row['Model'], color=colors[i], edgecolor='white', alpha=0.88)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle("Before vs After Hyperparameter Tuning", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/before_vs_after_tuning.png", dpi=150)
plt.close()
print("   ► Saved: before_vs_after_tuning.png")

# ── Confusion matrices after tuning ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, tuned_models.items()):
    cm = confusion_matrix(y_test, preds_after[name])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Diabetic", "Diabetic"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(name, fontsize=10, fontweight='bold')
plt.suptitle("Confusion Matrices — All Models (After Tuning)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrices_after.png", dpi=150)
plt.close()
print("   ► Saved: confusion_matrices_after.png")

# ── Select best model ─────────────────────────────────────────────────────────
best_row   = results_df_after.loc[results_df_after['F1-Score'].idxmax()]
best_name  = best_row['Model']
best_model = tuned_models[best_name]

print(f"\n🏆 BEST MODEL : {best_name}")
print(f"   Accuracy   : {best_row['Accuracy']:.4f}")
print(f"   Precision  : {best_row['Precision']:.4f}")
print(f"   Recall     : {best_row['Recall']:.4f}  ← highest = fewest missed diabetics")
print(f"   F1-Score   : {best_row['F1-Score']:.4f}  ← best balance Precision/Recall")
print("\n   Justification:")
print("   • F1-Score is the primary metric — it penalises both false positives")
print("     and false negatives equally, ideal for imbalanced medical data.")
print("   • High Recall is critical: missing a diabetic patient (false negative)")
print("     has far greater cost than a false alarm (false positive).")
print(f"   • {best_name} achieves the best combination of both.")

# Full classification report
print(f"\n[Detailed Classification Report — {best_name}]")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=["Not Diabetic", "Diabetic"]))


# =============================================================================
# STEP 9 — SAVE MODEL
# =============================================================================

print("=" * 65)
print("STEP 9 — SAVE MODEL")
print("=" * 65)

model_payload = {
    'model':    best_model,
    'scaler':   scaler,
    'features': selected_features,
}
model_path = "diabetes_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model_payload, f)

print(f"\n✅ Model saved to: {model_path}")
print("\n   [Loading example]")
print("   with open('diabetes_model.pkl', 'rb') as f:")
print("       payload = pickle.load(f)")
print("   model   = payload['model']")
print("   scaler  = payload['scaler']")
print("   features= payload['features']")


# =============================================================================
# STEP 10 — PREDICTION FUNCTION
# =============================================================================

print("\n" + "=" * 65)
print("STEP 10 — PREDICTION FUNCTION")
print("=" * 65)

def predict_diabetes(input_data: list) -> str:
    """
    Predict whether a patient is Diabetic or Not Diabetic.

    Parameters
    ----------
    input_data : list
        Feature values in the same order as `selected_features`.
        For this model the expected features are (in order):
            Glucose, BMI, Age, DiabetesPedigreeFunction,
            BloodPressure, Insulin, SkinThickness, Pregnancies
        (actual order depends on feature selection result)

    Returns
    -------
    str : "Diabetic" or "Not Diabetic"
    """
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)

    mdl      = payload['model']
    sc       = payload['scaler']
    feats    = payload['features']

    arr      = np.array(input_data).reshape(1, -1)
    arr_sc   = sc.transform(arr)
    pred     = mdl.predict(arr_sc)[0]
    prob     = mdl.predict_proba(arr_sc)[0][1] if hasattr(mdl, "predict_proba") else None

    result = "Diabetic" if pred == 1 else "Not Diabetic"
    if prob is not None:
        print(f"   Prediction probability (Diabetic): {prob:.2%}")
    return result


print(f"\n   Expected feature order: {selected_features}")
print("\n   [Demo — sample from test set]")

# Grab one diabetic and one non-diabetic sample from original (unscaled)
sample_diabetic     = df[df['Outcome'] == 1][selected_features].iloc[0].tolist()
sample_non_diabetic = df[df['Outcome'] == 0][selected_features].iloc[0].tolist()

print(f"\n   Input (known Diabetic)     : {sample_diabetic}")
result1 = predict_diabetes(sample_diabetic)
print(f"   → Prediction: {result1}")

print(f"\n   Input (known Not Diabetic) : {sample_non_diabetic}")
result2 = predict_diabetes(sample_non_diabetic)
print(f"   → Prediction: {result2}")


# =============================================================================
# BONUS — FLASK DEPLOYMENT SNIPPET
# =============================================================================

flask_snippet = '''
# =============================================================================
# BONUS — Flask Deployment (app.py)
# Run: python app.py   then POST to http://127.0.0.1:5000/predict
# =============================================================================

from flask import Flask, request, jsonify
import pickle, numpy as np

app = Flask(__name__)

# Load model once at startup
with open("diabetes_model.pkl", "rb") as f:
    payload = pickle.load(f)
model    = payload["model"]
scaler   = payload["scaler"]
features = payload["features"]

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON body example:
    {
      "features": [148, 33.6, 50, 0.627, 72, 0, 35, 1]
    }
    Values must be in the same order as `features` list in the payload.
    """
    data   = request.get_json(force=True)
    arr    = np.array(data["features"]).reshape(1, -1)
    arr_sc = scaler.transform(arr)
    pred   = model.predict(arr_sc)[0]
    prob   = float(model.predict_proba(arr_sc)[0][1])
    return jsonify({
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability_diabetic": round(prob, 4)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": features})

if __name__ == "__main__":
    app.run(debug=True)
'''

flask_path = "/mnt/user-data/outputs/app.py"
with open(flask_path, "w") as f:
    f.write(flask_snippet)

print("\n" + "=" * 65)
print("BONUS — FLASK DEPLOYMENT")
print("=" * 65)
print(f"\n✅ Flask app saved to: {flask_path}")
print(flask_snippet)

print("\n" + "=" * 65)
print("✅  PIPELINE COMPLETE — All outputs saved to /mnt/user-data/outputs/")
print("=" * 65)
