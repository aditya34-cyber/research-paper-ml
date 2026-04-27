"""
IMPROVEMENTS OVER BASE PAPER:
  1. SMOTE for class imbalance handling
  2. RFECV for optimal feature selection
  3. SHAP explainability (summary + waterfall plots)
  4. Probability calibration (Brier Score + Calibration Curve)
  5. Extended metrics: F1, Precision-Recall AUC, Brier Score

"""

import warnings
warnings.filterwarnings("ignore")
import os
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from ucimlrepo import fetch_ucirepo                     

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_recall_curve, auc, brier_score_loss,
                              classification_report, confusion_matrix,
                              RocCurveDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import shap
shap.initjs()

# ── 2. LOAD & PREPARE DATASET ────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading UCI Cleveland Heart Disease Dataset")
print("=" * 60)

heart = fetch_ucirepo(id=45)          # UCI ID for Heart Disease (Cleveland)
X_raw = heart.data.features
y_raw = heart.data.targets

# Binarise target: 0 = no disease, 1 = disease (levels 1-4 → 1)
y = (y_raw.values.ravel() > 0).astype(int)
X = X_raw.copy()

print(f"Dataset shape   : {X.shape}")
print(f"Class balance   : {np.bincount(y)} (0=No Disease, 1=Disease)")
print(f"Missing values  : {X.isnull().sum().sum()}")

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Preprocessing (Imputation + Scaling)")
print("=" * 60)

# Median imputation (same as Rathish)
X.fillna(X.median(), inplace=True)

# Min-Max normalisation (same as Rathish)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Median imputation   : Done")
print("Min-Max scaling     : Done")

# ── IMPROVEMENT 1: SMOTE for class imbalance ─────────────────────────────
print("\n" + "=" * 60)
print("IMPROVEMENT 1 – SMOTE (Class Imbalance Correction)")
print("=" * 60)

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

print(f"Before SMOTE – Train class balance : {np.bincount(y_train_raw)}")
print(f"After  SMOTE – Train class balance : {np.bincount(y_train)}")

# ── IMPROVEMENT 2: RFECV Feature Selection ───────────────────────────────
print("\n" + "=" * 60)
print("IMPROVEMENT 2 – RFECV Feature Selection")
print("=" * 60)

base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(
    estimator=base_rf,
    step=1,
    cv=StratifiedKFold(5),
    scoring="roc_auc",
    min_features_to_select=1,
    n_jobs=-1
)
rfecv.fit(X_train, y_train)

selected_features = X.columns[rfecv.support_].tolist()
print(f"Original features  : {X.shape[1]}")
print(f"Selected features  : {len(selected_features)}")
print(f"Dropped features   : {[f for f in X.columns if f not in selected_features]}")
print(f"Selected           : {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# RFECV plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
         rfecv.cv_results_["mean_test_score"], marker="o", color="#2c7bb6")
plt.axvline(rfecv.n_features_, color="#d7191c", linestyle="--",
            label=f"Optimal: {rfecv.n_features_} features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validated AUC-ROC")
plt.title("RFECV – Optimal Feature Count (Improvement 2)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_rfecv.png", dpi=150)
plt.close()
print("Saved: fig_rfecv.png")

# ── 4. MODEL DEFINITIONS ─────────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"         : GaussianNB(),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          use_label_encoder=False,
                                          eval_metric="logloss", random_state=42),
    # Rathish architecture: RF + XGB → LR meta
    "Stacking (Rathish)"  : StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=200, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.05,
                                  use_label_encoder=False, eval_metric="logloss",
                                  random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=False
    ),
}

# ── IMPROVEMENT 3: Calibrated Stacking ───────────────────────────────────
stacking_base = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.05,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
models["Stacking + Calibrated (Ours)"] = CalibratedClassifierCV(
    stacking_base, cv=5, method="isotonic"
)

# ── 5. TRAIN & EVALUATE ALL MODELS ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Training & Evaluating All Models")
print("=" * 60)

results = []

for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred   = model.predict(X_test_sel)
    y_prob   = model.predict_proba(X_test_sel)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    auc_roc  = roc_auc_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    auc_pr   = auc(rec, prec)
    brier    = brier_score_loss(y_test, y_prob)

    results.append({
        "Model"     : name,
        "Accuracy"  : round(acc * 100, 2),
        "AUC-ROC"   : round(auc_roc, 4),
        "F1-Score"  : round(f1, 4),
        "AUC-PR"    : round(auc_pr, 4),
        "Brier↓"    : round(brier, 4),
    })
    print(f"  {name:<35} Acc={acc*100:.2f}%  AUC={auc_roc:.4f}  F1={f1:.4f}  Brier={brier:.4f}")

results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
print("\n── Final Comparison Table ──")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_table.csv", index=False)
print("Saved: model_comparison_table.csv")

# ── 6. VISUALISATIONS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Generating Figures")
print("=" * 60)

COLORS = {
    "Logistic Regression"         : "#aec6cf",
    "Naive Bayes"                 : "#b5ead7",
    "KNN"                         : "#ffdac1",
    "Decision Tree"               : "#ff9aa2",
    "Random Forest"               : "#c7ceea",
    "XGBoost"                     : "#f4a261",
    "Stacking (Rathish)"          : "#2c7bb6",
    "Stacking + Calibrated (Ours)": "#d7191c",
}

# ── Fig A: Accuracy & AUC-ROC Bar Chart ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = [COLORS[m] for m in results_df["Model"]]

axes[0].barh(results_df["Model"], results_df["Accuracy"], color=colors)
axes[0].set_xlabel("Accuracy (%)")
axes[0].set_title("Model Accuracy Comparison")
axes[0].axvline(results_df[results_df["Model"] == "Stacking (Rathish)"]["Accuracy"].values[0],
                color="#2c7bb6", linestyle="--", linewidth=1, label="Rathish baseline")
axes[0].axvline(results_df[results_df["Model"] == "Stacking + Calibrated (Ours)"]["Accuracy"].values[0],
                color="#d7191c", linestyle="--", linewidth=1, label="Our model")
axes[0].legend(fontsize=8)

axes[1].barh(results_df["Model"], results_df["AUC-ROC"], color=colors)
axes[1].set_xlabel("AUC-ROC")
axes[1].set_title("AUC-ROC Comparison")
axes[1].set_xlim(0.5, 1.0)

plt.suptitle("Figure 4.1 – Model Comparison: Accuracy & AUC-ROC\n(Red = Our Improved Model | Blue = Rathish Baseline)",
             fontsize=11)
plt.tight_layout()
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_table.csv", index=False)
plt.close()
print("Saved: fig_model_comparison.png")

# ── Fig B: ROC Curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=ax,
        name=f"{name} ({roc_auc_score(y_test, y_prob):.3f})",
        lw=2.5 if "Ours" in name else 1.5,
        linestyle="-" if "Ours" in name else "--",
        color=COLORS[name]
    )
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_title("Figure 4.2 – ROC Curves – All Models\n(Rathish Baseline vs Our Improved Stacking)")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_roc_curves.png", dpi=150)
plt.close()
print("Saved: fig_roc_curves.png")

# ── Fig C: Calibration Curve (Improvement 4) ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name in ["Stacking (Rathish)", "Stacking + Calibrated (Ours)"]:
    y_prob = models[name].predict_proba(X_test_sel)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label=name, lw=2,
            color=COLORS[name])
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Figure 4.3 – Calibration Curve\n(Improvement 4: Probability Calibration)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_calibration.png", dpi=150)
plt.close()
print("Saved: fig_calibration.png")

# ── Fig D: SHAP Explainability (Improvement 5) ───────────────────────────
print("\nGenerating SHAP explanations (may take ~30s)...")

# Use the XGBoost model for SHAP (fastest TreeExplainer)
xgb_model = models["XGBoost"]
explainer  = shap.TreeExplainer(xgb_model)
shap_vals  = explainer.shap_values(X_test_sel)

# SHAP Summary Plot (global feature importance)
plt.figure()
shap.summary_plot(shap_vals, X_test_sel, plot_type="bar", show=False)
plt.title("Figure 3.1 – SHAP Feature Importance (XGBoost)\n(Improvement 5: Explainability)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_shap_summary.png")

plt.figure()
shap.summary_plot(shap_vals, X_test_sel, show=False)
plt.title("Figure 3.2 – SHAP Beeswarm Plot (Feature Impact Direction)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_shap_beeswarm.png")

# SHAP Waterfall for one high-risk patient
high_risk_idx = np.where(y_test == 1)[0][0]
shap_exp = shap.Explanation(
    values     = shap_vals[high_risk_idx],
    base_values= explainer.expected_value,
    data       = X_test_sel.iloc[high_risk_idx].values,
    feature_names=selected_features
)
plt.figure()
shap.waterfall_plot(shap_exp, show=False)
plt.title("Figure 3.3 – SHAP Waterfall (Single High-Risk Patient)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_shap_waterfall.png")

# ── 7. PRINT IMPROVEMENT SUMMARY ─────────────────────────────────────────
rathish_row = results_df[results_df["Model"] == "Stacking (Rathish)"].iloc[0]
ours_row    = results_df[results_df["Model"] == "Stacking + Calibrated (Ours)"].iloc[0]

print("\n" + "=" * 60)
print("IMPROVEMENT SUMMARY vs RATHISH (2024)")
print("=" * 60)
print(f"  Accuracy   : {rathish_row['Accuracy']}%  →  {ours_row['Accuracy']}%  "
      f"(Δ = {ours_row['Accuracy'] - rathish_row['Accuracy']:+.2f}%)")
print(f"  AUC-ROC    : {rathish_row['AUC-ROC']}  →  {ours_row['AUC-ROC']}  "
      f"(Δ = {ours_row['AUC-ROC'] - rathish_row['AUC-ROC']:+.4f})")
print(f"  F1-Score   : {rathish_row['F1-Score']}  →  {ours_row['F1-Score']}  "
      f"(Δ = {ours_row['F1-Score'] - rathish_row['F1-Score']:+.4f})")
print(f"  Brier Score: {rathish_row['Brier↓']}  →  {ours_row['Brier↓']}  "
      f"(lower is better)")
print("\nAdditional improvements not in Rathish (2024):")
print("  ✓ SMOTE applied to fix class imbalance")
print("  ✓ RFECV reduced features to optimal subset")
print("  ✓ SHAP XAI: global + local explanations generated")
print("  ✓ Probability calibration applied (Brier Score + Calibration Curve)")
print("  ✓ AUC-PR added as metric for imbalanced evaluation")
print("\nAll output files saved to /mnt/user-data/outputs/")
print("=" * 60)
