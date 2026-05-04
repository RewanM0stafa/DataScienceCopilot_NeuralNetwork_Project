import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# Helper: smarter problem type detection
# ============================================================
def detect_problem_type(y: pd.Series) -> str:
    """
    Rules (ordered by priority):
    1. dtype is object / bool  -> classification
    2. unique values <= 15 AND unique ratio < 5% -> classification
    3. dtype is float with many unique values    -> regression
    4. fallback                                  -> regression
    """
    if y.dtype == object or y.dtype == bool:
        return 'classification'

    n_unique = y.nunique()
    unique_ratio = n_unique / len(y)

    if n_unique <= 15 and unique_ratio < 0.05:
        return 'classification'

    if y.dtype in ['float64', 'float32'] and n_unique > 20:
        return 'regression'

    # integers with many unique values -> regression
    if n_unique > 20:
        return 'regression'

    return 'classification'


# ============================================================
# Visualization helpers
# ============================================================
def plot_cv_scores(all_scores: dict, problem_type: str, save_path='cv_scores.png'):
    """Bar chart comparing all model CV scores."""
    names  = list(all_scores.keys())
    scores = list(all_scores.values())

    colors = ['#4C72B0' if s != max(scores) else '#DD8452' for s in scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, scores, color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=4)

    metric = 'Accuracy (CV)' if problem_type == 'classification' else 'Neg-MSE (CV)'
    ax.set_xlabel(metric)
    ax.set_title('Model Comparison — Cross-Validation Scores')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [chart saved] {save_path}")


def plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
    """Saves a styled confusion matrix."""
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [chart saved] {save_path}")


def plot_feature_importance(model, feature_names: list, save_path='feature_importance.png'):
    """Saves a feature-importance chart (Random Forest / XGBoost only)."""
    if not hasattr(model, 'feature_importances_'):
        print("  [skip] feature importance not available for this model.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]   # top 20

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color='#4C72B0'
    )
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [chart saved] {save_path}")


# ============================================================
# Main function
# ============================================================
def select_best_model(df: pd.DataFrame, target_col: str, save_charts: bool = True) -> dict:
    """
    Steps:
    1.  Split features / target
    2.  Auto-detect problem type (smarter heuristic)
    3.  Train / test split  ← split BEFORE any preprocessing (fixes leakage)
    4.  Build sklearn Pipelines (scaler + selector + model)
    5.  Cross-validation on training folds only  ← no leakage
    6.  Pick best pipeline, evaluate on held-out test set
    7.  Save visualisation charts
    8.  Return metadata dict
    """

    # ── 1. Split features / target ──────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    # ── 2. Detect problem type ──────────────────────────────────
    problem_type = detect_problem_type(y)
    print(f"Problem type detected: {problem_type}")

    # ── 3. Train / test split (raw data, no preprocessing yet) ──
    #   ⚠️  Fix: we split FIRST, then preprocessing lives inside
    #           the Pipeline so it never sees test data during CV.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 4. Build Pipelines ──────────────────────────────────────
    score_func = f_classif if problem_type == 'classification' else f_regression

    def make_pipeline(estimator):
        return Pipeline([
            ('scaler',   StandardScaler()),
            ('selector', SelectKBest(score_func=score_func, k='all')),
            ('model',    estimator),
        ])

    if problem_type == 'classification':
        candidates = {
            'Neural Network (MLP)': make_pipeline(
                MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            ),
            'Random Forest': make_pipeline(
                RandomForestClassifier(n_estimators=100, random_state=42)
            ),
            'Logistic Regression': make_pipeline(
                LogisticRegression(max_iter=1000, random_state=42)
            ),
            'SVM': make_pipeline(
                SVC(kernel='rbf', random_state=42)
            ),
        }
        if XGBOOST_AVAILABLE:
            candidates['XGBoost'] = make_pipeline(
                XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric='logloss', verbosity=0)
            )
        scoring = 'accuracy'

    else:
        candidates = {
            'Neural Network (MLP)': make_pipeline(
                MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            ),
            'Random Forest': make_pipeline(
                RandomForestRegressor(n_estimators=100, random_state=42)
            ),
            'Linear Regression': make_pipeline(
                LinearRegression()
            ),
            'SVM': make_pipeline(
                SVR(kernel='rbf')
            ),
        }
        if XGBOOST_AVAILABLE:
            candidates['XGBoost'] = make_pipeline(
                XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            )
        scoring = 'neg_mean_squared_error'

    # ── 5. Cross-validation (training data only) ────────────────
    best_pipeline = None
    best_score    = -np.inf
    best_name     = ""
    all_scores    = {}

    print("\nCross-validation results:")
    for name, pipeline in candidates.items():
        cv_scores  = cross_val_score(pipeline, X_train_raw, y_train, cv=5, scoring=scoring)
        mean_score = np.mean(cv_scores)
        all_scores[name] = round(mean_score, 4)
        print(f"  {name:30s}: CV score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score    = mean_score
            best_pipeline = pipeline
            best_name     = name

    # ── 6. Final fit + test evaluation ──────────────────────────
    best_pipeline.fit(X_train_raw, y_train)
    y_pred = best_pipeline.predict(X_test_raw)

    if problem_type == 'classification':
        test_score = accuracy_score(y_test, y_pred)
        print(f"\nBest model : {best_name}")
        print(f"Test Accuracy : {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        test_score = mean_squared_error(y_test, y_pred)
        print(f"\nBest model : {best_name}")
        print(f"Test MSE   : {test_score:.4f}")

    # ── 7. Visualisations ────────────────────────────────────────
    if save_charts:
        plot_cv_scores(all_scores, problem_type)

        if problem_type == 'classification':
            plot_confusion_matrix(y_test, y_pred)

        # Feature importance: extract the raw model from the pipeline
        raw_model = best_pipeline.named_steps['model']
        plot_feature_importance(raw_model, feature_names)

    # ── 8. Return metadata ───────────────────────────────────────
    return {
        'best_model':    best_pipeline,        # full Pipeline object
        'best_name':     best_name,
        'test_score':    test_score,
        'problem_type':  problem_type,
        'feature_names': feature_names,
        'target_name':   target_col,
        'all_scores':    all_scores,
    }


# ============================================================
# Quick test — Breast Cancer dataset
# ============================================================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from data_cleaner import clean_data

    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols\n")

    df_clean   = clean_data(df)
    model_info = select_best_model(df_clean, target_col='target')

    print("\n--- Final Results ---")
    print(f"Best model : {model_info['best_name']}")
    print(f"Test score : {model_info['test_score']:.4f}")
    print(f"All scores : {model_info['all_scores']}")