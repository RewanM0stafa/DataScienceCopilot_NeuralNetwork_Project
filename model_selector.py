import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error


def select_best_model(df, target_col):
    """
    Steps:
    1. Split features and target
    2. Auto-detect problem type (classification vs regression)
    3. Feature engineering (Scaling + SelectKBest)
    4. Train and evaluate 3 models using cross-validation
    5. Return the best model with all metadata
    """

    # ============================================================
    # Step 1: Split features and target
    # ============================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.tolist()

    # ============================================================
    # Step 2: Auto-detect problem type
    # Less than 10 unique values -> classification, else -> regression
    # ============================================================
    if y.nunique() <= 10:
        problem_type = 'classification'
    else:
        problem_type = 'regression'

    print(f"Problem type detected: {problem_type}")

    # ============================================================
    # Step 3: Feature Engineering
    # 3a. StandardScaler -> required for Neural Network
    # 3b. SelectKBest  -> select most important features
    # ============================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score_func = f_classif if problem_type == 'classification' else f_regression
    selector = SelectKBest(score_func=score_func, k='all')
    X_selected = selector.fit_transform(X_scaled, y)

    # ============================================================
    # Step 4: Train/Test split
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # ============================================================
    # Step 5: Define models
    # Neural Network is the priority model for this course
    # ============================================================
    if problem_type == 'classification':
        models = {
            'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
        }
        scoring = 'accuracy'
    else:
        models = {
            'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression':    LinearRegression(),
        }
        scoring = 'neg_mean_squared_error'

    # ============================================================
    # Step 6: Cross-validation -> pick best model
    # ============================================================
    best_model = None
    best_score = -np.inf
    best_name  = ""
    all_scores = {}

    for name, model in models.items():
        cv_scores  = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
        mean_score = np.mean(cv_scores)
        all_scores[name] = round(mean_score, 4)
        print(f"  {name}: CV score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name  = name

    # ============================================================
    # Step 7: Final training and evaluation on test set
    # ============================================================
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if problem_type == 'classification':
        test_score = accuracy_score(y_test, y_pred)
    else:
        test_score = mean_squared_error(y_test, y_pred)

    print(f"\nBest model: {best_name} | Test score: {test_score:.4f}")

    return {
        'best_model':    best_model,
        'best_name':     best_name,
        'test_score':    test_score,
        'problem_type':  problem_type,
        'feature_names': feature_names,
        'target_name':   target_col,      # fix: was missing in old code
        'all_scores':    all_scores,
    }


# ============================================================
# Quick test using Breast Cancer dataset
# ============================================================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    from data_cleaner import clean_data

    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")

    df_clean   = clean_data(df)
    model_info = select_best_model(df_clean, target_col='target')

    print("\n--- Results ---")
    print(f"Best model : {model_info['best_name']}")
    print(f"Test score : {model_info['test_score']:.4f}")
    print(f"All scores : {model_info['all_scores']}")