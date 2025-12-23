import os
import argparse
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# seeding for consistent results
RANDOM_SEED=42

# path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT_DIR, "data", "features")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def load_data(path):
    """
    Load the feature dataset from CSV.
    """
    return pd.read_csv(path)

def split_data(df):
    """
    Split data into training and testing sets based on chronological order.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full dataset with features and target.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    df = df.sort_values("Date")
    
    # separate features (X) and target (y)
    # drop 'Date' as it is not a numeric feature for the model
    X = df.drop(["target", "Date"], axis=1)
    y = df["target"]

    # 80% train 20% test
    split_idx = int(len(df) * 0.8)
    
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:]
    )

def scale_features(X_train, X_test, ticker):
    """
    Normalize features using StandardScaler.
    
    This is important for linear models (e.g., logistic regression) so that
    features with large ranges (like volume) don't dominate the coefficients.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and testing feature sets.
    ticker : str
        Ticker symbol used for naming the saved scaler file.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    
    # fit on training data only to prevent data leakage from test set
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl"))
    return X_train_s, X_test_s

def train_models(X_train, y_train, ticker):
    """
    Train a set of baseline models with simple hyperparameters.
    
    Models:
    1. Logistic Regression (baseline)
    2. Random Forest
    3. XGBoost
    
    Returns
    -------
    dict
        Dictionary of trained model objects.
    """
    models = {
        "log_reg": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
        "xgb": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{ticker}_{name}.pkl"))
        print(f"Saved {ticker} {name} model.")

    return models

def tune_models(X_train, y_train, ticker):
    """
    Perform hyperparameter tuning using Grid Search.
    
    Uses TimeSeriesSplit to respect temporal order during cross-validation.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Scaled training data.
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict
        Dictionary of the best estimators found.
    """
    # split training data into n sequential folds
    tscv = TimeSeriesSplit(n_splits=5)

    param_grids = {
        "log_reg": {
            "C": [0.01, 0.1, 1.0, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        },
        "rf": {
            "n_estimators": [200, 250, 300],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10]
        },
        "xgb": {
            "n_estimators": [200, 400],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01,0.03, 0.05],
            "subsample": [0.8, 0.9],
            "min_child_weight": [1, 5, 10]
        },
    }

    base_models = {
        "log_reg": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "rf": RandomForestClassifier(random_state=RANDOM_SEED),
        "xgb": XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
    }

    best_models = {}
    best_params = {}

    for name in base_models:
        print(f"Tuning {name}...")

        grid = GridSearchCV(
            estimator=base_models[name],
            param_grid=param_grids[name],
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_
        best_params[name] = grid.best_params_

        joblib.dump(best_models[name], os.path.join(MODELS_DIR, f"{ticker}_{name}_best.pkl"))
        print(f"Saved tuned {name} model.")

    with open(os.path.join(MODELS_DIR, f"{ticker}_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Saved best hyperparameters to {ticker}_best_params.json")
    return best_models

def evaluate_models(models, X_test, y_test, ticker):
    """
    Calculate and save performance metrics for a dictionary of models.
    """
    metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "positive_rate": y_pred.mean()
        }

    df = pd.DataFrame(metrics).T
    df.to_csv(os.path.join(MODELS_DIR, f"{ticker}_metrics.csv"))
    print(f"Saved metrics to models/{ticker}_metrics.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., NVDA)")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    feature_path = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    print(f"Loading features for {ticker}...")
    df = load_data(feature_path)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s = scale_features(X_train, X_test, ticker)

    if args.tune:
        models = tune_models(X_train_s, y_train, ticker)
    else:
        models = train_models(X_train_s, y_train, ticker)

    evaluate_models(models, X_test_s, y_test, ticker)

if __name__ == "__main__":
    main()
