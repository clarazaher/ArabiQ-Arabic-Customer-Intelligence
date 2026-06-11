"""
Tune the XGBoost churn prediction model using Optuna.

This script:
1. Loads the engineered churn dataset.
2. Reuses the preprocessing pipeline.
3. Uses Optuna to search for better XGBoost hyperparameters.
4. Trains a final tuned XGBoost model.
5. Saves the tuned model locally.
6. Saves tuning results and best parameters to reports/.
"""

from pathlib import Path
import json
import warnings

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocess import build_preprocessing_pipeline


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data/processed/churn_engineered.csv"
MODELS_DIR = PROJECT_ROOT / "models/ml"
REPORTS_DIR = PROJECT_ROOT / "reports"

TUNED_MODEL_PATH = MODELS_DIR / "xgb_tuned.pkl"
BEST_PARAMS_PATH = REPORTS_DIR / "xgb_tuned_best_params.json"
TUNING_RESULTS_PATH = REPORTS_DIR / "xgb_tuned_results.csv"


def main() -> None:
    """Run Optuna tuning for XGBoost."""

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. "
            "Run scripts/ml/feature_engineering.py first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading engineered dataset from:")
    print(ENGINEERED_DATA_PATH)

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    print("Dataset shape:", df.shape)

    X_train, X_test, y_train, y_test, _ = build_preprocessing_pipeline(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    default_scale_pos_weight = negative_count / positive_count

    print("\nDefault scale_pos_weight:")
    print(round(default_scale_pos_weight, 3))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna tries different parameter combinations.
        The value returned here is what Optuna tries to maximize.
        """

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                max(1.0, default_scale_pos_weight - 1.0),
                default_scale_pos_weight + 2.0,
            ),
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)

        auc_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        return auc_scores.mean()

    print("\nStarting Optuna tuning...")
    print("This may take a few minutes on Mac.")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest cross-validation AUC:")
    print(round(study.best_value, 4))

    print("\nBest parameters:")
    print(study.best_params)

    final_params = {
        **study.best_params,
        "eval_metric": "auc",
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }

    tuned_model = xgb.XGBClassifier(**final_params)
    tuned_model.fit(X_train, y_train)

    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]

    results = {
        "Model": "XGBoost Tuned",
        "Best CV AUC": round(study.best_value, 4),
        "Test AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Test F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
    }

    print("\nTuned XGBoost test results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    joblib.dump(tuned_model, TUNED_MODEL_PATH)

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=4)

    pd.DataFrame([results]).to_csv(TUNING_RESULTS_PATH, index=False)

    print("\nSaved tuned model locally to:")
    print(TUNED_MODEL_PATH)

    print("\nSaved best parameters to:")
    print(BEST_PARAMS_PATH)

    print("\nSaved tuning results to:")
    print(TUNING_RESULTS_PATH)


if __name__ == "__main__":
    main()
