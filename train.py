import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bmw.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "price"
NUMERIC_FEATURES = ["year", "mileage", "tax", "mpg", "engineSize"]
CATEGORICAL_FEATURES = ["model", "transmission", "fuelType"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()
    df.columns = df.columns.str.strip()
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_baseline(model, X_train, y_train, X_test, y_test, cv) -> dict:
    cv_mae_scores = []
    for train_idx, valid_idx in cv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        cv_mae_scores.append(mean_absolute_error(y_val, preds))

    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    metrics = calculate_metrics(y_test, test_preds)
    return {
        "name": model.named_steps["model"].__class__.__name__,
        "model": model,
        **metrics,
        "cv_mae_mean": float(np.mean(cv_mae_scores)),
        "cv_mae_std": float(np.std(cv_mae_scores)),
        "best_params": {},
    }


def evaluate_search(search, X_train, y_train, X_test, y_test) -> dict:
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    metrics = calculate_metrics(y_test, preds)
    return {
        "name": best_model.named_steps["model"].__class__.__name__,
        "model": best_model,
        **metrics,
        "cv_mae_mean": float(-search.best_score_),
        "cv_mae_std": None,
        "best_params": search.best_params_,
    }


def main():
    df = load_data()
    df = cap_outliers_iqr(df, ["price", "mileage", "tax", "mpg", "engineSize"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # 1) Baseline
    linear_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    linear_result = evaluate_baseline(linear_pipeline, X_train, y_train, X_test, y_test, cv)

    # 2) Random Forest + tuning
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions={
            "model__n_estimators": [120, 180],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        n_iter=4,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    rf_result = evaluate_search(rf_search, X_train, y_train, X_test, y_test)

    # 3) Extra Trees + tuning
    et_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    et_search = RandomizedSearchCV(
        estimator=et_pipeline,
        param_distributions={
            "model__n_estimators": [120, 180],
            "model__max_depth": [None, 15, 25],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        n_iter=4,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    et_result = evaluate_search(et_search, X_train, y_train, X_test, y_test)

    all_results = [linear_result, rf_result, et_result]
    best_result = min(all_results, key=lambda x: x["mae"])
    best_model = best_result["model"]

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    serializable_results = []
    for item in all_results:
        serializable_results.append(
            {
                "name": item["name"],
                "mae": item["mae"],
                "rmse": item["rmse"],
                "r2": item["r2"],
                "cv_mae_mean": item["cv_mae_mean"],
                "cv_mae_std": item["cv_mae_std"],
                "best_params": item["best_params"],
            }
        )

    metadata = {
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_options": {
            "model": sorted(df["model"].dropna().unique().tolist()),
            "transmission": sorted(df["transmission"].dropna().unique().tolist()),
            "fuelType": sorted(df["fuelType"].dropna().unique().tolist()),
        },
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }
            for col in NUMERIC_FEATURES
        },
        "best_model_name": best_result["name"],
        "best_model_test_mae": best_result["mae"],
        "best_model_test_rmse": best_result["rmse"],
        "best_model_test_r2": best_result["r2"],
        "results": serializable_results,
    }

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=== MODEL RESULTS ===")
    for result in serializable_results:
        print(result)
    print("\nSaved: models/best_model.pkl")
    print("Saved: models/metadata.json")


if __name__ == "__main__":
    main()
