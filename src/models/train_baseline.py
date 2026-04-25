import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import METRICS_DIR, PROCESSED_DIR, ensure_directories


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def main() -> None:
    ensure_directories()
    data_path = PROCESSED_DIR / "f1_features.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Missing feature table. Run src/features/make_features.py first."
        )

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["season", "target_points"])

    feature_cols = [
        "grid",
        "qualifying_position",
        "driver_avg_finish_last5",
        "driver_avg_grid_last5",
        "driver_points_last5",
        "constructor_avg_finish_last5",
        "circuit_name",
        "constructor_name",
    ]
    target_col = "target_points"

    train_df = df[df["season"] <= 2021]
    test_df = df[df["season"] >= 2022]
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Time split is empty. Expand dataset range or adjust split years."
        )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    numeric_features = [
        "grid",
        "qualifying_position",
        "driver_avg_finish_last5",
        "driver_avg_grid_last5",
        "driver_points_last5",
        "constructor_avg_finish_last5",
    ]
    categorical_features = ["circuit_name", "constructor_name"]

    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "logistic_regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    metrics_path = METRICS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved baseline metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
