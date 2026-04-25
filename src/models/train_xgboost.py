import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import METRICS_DIR, PROCESSED_DIR, ensure_directories


def _make_classifier():
    """
    Prefer XGBoost when available; fall back to RandomForest when
    OpenMP/runtime dependencies are missing on local machines.
    """
    try:
        from xgboost import XGBClassifier

        return (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "XGBoost unavailable on this machine. "
            "Falling back to RandomForest.\n"
            f"Reason: {exc}"
        )
        return (
            "random_forest_fallback",
            RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
            ),
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), [
                "grid",
                "qualifying_position",
                "driver_avg_finish_last5",
                "driver_avg_grid_last5",
                "driver_points_last5",
                "constructor_avg_finish_last5",
            ]),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["circuit_name", "constructor_name"],
            ),
        ]
    )

    model_name, classifier = _make_classifier()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    metrics_path = METRICS_DIR / "xgboost_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
