import json

from src.utils.config import METRICS_DIR, ensure_directories


def _load_metrics(filename: str) -> dict | None:
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_directories()
    metric_files = ["baseline_metrics.json", "xgboost_metrics.json"]
    rows = [m for m in (_load_metrics(name) for name in metric_files) if m]

    if not rows:
        raise FileNotFoundError(
            "No metrics found. Run a training script in src/models first."
        )

    rows_sorted = sorted(rows, key=lambda r: r["roc_auc"], reverse=True)
    summary_path = METRICS_DIR / "model_comparison.json"
    summary_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    print(f"Saved model comparison to: {summary_path}")
    print("Model ranking (best to worst by ROC-AUC):")
    for idx, row in enumerate(rows_sorted, start=1):
        print(
            f"{idx}. {row['model']}: "
            f"ROC-AUC={row['roc_auc']:.4f}, F1={row['f1']:.4f}, "
            f"Accuracy={row['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
