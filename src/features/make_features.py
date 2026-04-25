import pandas as pd

from src.utils.config import PROCESSED_DIR, ensure_directories


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["driver_id", "race_date"])

    grouped = df.groupby("driver_id", group_keys=False)
    df["driver_avg_finish_last5"] = grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["driver_avg_grid_last5"] = grouped["grid"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["driver_points_last5"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )

    team_grouped = df.groupby("constructor_id", group_keys=False)
    df["constructor_avg_finish_last5"] = team_grouped["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    return df


def main() -> None:
    ensure_directories()
    input_path = PROCESSED_DIR / "f1_model_table.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing processed table. Run src/data/build_dataset.py first."
        )

    df = pd.read_csv(input_path)
    features_df = add_rolling_features(df)

    output_path = PROCESSED_DIR / "f1_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Saved feature table to: {output_path}")
    print(f"Rows: {len(features_df):,} | Columns: {len(features_df.columns)}")


if __name__ == "__main__":
    main()
