import json
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DIR, RAW_DIR, ensure_directories


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_results_table(results_payload: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for race in results_payload:
        season = int(race["season"])
        round_number = int(race["round"])
        race_name = race["raceName"]
        circuit_name = race["Circuit"]["circuitName"]
        race_date = race["date"]

        for result in race["Results"]:
            driver = result["Driver"]
            constructor = result["Constructor"]
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "race_name": race_name,
                    "race_date": race_date,
                    "circuit_name": circuit_name,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code"),
                    "driver_given_name": driver["givenName"],
                    "driver_family_name": driver["familyName"],
                    "constructor_id": constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid": pd.to_numeric(result.get("grid"), errors="coerce"),
                    "finish_position": pd.to_numeric(result.get("position"), errors="coerce"),
                    "points": pd.to_numeric(result.get("points"), errors="coerce"),
                    "status": result.get("status"),
                }
            )
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    # Older seasons can contain duplicate driver entries for a race (shared drives).
    # Keep the most relevant row per driver-race key for stable modeling.
    results_df = results_df.sort_values(
        ["season", "round", "driver_id", "points", "finish_position"],
        ascending=[True, True, True, False, True],
    )
    results_df = results_df.drop_duplicates(
        subset=["season", "round", "driver_id"], keep="first"
    )
    return results_df


def merge_qualifying(
    results_df: pd.DataFrame, qualifying_payload: list[dict]
) -> pd.DataFrame:
    q_rows: list[dict] = []
    for race in qualifying_payload:
        season = int(race["season"])
        round_number = int(race["round"])
        for q in race["QualifyingResults"]:
            driver = q["Driver"]
            q_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver["driverId"],
                    "qualifying_position": pd.to_numeric(q.get("position"), errors="coerce"),
                }
            )
    q_df = pd.DataFrame(q_rows)
    if q_df.empty:
        return results_df
    q_df = q_df.sort_values(
        ["season", "round", "driver_id", "qualifying_position"],
        ascending=[True, True, True, True],
    ).drop_duplicates(subset=["season", "round", "driver_id"], keep="first")
    return results_df.merge(
        q_df, on=["season", "round", "driver_id"], how="left", validate="1:1"
    )


def main() -> None:
    ensure_directories()
    results_path = RAW_DIR / "results.json"
    qualifying_path = RAW_DIR / "qualifying.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "Missing raw results file. Run src/data/download_data.py first."
        )

    results_payload = _load_json(results_path)
    results_df = build_results_table(results_payload)

    if qualifying_path.exists():
        qualifying_payload = _load_json(qualifying_path)
        model_df = merge_qualifying(results_df, qualifying_payload)
    else:
        model_df = results_df.copy()

    model_df["target_podium"] = (model_df["finish_position"] <= 3).astype(int)
    model_df["target_points"] = (model_df["points"] > 0).astype(int)

    output_path = PROCESSED_DIR / "f1_model_table.csv"
    model_df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")
    print(f"Rows: {len(model_df):,} | Columns: {len(model_df.columns)}")


if __name__ == "__main__":
    main()
