import argparse
import json
import time
from pathlib import Path

import requests

from src.utils.config import RAW_DIR, ensure_directories


ERGAST_BASES = [
    "https://ergast.com/api/f1",
    "https://api.jolpi.ca/ergast/f1",
]


def _get_json(endpoint: str, limit: int, retries: int = 3, retry_delay_s: float = 2.0) -> dict:
    params = {"limit": limit}
    headers = {"User-Agent": "f1-prediction-model/1.0"}
    errors: list[str] = []

    for base_url in ERGAST_BASES:
        url = f"{base_url}/{endpoint}.json"
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                errors.append(f"{url} attempt {attempt}: {exc}")
                if attempt < retries:
                    time.sleep(retry_delay_s * attempt)

    error_text = "\n".join(errors[-6:])
    raise RuntimeError(
        "Failed to fetch endpoint from all configured APIs.\n"
        f"Endpoint: {endpoint}\n"
        f"Params: {params}\n"
        f"Recent errors:\n{error_text}"
    )


def _get_seasons() -> list[int]:
    payload = _get_json("seasons", limit=1000)
    seasons_raw = payload["MRData"]["SeasonTable"]["Seasons"]
    return [int(item["season"]) for item in seasons_raw]


def download_results(limit: int = 2000) -> Path:
    # Pull per season to avoid API max-limit truncation on global endpoints.
    races: list[dict] = []
    for season in _get_seasons():
        payload = _get_json(f"{season}/results", limit=limit)
        races.extend(payload["MRData"]["RaceTable"]["Races"])
    output_path = RAW_DIR / "results.json"
    output_path.write_text(json.dumps(races), encoding="utf-8")
    return output_path


def download_qualifying(limit: int = 2000) -> Path:
    races: list[dict] = []
    for season in _get_seasons():
        payload = _get_json(f"{season}/qualifying", limit=limit)
        races.extend(payload["MRData"]["RaceTable"]["Races"])
    output_path = RAW_DIR / "qualifying.json"
    output_path.write_text(json.dumps(races), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download F1 data from Ergast API.")
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Per-season API limit (kept high to avoid truncation).",
    )
    args = parser.parse_args()

    ensure_directories()
    results_path = download_results(limit=args.limit)
    qualifying_path = download_qualifying(limit=args.limit)

    print(f"Saved results to: {results_path}")
    print(f"Saved qualifying to: {qualifying_path}")


if __name__ == "__main__":
    main()
