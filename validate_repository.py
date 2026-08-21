"""Validate the committed dataset and headline experiment outputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    dataset = root / "dataset"
    results = root / "results"
    metadata = json.loads((dataset / "experiment_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((results / "experiment_summary.json").read_text(encoding="utf-8"))

    require(metadata["seed"] == 20260820, "Unexpected dataset seed")
    require(metadata["operating_days"] == 90, "Expected 90 operating days")
    require(metadata["products"] == 8, "Expected 8 products")
    require(metadata["customers"] == 12, "Expected 12 customers")
    require(metadata["supply_lots"] == 5445, "Expected 5,445 supply lots")
    require(metadata["messages"] == 1012, "Expected 1,012 messages")
    require(metadata["order_lines"] == 2515, "Expected 2,515 order lines")
    require("not observations from a real farm" in metadata["synthetic_data_notice"], "Synthetic-data notice is missing")

    expected_rows = {
        "products.csv": 8,
        "customers.csv": 12,
        "grade_compatibility.csv": 4,
        "supply.csv": 5445,
        "order_messages.csv": 1012,
        "order_lines_ground_truth.csv": 2515,
        "parser_predictions.csv": 2515,
    }
    for name, expected in expected_rows.items():
        actual = len(pd.read_csv(dataset / name))
        require(actual == expected, f"{name}: expected {expected} rows, found {actual}")

    require(summary["test_days"] == 30, "Expected a 30-day held-out test period")
    require(summary["test_order_lines"] == 843, "Expected 843 held-out order lines")
    require(summary["all_robustness_checks_passed"] is True, "A robustness check failed")
    require(all(summary["hypotheses_supported"].values()), "At least one directional hypothesis was not supported")

    database = dataset / "synthetic_agri_dss.sqlite"
    with sqlite3.connect(database) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        require(integrity == "ok", f"SQLite integrity check returned: {integrity}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    require({"products", "customers", "supply", "daily_metrics"}.issubset(tables), "SQLite tables are incomplete")

    print("Repository validation passed.")
    print(json.dumps({
        "dataset_version": metadata["dataset_version"],
        "seed": metadata["seed"],
        "dataset_rows_checked": sum(expected_rows.values()),
        "test_days": summary["test_days"],
        "robustness_checks_passed": summary["all_robustness_checks_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
