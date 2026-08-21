from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTest(unittest.TestCase):
    def test_dataset_shape_and_split(self) -> None:
        dataset = ROOT / "dataset"
        messages = pd.read_csv(dataset / "order_messages.csv")
        order_lines = pd.read_csv(dataset / "order_lines_ground_truth.csv")

        self.assertEqual(len(messages), 1012)
        self.assertEqual(len(order_lines), 2515)
        self.assertEqual(messages.loc[messages["split"] == "test", "message_id"].nunique(), 330)
        test_dates = messages.loc[messages["split"] == "test", "allocation_date"].nunique()
        self.assertEqual(test_dates, 30)

    def test_experiment_summary(self) -> None:
        summary = json.loads(
            (ROOT / "results" / "experiment_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(summary["all_robustness_checks_passed"])
        self.assertTrue(all(summary["hypotheses_supported"].values()))
        self.assertEqual(summary["sensitivity_scenarios"], 27)
        self.assertEqual(summary["ablation_scenarios"], 4)

    def test_sqlite_integrity(self) -> None:
        database = ROOT / "dataset" / "synthetic_agri_dss.sqlite"
        with sqlite3.connect(database) as connection:
            status = connection.execute("PRAGMA integrity_check").fetchone()[0]
        self.assertEqual(status, "ok")


if __name__ == "__main__":
    unittest.main()
