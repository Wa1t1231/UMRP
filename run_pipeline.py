"""Run the complete synthetic-data and experiment pipeline.

This wrapper is platform-independent and keeps all generated files inside the
repository's dataset, results, and figures directories.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20260820


def run(command: list[str]) -> None:
    """Run one pipeline stage and stop immediately if it fails."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the synthetic agricultural DSS dataset and results."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Reuse the committed dataset and rerun only the experiments.",
    )
    args = parser.parse_args()

    if not args.skip_dataset:
        run(
            [
                sys.executable,
                str(ROOT / "code" / "generate_dataset.py"),
                "--output-dir",
                str(ROOT / "dataset"),
                "--seed",
                str(args.seed),
            ]
        )

    run(
        [
            sys.executable,
            str(ROOT / "code" / "run_experiments.py"),
            "--dataset-dir",
            str(ROOT / "dataset"),
            "--results-dir",
            str(ROOT / "results"),
            "--figures-dir",
            str(ROOT / "figures"),
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()
