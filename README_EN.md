# Synthetic Agricultural DSS: Reproducible Dataset and Experiment

This repository contains the code, generated dataset, and results for a reproducible hybrid decision-support-system experiment. The system converts synthetic free-text agricultural orders into structured records, validates them, allocates perishable supply with a minimum-cost-flow solver equivalent to the stated transportation LP, and records grounded explanations and simulated human review.

> All data and user-facing measurements are synthetic. No real farm records and no live LLM endpoint outputs are included.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python validate_repository.py
python -m unittest discover -s tests -v
```

To reuse the committed dataset and rerun only the experiments:

```bash
python run_pipeline.py --skip-dataset
```

## Repository structure

```text
.
|-- code/               # generator, experiment runner, SQL schema, PowerShell runner
|-- dataset/            # CSV inputs, data dictionary, metadata, and SQLite database
|-- results/            # experiment outputs and statistical tests
|-- figures/            # generated figure
|-- tests/              # repository validation tests
|-- run_pipeline.py     # cross-platform entry point
|-- validate_repository.py
|-- DATASET_CARD.md
|-- CITATION.cff
`-- requirements.txt
```

The frozen test set consists of the last 30 days: 330 messages and 843 order lines. See `README.md` for the full Chinese documentation and `DATASET_CARD.md` for intended use and limitations.

## Public release note

No software or dataset license is selected in this package. Before making the repository public, choose a license that is compatible with university and supervisor requirements.
