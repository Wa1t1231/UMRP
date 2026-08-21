# Dataset Card

## Overview

This repository contains a fully synthetic dataset for a hybrid agricultural decision-support-system experiment. It represents daily perishable-produce supply, free-text customer orders, structured ground truth, simulated parser predictions, and allocation outputs.

No record is a real farm observation, no customer is a real person or organization, and no live LLM endpoint was called.

## Dataset scope

| Item | Value |
|---|---:|
| Random seed | 20260820 |
| Operating period | 2025-01-01 to 2025-03-31 |
| Daily cycles | 90 |
| Development/test split | 60/30 days |
| Products | 8 |
| Anonymous customers | 12 |
| Supply lots | 5,445 |
| Order messages | 1,012 |
| Ground-truth order lines | 2,515 |
| Held-out test messages | 330 |
| Held-out test order lines | 843 |

## Main files

- `dataset/products.csv`: product master and unit conversions.
- `dataset/customers.csv`: anonymous customer segments and priority weights.
- `dataset/grade_compatibility.csv`: allowed grade mappings and penalties.
- `dataset/supply.csv`: dated, graded, age-bucketed supply lots.
- `dataset/order_messages.csv`: synthetic free-text messages and split labels.
- `dataset/order_lines_ground_truth.csv`: normalized order-line truth.
- `dataset/parser_predictions.csv`: predictions from a seeded error simulator, not a live LLM.
- `dataset/synthetic_agri_dss.sqlite`: relational copy of inputs and result tables.
- `dataset/data_dictionary.csv`: field-level descriptions.
- `dataset/experiment_metadata.json`: provenance, scope, and sample counts.

## Intended uses

- Reproducing the dissertation experiment and statistical analysis.
- Testing Python, SQL, or spreadsheet analysis pipelines.
- Prototyping a constrained allocation DSS before real data are available.
- Teaching reproducible simulation and human-in-the-loop evaluation.

## Inappropriate uses

- Claiming that the results measure a real farm, employee, customer, or LLM.
- Treating simulated user ratings or decision times as real user-study evidence.
- Deploying allocation recommendations without domain review and real-data validation.
- Using the dataset as proof of external validity or production performance.

## Reproduction

Run `python run_pipeline.py` from the repository root. The default seed rebuilds the committed CSV and SQLite inputs and then reproduces all experiment stages. Runtime measurements may vary slightly across machines; allocation and evaluation logic remain fixed.

## Known limitations

The generator covers one 90-day synthetic period, eight products, two grades, and twelve anonymous customers. Demand patterns, parser errors, human ratings, override records, and decision times are simulated. Real deployments require longer historical periods, local business rules, actual inventory data, live parser evaluation, and human-subject governance where applicable.
