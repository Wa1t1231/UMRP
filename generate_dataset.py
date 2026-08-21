from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260820
DATASET_VERSION = "synthetic-agri-dss-v1.0"


PRODUCTS = [
    ("P01", "Tomato", "番茄", "tomatoes|tomato|toms", 3, 3.20, 5.0, 10.0, 105),
    ("P02", "Lettuce", "生菜", "lettuce|lettuces|lett", 3, 2.60, 4.0, 8.0, 72),
    ("P03", "Cucumber", "黄瓜", "cucumber|cucumbers|cuke", 4, 2.10, 6.0, 12.0, 88),
    ("P04", "Strawberry", "草莓", "strawberry|strawberries|berries", 3, 6.80, 3.0, 6.0, 55),
    ("P05", "Bell Pepper", "甜椒", "bell pepper|peppers|capsicum", 5, 4.10, 5.0, 10.0, 64),
    ("P06", "Spinach", "菠菜", "spinach|spin|leaf spinach", 3, 3.00, 3.0, 6.0, 50),
    ("P07", "Carrot", "胡萝卜", "carrot|carrots|car", 7, 1.80, 8.0, 16.0, 95),
    ("P08", "Broccoli", "西兰花", "broccoli|broc|brocc", 5, 3.70, 5.0, 10.0, 60),
]


CUSTOMERS = [
    ("C001", "Key retail", 2.00, "protected", True),
    ("C002", "Key retail", 1.90, "protected", True),
    ("C003", "Hotel/restaurant", 1.65, "standard", True),
    ("C004", "Hotel/restaurant", 1.55, "standard", True),
    ("C005", "Wholesale", 1.35, "standard", True),
    ("C006", "Wholesale", 1.30, "standard", True),
    ("C007", "Local shop", 1.15, "standard", False),
    ("C008", "Local shop", 1.10, "standard", False),
    ("C009", "Food service", 1.45, "standard", True),
    ("C010", "Food service", 1.40, "standard", True),
    ("C011", "Market stall", 1.00, "spot", False),
    ("C012", "Market stall", 0.95, "spot", False),
]


def product_frame() -> pd.DataFrame:
    return pd.DataFrame(
        PRODUCTS,
        columns=[
            "product_id",
            "product_name_en",
            "product_name_zh",
            "aliases",
            "shelf_life_days",
            "unit_price_per_kg",
            "kg_per_box",
            "kg_per_crate",
            "base_daily_supply_kg",
        ],
    )


def customer_frame() -> pd.DataFrame:
    return pd.DataFrame(
        CUSTOMERS,
        columns=[
            "customer_id",
            "customer_segment",
            "priority_weight",
            "contract_class",
            "default_substitution_allowed",
        ],
    )


def compatibility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "A", 1, 0.0),
            ("A", "B", 1, 0.55),
            ("B", "A", 0, 999.0),
            ("B", "B", 1, 0.0),
        ],
        columns=[
            "supply_grade",
            "requested_grade",
            "compatible",
            "substitution_penalty_per_kg",
        ],
    )


def generate_supply(rng: np.random.Generator, products: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict] = []
    supply_counter = 1
    for day_idx, date in enumerate(dates):
        seasonal = 1.0 + 0.12 * np.sin(2.0 * np.pi * day_idx / 30.0)
        for product in products.itertuples(index=False):
            total = int(max(15, round(product.base_daily_supply_kg * seasonal * rng.lognormal(0.0, 0.20))))
            share_a = float(np.clip(rng.beta(6.0, 3.5), 0.45, 0.85))
            grade_totals = {"A": int(round(total * share_a)), "B": total - int(round(total * share_a))}
            life = int(product.shelf_life_days)
            alpha = np.linspace(5.5, 1.2, life)
            for grade, grade_total in grade_totals.items():
                shares = rng.dirichlet(alpha)
                quantities = np.floor(shares * grade_total).astype(int)
                remainder = int(grade_total - quantities.sum())
                for idx in np.argsort(-shares)[:remainder]:
                    quantities[idx] += 1
                for age, qty in enumerate(quantities.tolist()):
                    if qty <= 0:
                        continue
                    rows.append(
                        {
                            "supply_id": f"S{supply_counter:07d}",
                            "allocation_date": date.strftime("%Y-%m-%d"),
                            "product_id": product.product_id,
                            "supply_grade": grade,
                            "age_days": age,
                            "shelf_life_days": life,
                            "usable_quantity_kg": float(qty),
                            "expires_next_day": int(age + 1 >= life),
                            "unit_value_per_kg": float(product.unit_price_per_kg),
                        }
                    )
                    supply_counter += 1
    return pd.DataFrame(rows)


def choose_unit_and_quantity(
    rng: np.random.Generator, desired_kg: float, product: pd.Series
) -> tuple[float, str, float]:
    unit = str(rng.choice(["kg", "box", "crate"], p=[0.70, 0.20, 0.10]))
    factor = 1.0 if unit == "kg" else float(product["kg_per_box"] if unit == "box" else product["kg_per_crate"])
    raw_qty = max(1.0, float(round(desired_kg / factor)))
    return raw_qty, unit, raw_qty * factor


def render_message(date: pd.Timestamp, customer_id: str, lines: list[dict], difficulty: str, rng: np.random.Generator) -> str:
    fragments = []
    for line in lines:
        sub = "substitution OK" if line["substitution_allowed"] else "no substitutes"
        fragments.append(
            f'{line["requested_quantity"]:g} {line["unit"]} {line["raw_product_text"]} '
            f'grade {line["requested_grade"]}, {sub}'
        )
    body = "; also ".join(fragments)
    if difficulty == "hard":
        starters = ["Hi team pls arrange", "Amendment - need", "Morning, can u send"]
        tail = str(rng.choice(["confirm asap", "same delivery slot", "call if any issue"]))
        return f"{rng.choice(starters)} {body} for {date.strftime('%d/%m')}; {tail}. Ref {customer_id[-2:]}"
    return f"Please deliver {body} on {date.strftime('%Y-%m-%d')}. Customer ref {customer_id}."


def generate_orders(
    rng: np.random.Generator,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    supply: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_map = products.set_index("product_id")
    customer_ids = customers["customer_id"].tolist()
    draft_lines: list[dict] = []
    line_counter = 1

    daily_supply = supply.groupby(["allocation_date", "product_id"], as_index=False)["usable_quantity_kg"].sum()
    for day_idx, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        for product_id in products["product_id"]:
            supply_qty = float(
                daily_supply.loc[
                    (daily_supply["allocation_date"] == date_str) & (daily_supply["product_id"] == product_id),
                    "usable_quantity_kg",
                ].iloc[0]
            )
            demand_multiplier = float(np.clip(rng.normal(1.06, 0.22), 0.62, 1.52))
            desired_total = max(8.0, supply_qty * demand_multiplier)
            n_lines = int(rng.integers(2, 6))
            weights = rng.dirichlet(np.ones(n_lines) * 1.8)
            chosen_customers = rng.choice(customer_ids, size=n_lines, replace=False)
            product = product_map.loc[product_id]
            aliases = str(product["aliases"]).split("|")
            for customer_id, weight in zip(chosen_customers, weights):
                raw_qty, unit, qty_kg = choose_unit_and_quantity(rng, max(3.0, desired_total * weight), product)
                requested_grade = str(rng.choice(["A", "B"], p=[0.56, 0.44]))
                substitution_allowed = bool(rng.random() < (0.70 if requested_grade == "B" else 0.12))
                draft_lines.append(
                    {
                        "order_line_id": f"O{line_counter:07d}",
                        "allocation_date": date_str,
                        "customer_id": str(customer_id),
                        "product_id": product_id,
                        "raw_product_text": str(rng.choice(aliases)),
                        "requested_quantity": raw_qty,
                        "unit": unit,
                        "requested_quantity_kg": float(qty_kg),
                        "requested_grade": requested_grade,
                        "delivery_date": date_str,
                        "substitution_allowed": substitution_allowed,
                        "notes": "synthetic order line",
                        "day_index": day_idx,
                    }
                )
                line_counter += 1

    draft = pd.DataFrame(draft_lines)
    messages: list[dict] = []
    message_counter = 1
    message_ids: dict[tuple[str, str], str] = {}
    for (date_str, customer_id), group in draft.groupby(["allocation_date", "customer_id"], sort=True):
        date = pd.Timestamp(date_str)
        difficulty = "hard" if rng.random() < 0.26 else "standard"
        message_id = f"M{message_counter:06d}"
        message_ids[(date_str, customer_id)] = message_id
        raw_message = render_message(date, customer_id, group.to_dict("records"), difficulty, rng)
        day_index = int(group["day_index"].iloc[0])
        messages.append(
            {
                "message_id": message_id,
                "allocation_date": date_str,
                "authorized_customer_id": customer_id,
                "raw_message": raw_message,
                "message_difficulty": difficulty,
                "split": "development" if day_index < 60 else "test",
                "message_hash_sha256": hashlib.sha256(raw_message.encode("utf-8")).hexdigest(),
            }
        )
        message_counter += 1

    draft["message_id"] = [message_ids[(d, c)] for d, c in zip(draft["allocation_date"], draft["customer_id"])]
    order_lines = draft[
        [
            "order_line_id",
            "message_id",
            "allocation_date",
            "customer_id",
            "product_id",
            "raw_product_text",
            "requested_quantity",
            "unit",
            "requested_quantity_kg",
            "requested_grade",
            "delivery_date",
            "substitution_allowed",
            "notes",
        ]
    ].copy()
    return pd.DataFrame(messages), order_lines


def different_choice(rng: np.random.Generator, current: str, choices: list[str]) -> str:
    candidates = [x for x in choices if x != current]
    return str(rng.choice(candidates))


def generate_parser_predictions(
    rng: np.random.Generator,
    order_lines: pd.DataFrame,
    messages: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    product_ids = products["product_id"].tolist()
    customer_ids = customers["customer_id"].tolist()
    product_map = products.set_index("product_id")
    message_meta = messages.set_index("message_id").to_dict("index")
    rows: list[dict] = []

    base_probs = {
        "customer": (0.010, 0.030),
        "product": (0.020, 0.065),
        "quantity": (0.025, 0.075),
        "unit": (0.018, 0.055),
        "grade": (0.025, 0.075),
        "date": (0.015, 0.050),
        "substitution": (0.035, 0.095),
    }

    message_invalid: dict[str, bool] = {}
    for message in messages.itertuples(index=False):
        p = 0.012 if message.message_difficulty == "standard" else 0.038
        message_invalid[message.message_id] = bool(rng.random() < p)

    for truth in order_lines.itertuples(index=False):
        meta = message_meta[truth.message_id]
        hard = meta["message_difficulty"] == "hard"
        invalid_json = message_invalid[truth.message_id]
        pred = {
            "customer": truth.customer_id,
            "product": truth.product_id,
            "quantity": float(truth.requested_quantity),
            "unit": truth.unit,
            "grade": truth.requested_grade,
            "date": truth.delivery_date,
            "substitution": bool(truth.substitution_allowed),
        }
        injected: list[str] = []
        detected: list[str] = []

        if invalid_json:
            for key in pred:
                pred[key] = None
            injected.append("INVALID_JSON")
            detected.append("INVALID_JSON")
        else:
            for field, (p_standard, p_hard) in base_probs.items():
                if rng.random() >= (p_hard if hard else p_standard):
                    continue
                injected.append(field.upper())
                make_null = rng.random() < 0.36
                if make_null:
                    pred[field] = None
                    detected.append(f"MISSING_{field.upper()}")
                elif field == "customer":
                    pred[field] = different_choice(rng, str(truth.customer_id), customer_ids)
                elif field == "product":
                    pred[field] = different_choice(rng, str(truth.product_id), product_ids)
                elif field == "quantity":
                    pred[field] = float(max(1.0, round(float(truth.requested_quantity) * float(rng.choice([0.5, 0.8, 1.2, 1.5])))))
                elif field == "unit":
                    pred[field] = different_choice(rng, str(truth.unit), ["kg", "box", "crate"])
                elif field == "grade":
                    pred[field] = "B" if truth.requested_grade == "A" else "A"
                elif field == "date":
                    offset = int(rng.choice([-1, 1]))
                    pred[field] = (pd.Timestamp(truth.delivery_date) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                elif field == "substitution":
                    pred[field] = not bool(truth.substitution_allowed)
                if not make_null and rng.random() < (0.50 if hard else 0.38):
                    detected.append(f"LOW_CONFIDENCE_{field.upper()}")

        pred_qty_kg = None
        if pred["product"] is not None and pred["quantity"] is not None and pred["unit"] is not None:
            product = product_map.loc[str(pred["product"])]
            factor = 1.0 if pred["unit"] == "kg" else float(product["kg_per_box"] if pred["unit"] == "box" else product["kg_per_crate"])
            pred_qty_kg = float(pred["quantity"]) * factor

        rows.append(
            {
                "order_line_id": truth.order_line_id,
                "message_id": truth.message_id,
                "pred_customer_id": pred["customer"],
                "pred_product_id": pred["product"],
                "pred_requested_quantity": pred["quantity"],
                "pred_unit": pred["unit"],
                "pred_requested_quantity_kg": pred_qty_kg,
                "pred_requested_grade": pred["grade"],
                "pred_delivery_date": pred["date"],
                "pred_substitution_allowed": pred["substitution"],
                "json_valid": not invalid_json,
                "detected_issue_codes": "|".join(sorted(set(detected))),
                "injected_error_fields": "|".join(sorted(set(injected))),
                "prediction_source": "synthetic_error_model_not_real_llm",
            }
        )

    predictions = pd.DataFrame(rows)
    line_flags = predictions.groupby("message_id")["detected_issue_codes"].apply(lambda s: any(bool(x) for x in s))
    confirmation_map: dict[str, bool] = {}
    for message in messages.itertuples(index=False):
        false_alert = bool(message.message_difficulty == "hard" and rng.random() < 0.06)
        confirmation_map[message.message_id] = bool(line_flags.get(message.message_id, False) or false_alert)
    predictions["needs_human_confirmation"] = predictions["message_id"].map(confirmation_map)
    return predictions


def data_dictionary_frame() -> pd.DataFrame:
    rows = [
        ("products", "product_id", "匿名产品主键", "text"),
        ("products", "shelf_life_days", "最大可用货架期", "days"),
        ("products", "unit_price_per_kg", "合成单位价值，仅用于目标函数", "currency/kg"),
        ("customers", "customer_id", "匿名客户主键", "text"),
        ("customers", "priority_weight", "客户优先权重", "0.95-2.00"),
        ("grade_compatibility", "compatible", "供给等级能否满足请求等级", "0/1"),
        ("supply", "supply_id", "库存批次主键", "text"),
        ("supply", "age_days", "库存年龄", "days"),
        ("supply", "usable_quantity_kg", "当日可用供给", "kg"),
        ("supply", "expires_next_day", "若当日未分配则次日不可用", "0/1"),
        ("order_messages", "raw_message", "合成非结构化订单消息", "text"),
        ("order_messages", "split", "提示开发集或冻结测试集", "development/test"),
        ("order_lines_ground_truth", "requested_quantity_kg", "统一换算后的真实请求量", "kg"),
        ("order_lines_ground_truth", "substitution_allowed", "客户是否允许等级替代", "boolean"),
        ("parser_predictions", "pred_*", "合成解析器输出字段", "mixed"),
        ("parser_predictions", "json_valid", "模拟输出是否为有效JSON", "boolean"),
        ("parser_predictions", "needs_human_confirmation", "是否进入人工确认门", "boolean"),
        ("parser_predictions", "prediction_source", "明确该结果并非真实LLM调用", "text"),
    ]
    return pd.DataFrame(rows, columns=["table_name", "column_name", "description_zh", "unit_or_type"])


def save_sqlite(path: Path, tables: dict[str, pd.DataFrame], metadata: dict) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, if_exists="replace", index=False)
        pd.DataFrame([{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)} for key, value in metadata.items()]).to_sql(
            "experiment_metadata", con, if_exists="replace", index=False
        )
        con.execute("CREATE INDEX idx_supply_date_product ON supply(allocation_date, product_id)")
        con.execute("CREATE INDEX idx_orders_date_product ON order_lines_ground_truth(allocation_date, product_id)")
        con.execute("CREATE INDEX idx_messages_split ON order_messages(split)")
        con.execute("CREATE INDEX idx_predictions_line ON parser_predictions(order_line_id)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dissertation experiment dataset.")
    parser.add_argument("--output-dir", default="../dataset", help="Dataset output directory")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    products = product_frame()
    customers = customer_frame()
    compatibility = compatibility_frame()
    dates = pd.date_range("2025-01-01", periods=args.days, freq="D")
    supply = generate_supply(rng, products, dates)
    messages, order_lines = generate_orders(rng, products, customers, supply, dates)
    predictions = generate_parser_predictions(rng, order_lines, messages, products, customers)
    dictionary = data_dictionary_frame()

    tables = {
        "products": products,
        "customers": customers,
        "grade_compatibility": compatibility,
        "supply": supply,
        "order_messages": messages,
        "order_lines_ground_truth": order_lines,
        "parser_predictions": predictions,
        "data_dictionary": dictionary,
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    daily_supply = supply.groupby("allocation_date")["usable_quantity_kg"].sum()
    daily_demand = order_lines.groupby("allocation_date")["requested_quantity_kg"].sum()
    metadata = {
        "dataset_version": DATASET_VERSION,
        "synthetic_data_notice": "All records are generated for reproducible simulation; they are not observations from a real farm and parser predictions are not outputs from a real LLM endpoint.",
        "seed": int(args.seed),
        "date_start": dates.min().strftime("%Y-%m-%d"),
        "date_end": dates.max().strftime("%Y-%m-%d"),
        "operating_days": int(args.days),
        "development_days": 60,
        "test_days": int(args.days - 60),
        "customers": int(len(customers)),
        "products": int(len(products)),
        "supply_lots": int(len(supply)),
        "messages": int(len(messages)),
        "order_lines": int(len(order_lines)),
        "total_supply_kg": float(supply["usable_quantity_kg"].sum()),
        "total_demand_kg": float(order_lines["requested_quantity_kg"].sum()),
        "days_demand_exceeds_supply": int((daily_demand > daily_supply).sum()),
        "test_definition": "Chronological holdout: first 60 days development, last 30 days test.",
    }
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_sqlite(out / "synthetic_agri_dss.sqlite", tables, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
