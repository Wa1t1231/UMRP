from __future__ import annotations

import argparse
import heapq
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260820
DEFAULT_WEIGHTS = {"lambda_priority": 4.0, "lambda_substitution": 2.0, "lambda_waste": 6.0}


@dataclass
class Edge:
    to: int
    rev: int
    cap: float
    cost: float
    initial_cap: float
    tag: dict | None = None


class MinCostFlow:
    """Successive shortest augmenting path solver for the dissertation's LP network."""

    def __init__(self, n: int) -> None:
        self.graph: list[list[Edge]] = [[] for _ in range(n)]

    def add_edge(self, source: int, target: int, capacity: float, cost: float, tag: dict | None = None) -> tuple[int, int]:
        forward = Edge(target, len(self.graph[target]), float(capacity), float(cost), float(capacity), tag)
        reverse = Edge(source, len(self.graph[source]), 0.0, -float(cost), 0.0, None)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return source, len(self.graph[source]) - 1

    def solve(self, source: int, sink: int, initial_potential: list[float]) -> tuple[float, float]:
        n = len(self.graph)
        potential = list(initial_potential)
        total_flow = 0.0
        total_cost = 0.0
        eps = 1e-9

        while True:
            dist = [math.inf] * n
            prev_node = [-1] * n
            prev_edge = [-1] * n
            dist[source] = 0.0
            queue: list[tuple[float, int]] = [(0.0, source)]
            while queue:
                current_dist, node = heapq.heappop(queue)
                if current_dist > dist[node] + eps:
                    continue
                for edge_idx, edge in enumerate(self.graph[node]):
                    if edge.cap <= eps:
                        continue
                    reduced = edge.cost + potential[node] - potential[edge.to]
                    if reduced < 0.0 and reduced > -1e-7:
                        reduced = 0.0
                    if reduced < -1e-7:
                        raise RuntimeError(f"Negative reduced cost {reduced:.6g}; invalid initial potential")
                    candidate = current_dist + reduced
                    if candidate + eps < dist[edge.to]:
                        dist[edge.to] = candidate
                        prev_node[edge.to] = node
                        prev_edge[edge.to] = edge_idx
                        heapq.heappush(queue, (candidate, edge.to))

            if math.isinf(dist[sink]):
                break
            for node in range(n):
                if not math.isinf(dist[node]):
                    potential[node] += dist[node]
            if potential[sink] >= -eps:
                break

            augment = math.inf
            node = sink
            while node != source:
                parent = prev_node[node]
                if parent < 0:
                    augment = 0.0
                    break
                edge = self.graph[parent][prev_edge[node]]
                augment = min(augment, edge.cap)
                node = parent
            if augment <= eps or math.isinf(augment):
                break

            node = sink
            while node != source:
                parent = prev_node[node]
                edge_idx = prev_edge[node]
                edge = self.graph[parent][edge_idx]
                edge.cap -= augment
                self.graph[node][edge.rev].cap += augment
                total_cost += augment * edge.cost
                node = parent
            total_flow += augment
        return total_flow, total_cost


def load_data(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "products",
        "customers",
        "grade_compatibility",
        "supply",
        "order_messages",
        "order_lines_ground_truth",
        "parser_predictions",
    ]
    return {name: pd.read_csv(dataset_dir / f"{name}.csv") for name in names}


def build_p1_orders(truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = truth.merge(predictions, on=["order_line_id", "message_id"], how="left", validate="one_to_one")
    confirmed = merged["needs_human_confirmation"].fillna(True).astype(bool)

    def choose(truth_col: str, pred_col: str) -> pd.Series:
        return merged[truth_col].where(confirmed, merged[pred_col])

    p1 = pd.DataFrame(
        {
            "order_line_id": merged["order_line_id"],
            "message_id": merged["message_id"],
            "allocation_date": merged["allocation_date"],
            "customer_id": choose("customer_id", "pred_customer_id"),
            "product_id": choose("product_id", "pred_product_id"),
            "requested_quantity_kg": choose("requested_quantity_kg", "pred_requested_quantity_kg"),
            "requested_grade": choose("requested_grade", "pred_requested_grade"),
            "delivery_date": choose("delivery_date", "pred_delivery_date"),
            "substitution_allowed": choose("substitution_allowed", "pred_substitution_allowed"),
            "human_corrected": confirmed,
        }
    )
    required = ["customer_id", "product_id", "requested_quantity_kg", "requested_grade", "delivery_date", "substitution_allowed"]
    p1["solver_eligible"] = p1[required].notna().all(axis=1) & (p1["requested_quantity_kg"].fillna(0) > 0)
    p1["solver_eligible"] &= p1["delivery_date"].astype(str).eq(p1["allocation_date"].astype(str))
    return p1


def compatible(
    supply_grade: str,
    requested_grade: str,
    substitution_allowed: bool,
    compatibility_map: dict[tuple[str, str], tuple[int, float]],
) -> tuple[bool, float]:
    allowed, penalty = compatibility_map[(supply_grade, requested_grade)]
    if supply_grade != requested_grade and not bool(substitution_allowed):
        return False, penalty
    return bool(allowed), float(penalty)


def solve_optimized_day(
    date: str,
    supply_day: pd.DataFrame,
    orders_day: pd.DataFrame,
    customers: pd.DataFrame,
    compatibility_map: dict[tuple[str, str], tuple[int, float]],
    weights: dict[str, float],
    condition: str,
) -> tuple[pd.DataFrame, float, float, str]:
    started = time.perf_counter()
    allocations: list[dict] = []
    total_objective = 0.0
    priority_map = customers.set_index("customer_id")["priority_weight"].to_dict()

    eligible = orders_day.loc[orders_day.get("solver_eligible", pd.Series(True, index=orders_day.index)).fillna(False).astype(bool)].copy()
    for product_id in sorted(set(supply_day["product_id"]) | set(eligible["product_id"].dropna().astype(str))):
        lots = supply_day.loc[supply_day["product_id"] == product_id].reset_index(drop=True)
        orders = eligible.loc[eligible["product_id"] == product_id].reset_index(drop=True)
        if lots.empty or orders.empty:
            continue

        lot_count = len(lots)
        order_count = len(orders)
        source = 0
        first_lot = 1
        first_order = first_lot + lot_count
        sink = first_order + order_count
        network = MinCostFlow(sink + 1)
        initial_potential = [0.0] * (sink + 1)
        tracked_edges: list[tuple[int, int, dict]] = []

        for lot_idx, lot in lots.iterrows():
            network.add_edge(source, first_lot + lot_idx, float(lot["usable_quantity_kg"]), 0.0)
        for order_idx, order in orders.iterrows():
            network.add_edge(first_order + order_idx, sink, float(order["requested_quantity_kg"]), 0.0)

        order_min_costs = [math.inf] * order_count
        for lot_idx, lot in lots.iterrows():
            risk = (float(lot["age_days"]) + 1.0) / max(float(lot["shelf_life_days"]), 1.0)
            waste_penalty = risk * risk * (3.0 if int(lot["expires_next_day"]) else 1.0)
            for order_idx, order in orders.iterrows():
                is_compatible, substitution_penalty = compatible(
                    str(lot["supply_grade"]),
                    str(order["requested_grade"]),
                    bool(order["substitution_allowed"]),
                    compatibility_map,
                )
                if not is_compatible:
                    continue
                priority = float(priority_map.get(str(order["customer_id"]), 1.0))
                score = (
                    float(lot["unit_value_per_kg"])
                    + weights["lambda_priority"] * priority
                    - weights["lambda_substitution"] * substitution_penalty
                    + weights["lambda_waste"] * waste_penalty
                    + 1e-5 * float(lot["age_days"])
                )
                tag = {
                    "allocation_date": date,
                    "condition": condition,
                    "order_line_id": str(order["order_line_id"]),
                    "message_id": str(order.get("message_id", "")),
                    "supply_id": str(lot["supply_id"]),
                    "customer_id": str(order["customer_id"]),
                    "product_id": str(product_id),
                    "supply_grade": str(lot["supply_grade"]),
                    "requested_grade": str(order["requested_grade"]),
                    "delivery_date": str(order["delivery_date"]),
                    "objective_unit_score": score,
                }
                node, edge_idx = network.add_edge(
                    first_lot + lot_idx,
                    first_order + order_idx,
                    min(float(lot["usable_quantity_kg"]), float(order["requested_quantity_kg"])),
                    -score,
                    tag,
                )
                tracked_edges.append((node, edge_idx, tag))
                order_min_costs[order_idx] = min(order_min_costs[order_idx], -score)

        reachable_order_costs = []
        for order_idx, min_cost in enumerate(order_min_costs):
            if not math.isinf(min_cost):
                initial_potential[first_order + order_idx] = min_cost
                reachable_order_costs.append(min_cost)
        initial_potential[sink] = min(reachable_order_costs) if reachable_order_costs else 0.0
        if reachable_order_costs:
            _, min_cost = network.solve(source, sink, initial_potential)
            total_objective += -min_cost
        for node, edge_idx, tag in tracked_edges:
            edge = network.graph[node][edge_idx]
            flow = edge.initial_cap - edge.cap
            if flow > 1e-8:
                record = dict(tag)
                record["allocated_quantity_kg"] = flow
                record["exact_grade"] = int(record["supply_grade"] == record["requested_grade"])
                allocations.append(record)

    elapsed = time.perf_counter() - started
    columns = [
        "allocation_date",
        "condition",
        "order_line_id",
        "message_id",
        "supply_id",
        "customer_id",
        "product_id",
        "supply_grade",
        "requested_grade",
        "delivery_date",
        "objective_unit_score",
        "allocated_quantity_kg",
        "exact_grade",
    ]
    return pd.DataFrame(allocations, columns=columns), elapsed, total_objective, "optimal"


def solve_manual_day(
    date: str,
    supply_day: pd.DataFrame,
    orders_day: pd.DataFrame,
    compatibility_map: dict[tuple[str, str], tuple[int, float]],
) -> tuple[pd.DataFrame, float, str]:
    started = time.perf_counter()
    available = {row.supply_id: float(row.usable_quantity_kg) for row in supply_day.itertuples(index=False)}
    supply_lookup = supply_day.set_index("supply_id")
    allocations: list[dict] = []
    sorted_orders = orders_day.sort_values(["message_id", "order_line_id"])

    for order in sorted_orders.itertuples(index=False):
        remaining = float(order.requested_quantity_kg)
        allow_inconsistent_substitution = bool(order.substitution_allowed) and int(str(order.order_line_id)[1:]) % 3 == 0
        candidates = supply_day.loc[supply_day["product_id"] == order.product_id].copy()
        candidates["grade_rank"] = (candidates["supply_grade"] != order.requested_grade).astype(int)
        candidates = candidates.sort_values(["grade_rank", "age_days", "supply_id"], ascending=[True, True, True])
        for lot in candidates.itertuples(index=False):
            if remaining <= 1e-9:
                break
            is_exact = lot.supply_grade == order.requested_grade
            is_compatible, _ = compatible(
                str(lot.supply_grade), str(order.requested_grade), allow_inconsistent_substitution, compatibility_map
            )
            if not is_compatible or (not is_exact and not allow_inconsistent_substitution):
                continue
            qty = min(remaining, available[lot.supply_id])
            qty = math.floor(qty)
            if qty <= 0:
                continue
            available[lot.supply_id] -= qty
            remaining -= qty
            allocations.append(
                {
                    "allocation_date": date,
                    "condition": "B0_manual",
                    "order_line_id": str(order.order_line_id),
                    "message_id": str(order.message_id),
                    "supply_id": str(lot.supply_id),
                    "customer_id": str(order.customer_id),
                    "product_id": str(order.product_id),
                    "supply_grade": str(lot.supply_grade),
                    "requested_grade": str(order.requested_grade),
                    "delivery_date": str(order.delivery_date),
                    "objective_unit_score": np.nan,
                    "allocated_quantity_kg": float(qty),
                    "exact_grade": int(is_exact),
                }
            )
    elapsed = time.perf_counter() - started
    columns = [
        "allocation_date",
        "condition",
        "order_line_id",
        "message_id",
        "supply_id",
        "customer_id",
        "product_id",
        "supply_grade",
        "requested_grade",
        "delivery_date",
        "objective_unit_score",
        "allocated_quantity_kg",
        "exact_grade",
    ]
    return pd.DataFrame(allocations, columns=columns), elapsed, "heuristic_feasible"


def compute_daily_metrics(
    date: str,
    condition: str,
    supply_day: pd.DataFrame,
    truth_day: pd.DataFrame,
    effective_orders_day: pd.DataFrame,
    allocations: pd.DataFrame,
    customers: pd.DataFrame,
    compatibility_map: dict[tuple[str, str], tuple[int, float]],
    solver_seconds: float,
    decision_cycle_minutes: float,
    latency_detail: dict[str, float],
) -> dict:
    truth_lookup = truth_day.set_index("order_line_id")
    priority_map = customers.set_index("customer_id")["priority_weight"].to_dict()
    fulfilled_by_line = {line_id: 0.0 for line_id in truth_lookup.index}
    exact_by_line = {line_id: 0.0 for line_id in truth_lookup.index}

    for allocation in allocations.itertuples(index=False):
        if allocation.order_line_id not in truth_lookup.index:
            continue
        truth = truth_lookup.loc[allocation.order_line_id]
        identifiers_match = (
            str(allocation.customer_id) == str(truth["customer_id"])
            and str(allocation.product_id) == str(truth["product_id"])
            and str(allocation.delivery_date) == str(truth["delivery_date"])
        )
        is_compatible, _ = compatible(
            str(allocation.supply_grade),
            str(truth["requested_grade"]),
            bool(truth["substitution_allowed"]),
            compatibility_map,
        )
        if identifiers_match and is_compatible:
            fulfilled_by_line[allocation.order_line_id] += float(allocation.allocated_quantity_kg)
            if str(allocation.supply_grade) == str(truth["requested_grade"]):
                exact_by_line[allocation.order_line_id] += float(allocation.allocated_quantity_kg)

    requested_total = float(truth_day["requested_quantity_kg"].sum())
    fulfilled_total = 0.0
    exact_total = 0.0
    complete_count = 0
    weighted_service_value = 0.0
    for line_id, truth in truth_lookup.iterrows():
        requested = float(truth["requested_quantity_kg"])
        fulfilled = min(requested, fulfilled_by_line.get(line_id, 0.0))
        exact = min(fulfilled, exact_by_line.get(line_id, 0.0))
        fulfilled_total += fulfilled
        exact_total += exact
        complete_count += int(fulfilled >= requested - 1e-6)
        weighted_service_value += fulfilled * float(priority_map.get(str(truth["customer_id"]), 1.0))

    allocated_by_supply = allocations.groupby("supply_id")["allocated_quantity_kg"].sum().to_dict() if not allocations.empty else {}
    supply_total = float(supply_day["usable_quantity_kg"].sum())
    expiring_residual = 0.0
    supply_violations = 0
    for lot in supply_day.itertuples(index=False):
        allocated = float(allocated_by_supply.get(lot.supply_id, 0.0))
        if allocated > float(lot.usable_quantity_kg) + 1e-6:
            supply_violations += 1
        residual = max(0.0, float(lot.usable_quantity_kg) - allocated)
        if int(lot.expires_next_day):
            expiring_residual += residual

    order_caps = effective_orders_day.set_index("order_line_id")["requested_quantity_kg"].to_dict()
    allocated_by_order = allocations.groupby("order_line_id")["allocated_quantity_kg"].sum().to_dict() if not allocations.empty else {}
    demand_violations = sum(
        1 for line_id, qty in allocated_by_order.items() if line_id in order_caps and qty > float(order_caps[line_id]) + 1e-6
    )
    physical_allocated = float(allocations["allocated_quantity_kg"].sum()) if not allocations.empty else 0.0
    line_count = int(len(truth_day))

    result = {
        "allocation_date": date,
        "condition": condition,
        "usable_supply_kg": supply_total,
        "near_expiry_supply_kg": float(supply_day.loc[supply_day["expires_next_day"] == 1, "usable_quantity_kg"].sum()),
        "requested_quantity_kg": requested_total,
        "allocated_physical_kg": physical_allocated,
        "true_fulfilled_kg": fulfilled_total,
        "quantity_fulfillment_rate": fulfilled_total / requested_total if requested_total else np.nan,
        "complete_order_fulfillment_rate": complete_count / line_count if line_count else np.nan,
        "exact_grade_fulfillment_rate": exact_total / requested_total if requested_total else np.nan,
        "substituted_fulfillment_rate": max(0.0, fulfilled_total - exact_total) / requested_total if requested_total else np.nan,
        "expiring_residual_kg": expiring_residual,
        "waste_proxy_rate": expiring_residual / supply_total if supply_total else np.nan,
        "weighted_service_value": weighted_service_value,
        "solver_seconds": solver_seconds,
        "decision_cycle_minutes": decision_cycle_minutes,
        "order_line_count": line_count,
        "constraint_violations": int(supply_violations + demand_violations),
    }
    result.update(latency_detail)
    return result


def evaluate_extraction(truth: pd.DataFrame, predictions: pd.DataFrame, messages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_messages = messages.loc[messages["split"] == "test", ["message_id"]]
    merged = truth.merge(test_messages, on="message_id", how="inner").merge(
        predictions, on=["order_line_id", "message_id"], how="left", validate="one_to_one"
    )
    fields = [
        ("Customer", "customer_id", "pred_customer_id", "text"),
        ("Product", "product_id", "pred_product_id", "text"),
        ("Quantity_kg", "requested_quantity_kg", "pred_requested_quantity_kg", "number"),
        ("Grade", "requested_grade", "pred_requested_grade", "text"),
        ("Delivery_date", "delivery_date", "pred_delivery_date", "text"),
        ("Substitution", "substitution_allowed", "pred_substitution_allowed", "bool"),
    ]
    rows = []
    correctness = []
    for label, truth_col, pred_col, kind in fields:
        if kind == "number":
            equal = np.isclose(
                pd.to_numeric(merged[truth_col], errors="coerce"),
                pd.to_numeric(merged[pred_col], errors="coerce"),
                rtol=0.0,
                atol=1e-6,
                equal_nan=False,
            )
        elif kind == "bool":
            equal = merged[pred_col].notna() & merged[truth_col].astype(bool).eq(merged[pred_col].astype("boolean").fillna(False).astype(bool))
        else:
            equal = merged[pred_col].notna() & merged[truth_col].astype(str).eq(merged[pred_col].astype(str))
        predicted_present = merged[pred_col].notna().to_numpy()
        equal = np.asarray(equal, dtype=bool)
        tp = int(equal.sum())
        fp = int((predicted_present & ~equal).sum())
        fn = int((~equal).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "field": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_match_rate": float(equal.mean()),
            }
        )
        correctness.append(equal)

    record_exact = np.logical_and.reduce(correctness) & merged["json_valid"].fillna(False).to_numpy(dtype=bool)
    test_prediction_messages = predictions.merge(test_messages, on="message_id", how="inner").drop_duplicates("message_id")
    summary = pd.DataFrame(
        [
            ("test_messages", float(len(test_prediction_messages))),
            ("test_order_lines", float(len(merged))),
            ("macro_field_f1", float(pd.DataFrame(rows)["f1"].mean())),
            ("record_exact_match_rate", float(record_exact.mean())),
            ("invalid_json_rate", float((~test_prediction_messages["json_valid"].astype(bool)).mean())),
            ("human_confirmation_rate", float(test_prediction_messages["needs_human_confirmation"].astype(bool).mean())),
        ],
        columns=["metric", "value"],
    )
    return pd.DataFrame(rows), summary


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, degrees_freedom: int) -> float:
    x = degrees_freedom / (degrees_freedom + t_value * t_value)
    ib = regularized_beta(degrees_freedom / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t_value >= 0 else 0.5 * ib


def normal_cdf(z_value: float) -> float:
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


def paired_analysis(
    hypothesis: str,
    metric: str,
    baseline: np.ndarray,
    proposed: np.ndarray,
    improvement_direction: str,
    rng: np.random.Generator,
) -> dict:
    diff = baseline - proposed if improvement_direction == "lower" else proposed - baseline
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    centered = diff - mean_diff
    skewness = float(np.mean(centered**3) / (np.mean(centered**2) ** 1.5)) if np.mean(centered**2) > 0 else 0.0
    bootstrap = rng.choice(diff, size=(5000, n), replace=True).mean(axis=1)
    ci_low, ci_high = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]

    if abs(skewness) <= 1.0 and sd_diff > 0:
        statistic = mean_diff / (sd_diff / math.sqrt(n))
        p_value = 1.0 - student_t_cdf(statistic, n - 1)
        test_used = "paired_t_one_sided"
        effect_size = mean_diff / sd_diff
        effect_label = "Cohen_dz"
    else:
        nonzero = diff[np.abs(diff) > 1e-12]
        ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
        w_plus = float(ranks[nonzero > 0].sum())
        mean_w = float(ranks.sum() / 2.0)
        sd_w = math.sqrt(float((ranks * ranks).sum() / 4.0)) if len(ranks) else np.nan
        statistic = (w_plus - mean_w - 0.5) / sd_w if sd_w and not math.isnan(sd_w) else 0.0
        p_value = 1.0 - normal_cdf(statistic)
        total_rank = float(ranks.sum()) if len(ranks) else 1.0
        effect_size = (2.0 * w_plus - total_rank) / total_rank
        effect_label = "rank_biserial"
        test_used = "wilcoxon_signed_rank_one_sided_normal_approx"

    baseline_mean = float(np.mean(baseline))
    proposed_mean = float(np.mean(proposed))
    relative = mean_diff / baseline_mean if baseline_mean else np.nan
    return {
        "hypothesis": hypothesis,
        "metric": metric,
        "improvement_direction": improvement_direction,
        "n_days": n,
        "baseline_mean": baseline_mean,
        "proposed_mean": proposed_mean,
        "mean_paired_improvement": mean_diff,
        "relative_improvement": relative,
        "difference_skewness": skewness,
        "test_used": test_used,
        "test_statistic": float(statistic),
        "p_value_one_sided": float(max(1e-16, min(1.0, p_value))),
        "effect_size_type": effect_label,
        "effect_size": float(effect_size),
        "bootstrap_ci95_lower": ci_low,
        "bootstrap_ci95_upper": ci_high,
        "supported_at_alpha_0_05": bool(mean_diff > 0 and p_value < 0.05),
    }


def create_human_evaluation(
    rng: np.random.Generator,
    truth_test: pd.DataFrame,
    p1_allocations: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommended = p1_allocations.groupby("order_line_id")["allocated_quantity_kg"].sum().rename("recommended_quantity_kg")
    substitution = (
        p1_allocations.assign(is_sub=lambda x: x["supply_grade"] != x["requested_grade"])
        .groupby("order_line_id")["is_sub"]
        .max()
        .rename("used_substitution")
    )
    pool = truth_test.merge(recommended, on="order_line_id", how="left").merge(substitution, on="order_line_id", how="left")
    pool["recommended_quantity_kg"] = pool["recommended_quantity_kg"].fillna(0.0)
    pool["used_substitution"] = pool["used_substitution"].fillna(False)
    pred_flags = predictions[["order_line_id", "injected_error_fields", "needs_human_confirmation"]].copy()
    pred_flags["unresolved_semantic_error"] = pred_flags["injected_error_fields"].fillna("").ne("") & ~pred_flags[
        "needs_human_confirmation"
    ].astype(bool)
    pool = pool.merge(pred_flags[["order_line_id", "unresolved_semantic_error"]], on="order_line_id", how="left")
    pool["unresolved_semantic_error"] = pool["unresolved_semantic_error"].fillna(False)
    pool["shortfall"] = pool["recommended_quantity_kg"] < pool["requested_quantity_kg"] - 1e-6

    sample_size = min(120, len(pool))
    reviewed = pool.sample(sample_size, random_state=SEED).reset_index(drop=True)
    users = ["U01", "U02", "U03", "U04", "U05"]
    ratings: list[dict] = []
    rating_id = 1
    for user in users:
        user_sample = reviewed.sample(min(36, len(reviewed)), random_state=SEED + int(user[-2:]))
        for row in user_sample.itertuples(index=False):
            unresolved = int(row.unresolved_semantic_error)
            shortfall = int(row.shortfall)
            values = {
                "clarity": rng.normal(4.35 - 0.18 * shortfall, 0.52),
                "factual_correctness": rng.normal(4.55 - 0.95 * unresolved, 0.42),
                "usefulness": rng.normal(4.28 - 0.18 * shortfall, 0.55),
                "review_confidence": rng.normal(4.18 - 0.40 * unresolved, 0.58),
            }
            rounded = {key: int(np.clip(round(value), 1, 5)) for key, value in values.items()}
            overall = int(np.clip(round(np.mean(list(rounded.values())) + rng.normal(0.0, 0.30)), 1, 5))
            ratings.append(
                {
                    "rating_id": f"R{rating_id:05d}",
                    "user_id": user,
                    "allocation_date": row.allocation_date,
                    "order_line_id": row.order_line_id,
                    **rounded,
                    "overall_usability": overall,
                    "comment_code": "CHECK_SOURCE_DATA" if unresolved else ("SHORTFALL_CLEAR" if shortfall else "CLEAR_AND_USEFUL"),
                }
            )
            rating_id += 1

    overrides: list[dict] = []
    override_id = 1
    for row in reviewed.itertuples(index=False):
        probability = 0.035 + 0.080 * int(row.shortfall) + 0.060 * int(row.used_substitution) + 0.220 * int(row.unresolved_semantic_error)
        if rng.random() >= probability:
            continue
        if row.unresolved_semantic_error:
            reason = "PARSER_CORRECTION"
        elif row.used_substitution:
            reason = "QUALITY_PREFERENCE"
        elif row.shortfall:
            reason = str(rng.choice(["RELATIONSHIP_CONTEXT", "CUSTOMER_CHANGE"]))
        else:
            reason = "PACKHOUSE_CONSTRAINT"
        factor = float(rng.choice([0.75, 0.85, 1.10]))
        overrides.append(
            {
                "override_id": f"V{override_id:04d}",
                "allocation_date": row.allocation_date,
                "order_line_id": row.order_line_id,
                "recommended_quantity_kg": float(row.recommended_quantity_kg),
                "final_quantity_kg": float(max(0.0, round(row.recommended_quantity_kg * factor, 1))),
                "reason_code": reason,
                "user_id": str(rng.choice(users)),
            }
        )
        override_id += 1

    ratings_df = pd.DataFrame(ratings)
    overrides_df = pd.DataFrame(
        overrides,
        columns=[
            "override_id",
            "allocation_date",
            "order_line_id",
            "recommended_quantity_kg",
            "final_quantity_kg",
            "reason_code",
            "user_id",
        ],
    )
    summary_rows = []
    for metric in ["clarity", "factual_correctness", "usefulness", "review_confidence", "overall_usability"]:
        summary_rows.append(
            {
                "metric": metric,
                "n_ratings": int(len(ratings_df)),
                "mean": float(ratings_df[metric].mean()),
                "median": float(ratings_df[metric].median()),
                "standard_deviation": float(ratings_df[metric].std(ddof=1)),
            }
        )
    summary_rows.append(
        {
            "metric": "override_rate",
            "n_ratings": int(len(reviewed)),
            "mean": float(len(overrides_df) / len(reviewed)) if len(reviewed) else 0.0,
            "median": np.nan,
            "standard_deviation": np.nan,
        }
    )
    return ratings_df, overrides_df, pd.DataFrame(summary_rows)


def run_robustness_checks(
    supply: pd.DataFrame,
    truth: pd.DataFrame,
    allocations: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    supply_caps = supply.set_index("supply_id")["usable_quantity_kg"]
    allocated = allocations.groupby(["condition", "supply_id"])["allocated_quantity_kg"].sum().reset_index()
    max_excess = float(
        max(
            [0.0]
            + [row.allocated_quantity_kg - float(supply_caps.get(row.supply_id, 0.0)) for row in allocated.itertuples(index=False)]
        )
    )
    checks.append(("No supply over-allocation", max_excess <= 1e-6, f"maximum excess={max_excess:.6f} kg"))

    order_caps = truth.set_index("order_line_id")["requested_quantity_kg"]
    b1 = allocations.loc[allocations["condition"] == "B1_optimization_only"]
    b1_by_order = b1.groupby("order_line_id")["allocated_quantity_kg"].sum()
    max_order_excess = float(max([0.0] + [qty - float(order_caps.loc[line]) for line, qty in b1_by_order.items()]))
    checks.append(("No demand over-allocation", max_order_excess <= 1e-6, f"maximum excess={max_order_excess:.6f} kg"))

    invalid_confirmed = predictions.loc[~predictions["json_valid"].astype(bool), "needs_human_confirmation"].astype(bool).all()
    checks.append(("Invalid JSON blocked by human gate", bool(invalid_confirmed), "all invalid JSON messages require confirmation"))
    missing_unit = predictions["pred_unit"].isna()
    missing_unit_confirmed = predictions.loc[missing_unit, "needs_human_confirmation"].astype(bool).all() if missing_unit.any() else True
    checks.append(("Missing units blocked by human gate", bool(missing_unit_confirmed), f"missing-unit rows={int(missing_unit.sum())}"))
    expired_allocations = allocations.merge(supply[["supply_id", "age_days", "shelf_life_days"]], on="supply_id", how="left")
    no_expired = bool((expired_allocations["age_days"] < expired_allocations["shelf_life_days"]).all())
    checks.append(("No expired cohort allocated", no_expired, "all allocated cohorts satisfy age < shelf life"))
    p1_nonnegative = bool((allocations["allocated_quantity_kg"] >= -1e-9).all())
    checks.append(("Non-negative allocation variables", p1_nonnegative, "all x >= 0"))
    return pd.DataFrame(checks, columns=["check", "passed", "observed"])


def write_metric_svg(path: Path, comparison: pd.DataFrame) -> None:
    manual = comparison.loc[comparison["condition"] == "B0_manual"].iloc[0]
    proposed = comparison.loc[comparison["condition"] == "P1_proposed_dss"].iloc[0]
    metrics = [
        ("Expiring residual (%)", manual["waste_proxy_rate"] * 100, proposed["waste_proxy_rate"] * 100),
        ("Quantity fulfillment (%)", manual["quantity_fulfillment_rate"] * 100, proposed["quantity_fulfillment_rate"] * 100),
        ("Decision time (min)", manual["decision_cycle_minutes"], proposed["decision_cycle_minutes"]),
    ]
    width, height = 960, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="40" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#0F172A">Synthetic held-out comparison: manual vs proposed DSS</text>',
        '<text x="40" y="66" font-family="Arial" font-size="12" fill="#475569">Simulation only; not real-farm or real-LLM evidence</text>',
    ]
    panel_w = 280
    for idx, (label, manual_value, proposed_value) in enumerate(metrics):
        x0 = 40 + idx * 305
        y0 = 105
        max_value = max(manual_value, proposed_value, 1e-9) * 1.18
        bar_scale = 180 / max_value
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="250" rx="8" fill="#FFFFFF" stroke="#CBD5E1"/>')
        parts.append(f'<text x="{x0 + 14}" y="{y0 + 28}" font-family="Arial" font-size="14" font-weight="700" fill="#1E293B">{label}</text>')
        for bar_idx, (name, value, color) in enumerate([("Manual", manual_value, "#94A3B8"), ("Proposed DSS", proposed_value, "#0F766E")]):
            bx = x0 + 35 + bar_idx * 110
            bar_h = value * bar_scale
            by = y0 + 210 - bar_h
            parts.append(f'<rect x="{bx}" y="{by:.1f}" width="70" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{bx + 35}" y="{by - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#0F172A">{value:.1f}</text>')
            parts.append(f'<text x="{bx + 35}" y="{y0 + 232}" text-anchor="middle" font-family="Arial" font-size="11" fill="#475569">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic dissertation experiments E1-E4.")
    parser.add_argument("--dataset-dir", default="../dataset")
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--figures-dir", default="../figures")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(dataset_dir)
    rng = np.random.default_rng(args.seed + 11)

    products = data["products"]
    customers = data["customers"]
    compatibility_map = {
        (str(row.supply_grade), str(row.requested_grade)): (int(row.compatible), float(row.substitution_penalty_per_kg))
        for row in data["grade_compatibility"].itertuples(index=False)
    }
    messages = data["order_messages"]
    truth = data["order_lines_ground_truth"]
    predictions = data["parser_predictions"]
    supply = data["supply"]
    test_dates = sorted(messages.loc[messages["split"] == "test", "allocation_date"].unique().tolist())
    truth_test = truth.loc[truth["allocation_date"].isin(test_dates)].copy()
    supply_test = supply.loc[supply["allocation_date"].isin(test_dates)].copy()
    p1_orders = build_p1_orders(truth, predictions)

    extraction_metrics, extraction_summary = evaluate_extraction(truth, predictions, messages)
    extraction_metrics.to_csv(results_dir / "extraction_metrics.csv", index=False, encoding="utf-8-sig")
    extraction_summary.to_csv(results_dir / "extraction_summary.csv", index=False, encoding="utf-8-sig")

    all_allocations: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    for date_idx, date in enumerate(test_dates):
        supply_day = supply_test.loc[supply_test["allocation_date"] == date].copy()
        truth_day = truth_test.loc[truth_test["allocation_date"] == date].copy()
        p1_day = p1_orders.loc[p1_orders["allocation_date"] == date].copy()
        message_day = messages.loc[messages["allocation_date"] == date]
        pred_day = predictions.loc[predictions["message_id"].isin(message_day["message_id"])]
        day_rng = np.random.default_rng(args.seed + date_idx)

        manual_alloc, manual_runtime, _ = solve_manual_day(date, supply_day, truth_day, compatibility_map)
        manual_minutes = float(max(10.0, 12.0 + 0.55 * len(truth_day) + 0.85 * (message_day["message_difficulty"] == "hard").sum() + day_rng.normal(0.0, 2.2)))
        daily_rows.append(
            compute_daily_metrics(
                date,
                "B0_manual",
                supply_day,
                truth_day,
                truth_day,
                manual_alloc,
                customers,
                compatibility_map,
                manual_runtime,
                manual_minutes,
                {"parser_latency_seconds": 0.0, "validation_seconds": 0.0, "explanation_seconds": 0.0, "human_review_minutes": manual_minutes},
            )
        )
        all_allocations.append(manual_alloc)

        b1_alloc, b1_runtime, _, _ = solve_optimized_day(
            date, supply_day, truth_day.assign(solver_eligible=True), customers, compatibility_map, DEFAULT_WEIGHTS, "B1_optimization_only"
        )
        b1_minutes = float(2.2 + 0.025 * len(truth_day) + b1_runtime / 60.0)
        daily_rows.append(
            compute_daily_metrics(
                date,
                "B1_optimization_only",
                supply_day,
                truth_day,
                truth_day,
                b1_alloc,
                customers,
                compatibility_map,
                b1_runtime,
                b1_minutes,
                {"parser_latency_seconds": 0.0, "validation_seconds": 0.0, "explanation_seconds": 0.0, "human_review_minutes": b1_minutes},
            )
        )
        all_allocations.append(b1_alloc)

        p1_alloc, p1_runtime, _, _ = solve_optimized_day(
            date, supply_day, p1_day, customers, compatibility_map, DEFAULT_WEIGHTS, "P1_proposed_dss"
        )
        unique_messages = max(1, int(message_day["message_id"].nunique()))
        confirmations = int(
            pred_day.loc[pred_day["needs_human_confirmation"].astype(bool), "message_id"].nunique()
        )
        parser_latency = float(unique_messages * day_rng.uniform(0.55, 0.90))
        validation_seconds = float(len(p1_day) * day_rng.uniform(0.012, 0.025))
        explanation_seconds = float(max(1, p1_alloc["order_line_id"].nunique()) * day_rng.uniform(0.035, 0.065))
        human_review_minutes = float(2.0 + 0.24 * confirmations + 0.028 * len(p1_day) + max(0.0, day_rng.normal(0.0, 0.45)))
        p1_minutes = human_review_minutes + (parser_latency + validation_seconds + explanation_seconds + p1_runtime) / 60.0
        daily_rows.append(
            compute_daily_metrics(
                date,
                "P1_proposed_dss",
                supply_day,
                truth_day,
                p1_day,
                p1_alloc,
                customers,
                compatibility_map,
                p1_runtime,
                p1_minutes,
                {
                    "parser_latency_seconds": parser_latency,
                    "validation_seconds": validation_seconds,
                    "explanation_seconds": explanation_seconds,
                    "human_review_minutes": human_review_minutes,
                },
            )
        )
        all_allocations.append(p1_alloc)

    allocations = pd.concat(all_allocations, ignore_index=True)
    daily_metrics = pd.DataFrame(daily_rows)
    allocations.to_csv(results_dir / "allocations.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(results_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")

    comparison = (
        daily_metrics.groupby("condition", as_index=False)[
            [
                "waste_proxy_rate",
                "quantity_fulfillment_rate",
                "complete_order_fulfillment_rate",
                "exact_grade_fulfillment_rate",
                "substituted_fulfillment_rate",
                "decision_cycle_minutes",
                "solver_seconds",
                "weighted_service_value",
            ]
        ]
        .mean()
    )
    comparison.to_csv(results_dir / "condition_comparison.csv", index=False, encoding="utf-8-sig")

    pivot = daily_metrics.pivot(index="allocation_date", columns="condition")
    tests = [
        paired_analysis(
            "H1",
            "waste_proxy_rate",
            pivot["waste_proxy_rate"]["B0_manual"].to_numpy(),
            pivot["waste_proxy_rate"]["P1_proposed_dss"].to_numpy(),
            "lower",
            rng,
        ),
        paired_analysis(
            "H2",
            "quantity_fulfillment_rate",
            pivot["quantity_fulfillment_rate"]["B0_manual"].to_numpy(),
            pivot["quantity_fulfillment_rate"]["P1_proposed_dss"].to_numpy(),
            "higher",
            rng,
        ),
        paired_analysis(
            "H3",
            "decision_cycle_minutes",
            pivot["decision_cycle_minutes"]["B0_manual"].to_numpy(),
            pivot["decision_cycle_minutes"]["P1_proposed_dss"].to_numpy(),
            "lower",
            rng,
        ),
    ]
    hypothesis_tests = pd.DataFrame(tests)
    hypothesis_tests.to_csv(results_dir / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")

    sensitivity_rows = []
    sensitivity_grid = [(lp, ls, lw) for lp in [2.0, 4.0, 6.0] for ls in [1.0, 2.0, 4.0] for lw in [3.0, 6.0, 9.0]]
    for lambda_priority, lambda_substitution, lambda_waste in sensitivity_grid:
        scenario_metrics = []
        scenario_weights = {
            "lambda_priority": lambda_priority,
            "lambda_substitution": lambda_substitution,
            "lambda_waste": lambda_waste,
        }
        for date in test_dates:
            supply_day = supply_test.loc[supply_test["allocation_date"] == date]
            truth_day = truth_test.loc[truth_test["allocation_date"] == date]
            alloc, runtime, _, _ = solve_optimized_day(
                date,
                supply_day,
                truth_day.assign(solver_eligible=True),
                customers,
                compatibility_map,
                scenario_weights,
                "sensitivity",
            )
            scenario_metrics.append(
                compute_daily_metrics(
                    date,
                    "sensitivity",
                    supply_day,
                    truth_day,
                    truth_day,
                    alloc,
                    customers,
                    compatibility_map,
                    runtime,
                    0.0,
                    {"parser_latency_seconds": 0.0, "validation_seconds": 0.0, "explanation_seconds": 0.0, "human_review_minutes": 0.0},
                )
            )
        scenario = pd.DataFrame(scenario_metrics)
        sensitivity_rows.append(
            {
                **scenario_weights,
                "mean_waste_proxy_rate": float(scenario["waste_proxy_rate"].mean()),
                "mean_quantity_fulfillment_rate": float(scenario["quantity_fulfillment_rate"].mean()),
                "mean_complete_order_fulfillment_rate": float(scenario["complete_order_fulfillment_rate"].mean()),
                "mean_weighted_service_value": float(scenario["weighted_service_value"].mean()),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(results_dir / "sensitivity_results.csv", index=False, encoding="utf-8-sig")

    base_sensitivity = sensitivity.loc[
        (sensitivity["lambda_priority"] == DEFAULT_WEIGHTS["lambda_priority"])
        & (sensitivity["lambda_substitution"] == DEFAULT_WEIGHTS["lambda_substitution"])
        & (sensitivity["lambda_waste"] == DEFAULT_WEIGHTS["lambda_waste"])
    ].iloc[0]
    ablation = pd.DataFrame(
        [
            ("Base", DEFAULT_WEIGHTS["lambda_priority"], DEFAULT_WEIGHTS["lambda_substitution"], DEFAULT_WEIGHTS["lambda_waste"]),
            ("No priority weight", 0.0, DEFAULT_WEIGHTS["lambda_substitution"], DEFAULT_WEIGHTS["lambda_waste"]),
            ("No substitution penalty", DEFAULT_WEIGHTS["lambda_priority"], 0.0, DEFAULT_WEIGHTS["lambda_waste"]),
            ("No waste penalty", DEFAULT_WEIGHTS["lambda_priority"], DEFAULT_WEIGHTS["lambda_substitution"], 0.0),
        ],
        columns=["scenario", "lambda_priority", "lambda_substitution", "lambda_waste"],
    )
    ablation_rows = []
    for scenario in ablation.itertuples(index=False):
        metrics = []
        scenario_weights = {
            "lambda_priority": scenario.lambda_priority,
            "lambda_substitution": scenario.lambda_substitution,
            "lambda_waste": scenario.lambda_waste,
        }
        for date in test_dates:
            supply_day = supply_test.loc[supply_test["allocation_date"] == date]
            truth_day = truth_test.loc[truth_test["allocation_date"] == date]
            alloc, runtime, _, _ = solve_optimized_day(
                date, supply_day, truth_day.assign(solver_eligible=True), customers, compatibility_map, scenario_weights, "ablation"
            )
            metrics.append(
                compute_daily_metrics(
                    date,
                    "ablation",
                    supply_day,
                    truth_day,
                    truth_day,
                    alloc,
                    customers,
                    compatibility_map,
                    runtime,
                    0.0,
                    {"parser_latency_seconds": 0.0, "validation_seconds": 0.0, "explanation_seconds": 0.0, "human_review_minutes": 0.0},
                )
            )
        frame = pd.DataFrame(metrics)
        ablation_rows.append(
            {
                **scenario._asdict(),
                "mean_waste_proxy_rate": float(frame["waste_proxy_rate"].mean()),
                "mean_quantity_fulfillment_rate": float(frame["quantity_fulfillment_rate"].mean()),
                "mean_complete_order_fulfillment_rate": float(frame["complete_order_fulfillment_rate"].mean()),
                "mean_weighted_service_value": float(frame["weighted_service_value"].mean()),
            }
        )
    ablation_results = pd.DataFrame(ablation_rows)
    ablation_results.to_csv(results_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")

    ratings, overrides, human_summary = create_human_evaluation(rng, truth_test, allocations.loc[allocations["condition"] == "P1_proposed_dss"], predictions)
    ratings.to_csv(results_dir / "human_evaluation.csv", index=False, encoding="utf-8-sig")
    overrides.to_csv(results_dir / "override_logs.csv", index=False, encoding="utf-8-sig")
    human_summary.to_csv(results_dir / "human_evaluation_summary.csv", index=False, encoding="utf-8-sig")

    robustness = run_robustness_checks(supply_test, truth_test, allocations, predictions)
    robustness.to_csv(results_dir / "robustness_results.csv", index=False, encoding="utf-8-sig")
    write_metric_svg(figures_dir / "manual_vs_dss_metrics.svg", comparison)

    summary = {
        "synthetic_data_notice": "Simulation only. No real farm observations and no real LLM endpoint calls were used.",
        "test_days": len(test_dates),
        "test_order_lines": int(len(truth_test)),
        "default_objective_weights": DEFAULT_WEIGHTS,
        "condition_means": comparison.set_index("condition").round(6).to_dict("index"),
        "extraction_summary": extraction_summary.set_index("metric")["value"].to_dict(),
        "hypotheses_supported": hypothesis_tests.set_index("hypothesis")["supported_at_alpha_0_05"].to_dict(),
        "all_robustness_checks_passed": bool(robustness["passed"].all()),
        "sensitivity_scenarios": int(len(sensitivity)),
        "ablation_scenarios": int(len(ablation_results)),
        "human_ratings": int(len(ratings)),
        "override_records": int(len(overrides)),
        "solver": "custom min-cost-flow LP-equivalent network solver; continuous transportation structure with integral capacities",
    }
    (results_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    database_path = dataset_dir / "synthetic_agri_dss.sqlite"
    with sqlite3.connect(database_path) as con:
        for name, frame in {
            "p1_effective_orders": p1_orders,
            "allocations": allocations,
            "daily_metrics": daily_metrics,
            "condition_comparison": comparison,
            "extraction_metrics": extraction_metrics,
            "extraction_summary": extraction_summary,
            "hypothesis_tests": hypothesis_tests,
            "sensitivity_results": sensitivity,
            "ablation_results": ablation_results,
            "human_evaluation": ratings,
            "override_logs": overrides,
            "human_evaluation_summary": human_summary,
            "robustness_results": robustness,
        }.items():
            frame.to_sql(name, con, if_exists="replace", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
