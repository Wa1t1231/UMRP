-- Synthetic agricultural DSS experiment schema.
-- All customer and operational records are generated; no real personal data are included.

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name_en TEXT NOT NULL,
    product_name_zh TEXT NOT NULL,
    aliases TEXT NOT NULL,
    shelf_life_days INTEGER NOT NULL CHECK (shelf_life_days > 0),
    unit_price_per_kg REAL NOT NULL CHECK (unit_price_per_kg >= 0),
    kg_per_box REAL NOT NULL CHECK (kg_per_box > 0),
    kg_per_crate REAL NOT NULL CHECK (kg_per_crate > 0),
    base_daily_supply_kg REAL NOT NULL CHECK (base_daily_supply_kg >= 0)
);

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    customer_segment TEXT NOT NULL,
    priority_weight REAL NOT NULL CHECK (priority_weight > 0),
    contract_class TEXT NOT NULL,
    default_substitution_allowed INTEGER NOT NULL CHECK (default_substitution_allowed IN (0, 1))
);

CREATE TABLE grade_compatibility (
    supply_grade TEXT NOT NULL,
    requested_grade TEXT NOT NULL,
    compatible INTEGER NOT NULL CHECK (compatible IN (0, 1)),
    substitution_penalty_per_kg REAL NOT NULL CHECK (substitution_penalty_per_kg >= 0),
    PRIMARY KEY (supply_grade, requested_grade)
);

CREATE TABLE supply (
    supply_id TEXT PRIMARY KEY,
    allocation_date TEXT NOT NULL,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    supply_grade TEXT NOT NULL,
    age_days INTEGER NOT NULL CHECK (age_days >= 0),
    shelf_life_days INTEGER NOT NULL CHECK (shelf_life_days > 0),
    usable_quantity_kg REAL NOT NULL CHECK (usable_quantity_kg >= 0),
    expires_next_day INTEGER NOT NULL CHECK (expires_next_day IN (0, 1)),
    unit_value_per_kg REAL NOT NULL CHECK (unit_value_per_kg >= 0)
);

CREATE TABLE order_messages (
    message_id TEXT PRIMARY KEY,
    allocation_date TEXT NOT NULL,
    authorized_customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    raw_message TEXT NOT NULL,
    message_difficulty TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('development', 'test')),
    message_hash_sha256 TEXT NOT NULL
);

CREATE TABLE order_lines_ground_truth (
    order_line_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES order_messages(message_id),
    allocation_date TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    raw_product_text TEXT NOT NULL,
    requested_quantity REAL NOT NULL CHECK (requested_quantity > 0),
    unit TEXT NOT NULL,
    requested_quantity_kg REAL NOT NULL CHECK (requested_quantity_kg > 0),
    requested_grade TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    substitution_allowed INTEGER NOT NULL CHECK (substitution_allowed IN (0, 1)),
    notes TEXT
);

CREATE TABLE parser_predictions (
    order_line_id TEXT PRIMARY KEY REFERENCES order_lines_ground_truth(order_line_id),
    message_id TEXT NOT NULL REFERENCES order_messages(message_id),
    pred_customer_id TEXT,
    pred_product_id TEXT,
    pred_requested_quantity REAL,
    pred_unit TEXT,
    pred_requested_quantity_kg REAL,
    pred_requested_grade TEXT,
    pred_delivery_date TEXT,
    pred_substitution_allowed INTEGER,
    json_valid INTEGER NOT NULL CHECK (json_valid IN (0, 1)),
    detected_issue_codes TEXT,
    injected_error_fields TEXT,
    prediction_source TEXT NOT NULL,
    needs_human_confirmation INTEGER NOT NULL CHECK (needs_human_confirmation IN (0, 1))
);

CREATE INDEX idx_supply_date_product ON supply(allocation_date, product_id);
CREATE INDEX idx_orders_date_product ON order_lines_ground_truth(allocation_date, product_id);
CREATE INDEX idx_messages_split ON order_messages(split);

-- run_experiments.py adds result tables such as allocations, daily_metrics,
-- hypothesis_tests, sensitivity_results, human_evaluation, and robustness_results.
