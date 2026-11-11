"""PostgreSQL access. Parameterised queries only; no string-built SQL."""

from __future__ import annotations

from datetime import date
from importlib import resources

import pandas as pd
import psycopg

from .features import add_derived

TABLE_COLUMNS = {
    "customers": [
        "customer_id",
        "signup_date",
        "plan",
        "country",
        "age",
        "monthly_fee",
    ],
    "usage_monthly": [
        "customer_id",
        "month",
        "sessions",
        "support_tickets",
        "payment_failures",
    ],
    "labels": ["customer_id", "churned"],
}


def _sql(name: str) -> str:
    return resources.files("churnml").joinpath(name).read_text(encoding="utf-8")


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(_sql("schema.sql"))
    conn.commit()


def load_tables(conn: psycopg.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """Bulk-load with COPY inside one transaction (all tables or none)."""
    with conn.cursor() as cur:
        for name in ("customers", "usage_monthly", "labels"):  # FK order
            cols = TABLE_COLUMNS[name]
            frame = tables[name][cols]
            statement = f"COPY {name} ({', '.join(cols)}) FROM STDIN"
            with cur.copy(statement) as copy:
                for row in frame.itertuples(index=False, name=None):
                    copy.write_row(tuple(_py(v) for v in row))
    conn.commit()


def _py(value: object) -> object:
    """Convert numpy scalars to plain Python values for psycopg."""
    return value.item() if hasattr(value, "item") else value


def _frame(cur: psycopg.Cursor) -> pd.DataFrame:
    columns = [d.name for d in cur.description or []]
    return pd.DataFrame(cur.fetchall(), columns=columns)


def read_features(conn: psycopg.Connection, as_of: date) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(_sql("features.sql"), {"as_of": as_of})
        return add_derived(_frame(cur))


def read_labels(conn: psycopg.Connection) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id, churned FROM labels ORDER BY customer_id")
        return _frame(cur)
