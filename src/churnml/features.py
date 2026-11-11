"""Feature engineering. The pandas version is the reference implementation."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "age",
    "monthly_fee",
    "tenure_days",
    "sessions_last3",
    "sessions_prev3",
    "tickets_total",
    "failures_total",
    "session_trend",
]
CATEGORICAL = ["plan", "country"]
TARGET = "churned"


def build_features(
    customers: pd.DataFrame,
    usage: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """One row per customer, using only usage strictly before `as_of`."""
    cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=3)
    usage = usage.assign(month=pd.to_datetime(usage["month"]))
    usage = usage[usage["month"] < pd.Timestamp(as_of)]
    recent = usage[usage["month"] >= cutoff]
    older = usage[usage["month"] < cutoff]

    agg = pd.DataFrame(
        {
            "sessions_last3": recent.groupby("customer_id")["sessions"].mean(),
            "sessions_prev3": older.groupby("customer_id")["sessions"].mean(),
            "tickets_total": usage.groupby("customer_id")["support_tickets"].sum(),
            "failures_total": usage.groupby("customer_id")["payment_failures"].sum(),
        }
    ).reset_index()

    frame = customers.merge(agg, on="customer_id", how="inner")
    signup = pd.to_datetime(frame["signup_date"])
    frame["tenure_days"] = (pd.Timestamp(as_of) - signup).dt.days
    return add_derived(frame)


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns computed the same way for pandas-built and SQL-built features."""
    frame = frame.assign(
        session_trend=pd.col("sessions_last3") / (pd.col("sessions_prev3") + 1.0)
    )
    columns = ["customer_id", *CATEGORICAL, *NUMERIC]
    return frame[columns].sort_values("customer_id").reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )
