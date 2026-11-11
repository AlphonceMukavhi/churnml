"""Deterministic synthetic subscription data (no real customer data needed)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

PLANS = {"basic": 9.0, "standard": 19.0, "premium": 39.0}
COUNTRIES = ["ZW", "ZA", "KE", "NG", "GH"]
AS_OF = date(2026, 7, 1)
MONTHS = 6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def month_starts(as_of: date = AS_OF, months: int = MONTHS) -> list[date]:
    """First day of each of the `months` months before `as_of`, oldest first."""
    first = pd.Timestamp(as_of).replace(day=1)
    return [(first - pd.DateOffset(months=k)).date() for k in range(months, 0, -1)]


def generate(
    n_customers: int = 2000, seed: int = 42, as_of: date = AS_OF
) -> dict[str, pd.DataFrame]:
    """Return customers, usage_monthly and labels tables."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_customers + 1)
    plan = rng.choice(list(PLANS), size=n_customers, p=[0.45, 0.40, 0.15])
    age = np.clip(rng.normal(35, 10, n_customers), 18, 80).astype(int)
    tenure = rng.integers(30, 1500, n_customers)
    signup = [(pd.Timestamp(as_of) - pd.Timedelta(days=int(t))).date() for t in tenure]
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "signup_date": signup,
            "plan": plan,
            "country": rng.choice(COUNTRIES, size=n_customers),
            "age": age,
            "monthly_fee": [PLANS[p] for p in plan],
        }
    )

    z = (
        -1.4
        + 0.7 * (plan == "basic")
        - 0.5 * (plan == "premium")
        + 0.02 * (35 - age)
        - 0.0006 * tenure
        + rng.normal(0, 0.8, n_customers)
    )
    churned = rng.random(n_customers) < _sigmoid(z)

    plan_mult = np.array(
        [{"basic": 0.7, "standard": 1.0, "premium": 1.3}[p] for p in plan]
    )
    engagement = rng.lognormal(np.log(20), 0.5, n_customers) * plan_mult

    rows = []
    months = month_starts(as_of)
    for m_idx, month in enumerate(months):
        progress = m_idx / (len(months) - 1)
        factor = np.where(churned, 1.0 - 0.28 * progress, 1.0)
        sessions = rng.poisson(engagement * factor)
        tickets = rng.poisson(0.25 + 0.35 * churned)
        failures = (rng.random(n_customers) < 0.03 + 0.07 * churned).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": ids,
                    "month": month,
                    "sessions": sessions,
                    "support_tickets": tickets,
                    "payment_failures": failures,
                }
            )
        )
    usage = pd.concat(rows, ignore_index=True)
    labels = pd.DataFrame({"customer_id": ids, "churned": churned})
    return {"customers": customers, "usage_monthly": usage, "labels": labels}
