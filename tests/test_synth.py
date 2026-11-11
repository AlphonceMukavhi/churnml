import pandas as pd

from churnml.synth import AS_OF, generate, month_starts


def test_generation_is_deterministic():
    a = generate(200, seed=1)
    b = generate(200, seed=1)
    pd.testing.assert_frame_equal(a["usage_monthly"], b["usage_monthly"])
    assert not a["labels"].equals(generate(200, seed=2)["labels"])


def test_table_shapes_and_keys(tables):
    customers, usage = tables["customers"], tables["usage_monthly"]
    assert customers["customer_id"].is_unique
    assert len(usage) == len(customers) * 6
    assert not usage.duplicated(["customer_id", "month"]).any()


def test_value_ranges(tables):
    usage = tables["usage_monthly"]
    assert (usage["sessions"] >= 0).all()
    assert set(usage["payment_failures"]) <= {0, 1}
    churn_rate = tables["labels"]["churned"].mean()
    assert 0.05 < churn_rate < 0.5


def test_months_end_before_as_of():
    months = month_starts(AS_OF)
    assert months[-1] < AS_OF
    assert months == sorted(months)
