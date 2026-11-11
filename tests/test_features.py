import pandas as pd
import pytest

from churnml.features import CATEGORICAL, NUMERIC, add_derived, build_features
from churnml.synth import AS_OF


def test_one_row_per_customer_and_no_missing(tables):
    feats = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    assert len(feats) == len(tables["customers"])
    assert feats["customer_id"].is_unique
    assert not feats[[*CATEGORICAL, *NUMERIC]].isna().any().any()


def test_usage_on_or_after_as_of_is_ignored(tables):
    """Leakage guard: adding 'future' usage must not change any feature."""
    usage = tables["usage_monthly"]
    future = usage[usage["month"] == usage["month"].max()].assign(
        month=pd.Timestamp(AS_OF).date(), sessions=10_000
    )
    polluted = pd.concat([usage, future], ignore_index=True)
    a = build_features(tables["customers"], usage, AS_OF)
    b = build_features(tables["customers"], polluted, AS_OF)
    pd.testing.assert_frame_equal(a, b)


def test_session_trend_formula():
    frame = pd.DataFrame(
        {
            "customer_id": [1],
            "plan": ["basic"],
            "country": ["ZW"],
            "age": [30],
            "monthly_fee": [9.0],
            "tenure_days": [100],
            "sessions_last3": [10.0],
            "sessions_prev3": [19.0],
            "tickets_total": [0],
            "failures_total": [0],
        }
    )
    assert add_derived(frame)["session_trend"].iloc[0] == pytest.approx(0.5)
