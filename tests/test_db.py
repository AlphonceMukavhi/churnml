"""Needs PostgreSQL. Run with CHURNML_TEST_DSN=postgresql://user:pw@host/db."""

import numpy as np
import pytest

from churnml.db import connect, init_schema, load_tables, read_features, read_labels
from churnml.features import build_features
from churnml.synth import AS_OF

pytestmark = pytest.mark.db


def test_sql_features_match_pandas_features(tables, dsn):
    with connect(dsn) as conn:
        init_schema(conn)
        load_tables(conn, tables)
        from_sql = read_features(conn, AS_OF)
        labels = read_labels(conn)
    reference = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    assert list(from_sql.columns) == list(reference.columns)
    for column in reference.select_dtypes("number").columns:
        assert np.allclose(from_sql[column], reference[column]), column
    assert len(labels) == len(tables["labels"])


def test_constraints_reject_bad_rows(tables, dsn):
    import psycopg

    with connect(dsn) as conn:
        init_schema(conn)
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO customers VALUES (1, '2024-01-01', 'gold', 'ZW', 30, 9)"
            )
