import os

import pytest

from churnml.synth import AS_OF, generate


@pytest.fixture(scope="session")
def tables():
    return generate(n_customers=800, seed=7, as_of=AS_OF)


@pytest.fixture
def dsn():
    value = os.environ.get("CHURNML_TEST_DSN")
    if not value:
        pytest.skip("set CHURNML_TEST_DSN to run PostgreSQL tests")
    return value
