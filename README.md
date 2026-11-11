# churnml

An end-to-end machine-learning pipeline that predicts which subscription customers are likely to leave (churn). It covers generating data, loading it into PostgreSQL, building features in SQL and pandas, training and comparing models, and tracking every run in MLflow.

The data is synthetic and generated with a fixed seed, so every run is reproducible and no real customer data is involved.

## Quick start (no database needed)

```bash
uv sync
uv run churnml generate --out data
uv run churnml train --data data --register churn
uv run mlflow ui --port 5000        # browse runs at http://localhost:5000
```

`--register churn` saves the best model in the MLflow model registry with the alias `champion`. My [churn-api](https://github.com/AlphonceMukavhi/churn-api) service loads that model and serves predictions over HTTP.

## With PostgreSQL

```bash
export DSN=postgresql://churn:churn@localhost:5432/churn
uv run churnml load --dsn "$DSN" --data data     # creates the schema and loads the CSVs
uv run churnml train --dsn "$DSN"                # features are built in SQL
CHURNML_TEST_DSN="$DSN" uv run pytest            # also runs the database tests
```

## Design decisions

- **Features are written twice, in SQL and in pandas.** A test checks that both versions give the same numbers, so the database path and the file path cannot drift apart.
- **No leakage.** Features only use activity before the prediction date, and a test proves that later data is ignored.
- **Honest evaluation.** The data is split 60/20/20. Models are compared on the validation set by PR-AUC, because churners are the minority. The test set is used once, for the final model only.
- **Reproducible.** The seed is fixed everywhere, and a test trains twice and checks the results are identical.
- **Safe model files.** MLflow stores the scikit-learn model with skops, and only types this code produces are trusted.

## Tests and CI

```bash
uv run pytest -q
uv run ruff check . && uv run mypy src
```

GitHub Actions runs everything against a real PostgreSQL service container.

## Why I built it

I built this while teaching myself machine learning engineering. I wanted to practise the parts that make a model trustworthy, not only the model itself.
