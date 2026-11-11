"""Command line: generate data, load PostgreSQL, train and track."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from . import __version__
from .db import connect, init_schema, load_tables, read_features, read_labels
from .features import build_features
from .synth import AS_OF, generate

TABLES = ("customers", "usage_monthly", "labels")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churnml", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write synthetic CSV files")
    gen.add_argument("--out", type=Path, default=Path("data"))
    gen.add_argument("--customers", type=int, default=2000)
    gen.add_argument("--seed", type=int, default=42)

    load = sub.add_parser("load", help="create schema and load CSVs into PostgreSQL")
    load.add_argument("--dsn", required=True)
    load.add_argument("--data", type=Path, default=Path("data"))

    train = sub.add_parser("train", help="train, evaluate and log to MLflow")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--dsn", help="read features from PostgreSQL")
    source.add_argument("--data", type=Path, help="read CSVs and build features")
    train.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    train.add_argument("--experiment", default="churn")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--register",
        metavar="NAME",
        help="register the best model under NAME with the alias 'champion'",
    )
    return parser


def read_csvs(folder: Path) -> dict[str, pd.DataFrame]:
    return {t: pd.read_csv(folder / f"{t}.csv") for t in TABLES}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    as_of: date = AS_OF
    try:
        if args.command == "generate":
            args.out.mkdir(parents=True, exist_ok=True)
            for name, frame in generate(args.customers, args.seed).items():
                frame.to_csv(args.out / f"{name}.csv", index=False)
            print(f"wrote {len(TABLES)} files to {args.out}")
        elif args.command == "load":
            tables = read_csvs(args.data)
            with connect(args.dsn) as conn:
                init_schema(conn)
                load_tables(conn, tables)
            print(f"loaded {len(tables['customers'])} customers")
        else:
            if args.dsn:
                with connect(args.dsn) as conn:
                    features = read_features(conn, as_of)
                    labels = read_labels(conn)
            else:
                tables = read_csvs(args.data)
                features = build_features(
                    tables["customers"], tables["usage_monthly"], as_of
                )
                labels = tables["labels"]
            from .train import (
                train_and_log,
            )  # heavy import (mlflow): only when training

            result = train_and_log(
                features,
                labels,
                tracking_uri=args.tracking_uri,
                experiment=args.experiment,
                seed=args.seed,
                register_as=args.register,
            )
            print(f"best model: {result.best_model}")
            for key, value in result.test.items():
                print(f"  test_{key}: {value:.3f}")
            print(f"run id: {result.run_id}")
            if result.model_version:
                print(f"registered {args.register} v{result.model_version} @champion")
    except (OSError, ValueError) as exc:
        print(f"churnml: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
