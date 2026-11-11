import mlflow

from churnml.features import build_features
from churnml.synth import AS_OF
from churnml.train import split_data, train_and_log


def _run(tables, tmp_path, name):
    feats = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    uri = f"sqlite:///{tmp_path / name}"
    return train_and_log(feats, tables["labels"], tracking_uri=uri, seed=3), uri


def test_split_has_no_overlap_and_is_stratified(tables):
    feats = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    x_tr, x_va, x_te, y_tr, y_va, y_te = split_data(feats, tables["labels"], 3)
    assert not (set(x_tr.index) & set(x_va.index))
    assert not (set(x_va.index) & set(x_te.index))
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_model_beats_a_floor_and_is_logged(tables, tmp_path):
    result, uri = _run(tables, tmp_path, "a.db")
    assert result.best_model in {"logreg", "hist_gb"}
    assert result.test["roc_auc"] > 0.75  # a regression guard, not a target
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(result.run_id)
    assert run.data.params["best_model"] == result.best_model
    assert "test_roc_auc" in run.data.metrics


def test_training_is_reproducible(tables, tmp_path):
    first, _ = _run(tables, tmp_path, "b.db")
    second, _ = _run(tables, tmp_path, "c.db")
    assert first.test == second.test


def test_register_sets_champion_alias(tables, tmp_path):
    feats = build_features(tables["customers"], tables["usage_monthly"], AS_OF)
    uri = f"sqlite:///{tmp_path / 'r.db'}"
    result = train_and_log(
        feats, tables["labels"], tracking_uri=uri, seed=3, register_as="churn"
    )
    assert result.model_version == "1"
    mlflow.set_tracking_uri(uri)
    champion = mlflow.MlflowClient().get_model_version_by_alias("churn", "champion")
    assert str(champion.version) == "1"
    model = mlflow.sklearn.load_model("models:/churn@champion")
    assert model.predict_proba(feats.head(2)).shape == (2, 2)
