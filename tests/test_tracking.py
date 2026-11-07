from unittest.mock import MagicMock

from benchmarks.tracking import log_run, sanitize_metric_name


def test_sanitize_metric_name():
    assert sanitize_metric_name("recall@5 (en)") == "recall_5 _en_"
    assert sanitize_metric_name("recall_en") == "recall_en"


def test_log_run_records_params_metrics_and_artifacts(tmp_path):
    artifact = tmp_path / "r.json"
    artifact.write_text("{}", encoding="utf-8")
    fake = MagicMock()

    log_run(
        "exp",
        "run1",
        {"model": "m", "top_k": 5},
        {"recall_en": 0.8, "bad": float("nan"), "none": None},
        artifacts=[str(artifact), str(tmp_path / "missing.json")],
        mlflow_module=fake,
    )

    fake.set_experiment.assert_called_once_with("exp")
    fake.start_run.assert_called_once_with(run_name="run1")
    fake.log_param.assert_any_call("model", "m")
    fake.log_param.assert_any_call("top_k", "5")
    logged = [c.args[0] for c in fake.log_metric.call_args_list]
    assert logged == ["recall_en"]  # nan et None ignorés
    fake.log_artifact.assert_called_once_with(str(artifact))  # fichier absent ignoré
