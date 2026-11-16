import json

import pandas as pd

from monitoring.drift_report import (
    common_columns,
    load_baseline,
    load_logs,
    prepare,
    split_reference_from_logs,
    summarize,
)


def _write_logs(path, rows, bad_line=False):
    lines = [json.dumps(r) for r in rows]
    if bad_line:
        lines.insert(1, "{ligne corrompue")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_logs_skips_corrupted_lines_and_sorts_by_time(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [
        {"ts": "2026-11-02T10:00:00+00:00", "language": "en", "min_distance": 0.5},
        {"ts": "2026-11-01T10:00:00+00:00", "language": "fr", "min_distance": 0.4},
    ]
    _write_logs(path, rows, bad_line=True)
    frame = load_logs(str(path))
    assert list(frame["language"]) == ["fr", "en"]


def test_load_baseline_ignores_errors_and_missing_distances(tmp_path):
    path = tmp_path / "baseline.json"
    data = {
        "rows": [
            {"lang": "ar", "top1_distance": 0.7, "error": None},
            {"lang": "en", "top1_distance": None, "error": None},
            {"lang": "fr", "top1_distance": 0.3, "error": "boom"},
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    frame = load_baseline(str(path))
    assert frame.to_dict("records") == [{"language": "ar", "min_distance": 0.7}]


def test_split_and_common_columns_and_prepare():
    logs = pd.DataFrame(
        {"language": list("enfr"), "min_distance": [0.1, 0.2, 0.3, 0.4], "refused": [False] * 4}
    )
    reference, current = split_reference_from_logs(logs, 3)
    assert len(reference) == 3 and len(current) == 1
    baseline = pd.DataFrame({"language": ["en"], "min_distance": [0.5]})
    assert common_columns(baseline, logs) == ["language", "min_distance"]
    assert prepare(logs, ["language", "refused"])["refused"].iloc[0] == "False"


def test_summarize_reports_refusal_rate():
    frame = pd.DataFrame({"language": ["en", "fr"], "refused": ["True", "False"]})
    assert "50.0%" in summarize(frame, "Actuel")
