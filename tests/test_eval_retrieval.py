import json

import pytest

from benchmarks.eval_retrieval import (
    load_questions,
    mean_distance,
    run_retrieval,
    tracking_metrics,
)
from benchmarks.io_utils import read_json, write_json_atomic
from benchmarks.metrics import aggregate_retrieval


def _q(id_, lang, expected, trap=False):
    return {
        "id": id_,
        "topic": "t",
        "lang": lang,
        "question": f"Q {id_}",
        "expected_sources": expected,
        "trap": trap,
    }


def _fake_retrieve(mapping):
    def retrieve(text, k):
        value = mapping[text]
        if isinstance(value, Exception):
            raise value
        return [{"source": s, "score": d, "text": "t"} for s, d in value][:k]

    return retrieve


def test_run_retrieval_computes_rank_and_top_distance():
    questions = [_q("a", "en", ["x.md"])]
    rows = run_retrieval(
        questions, _fake_retrieve({"Q a": [("y.md", 0.4), ("x.md", 0.6)]}), top_k=5
    )
    assert rows[0]["rank"] == 2 and rows[0]["top1_distance"] == 0.4
    assert rows[0]["sources"] == ["y.md", "x.md"] and rows[0]["error"] is None


def test_run_retrieval_leaves_rank_empty_for_traps_but_keeps_distance():
    rows = run_retrieval(
        [_q("t", "fr", [], trap=True)], _fake_retrieve({"Q t": [("a.md", 0.9)]}), 5
    )
    assert rows[0]["rank"] is None and rows[0]["top1_distance"] == 0.9


def test_run_retrieval_records_errors_and_continues():
    questions = [_q("a", "en", ["x.md"]), _q("b", "en", ["x.md"])]
    mapping = {"Q a": RuntimeError("index absent"), "Q b": [("x.md", 0.2)]}
    rows = run_retrieval(questions, _fake_retrieve(mapping), 5)
    assert rows[0]["error"] == "index absent" and rows[1]["rank"] == 1


def test_summary_and_tracking_metrics():
    rows = [
        {"lang": "en", "rank": 1, "trap": False},
        {"lang": "en", "rank": None, "trap": False},
        {"lang": "ar", "rank": 2, "trap": False},
    ]
    summary = aggregate_retrieval(rows, 5)
    metrics = tracking_metrics(summary, n_errors=0)
    assert metrics["recall_en"] == 0.5 and metrics["recall_ar"] == 1.0
    assert metrics["recall_overall"] == pytest.approx(2 / 3)
    assert metrics["n_errors"] == 0.0


def test_mean_distance_separates_traps_and_normal_questions():
    rows = [
        {"trap": False, "top1_distance": 0.2},
        {"trap": False, "top1_distance": 0.4},
        {"trap": True, "top1_distance": 0.9},
        {"trap": True, "top1_distance": None},
    ]
    assert mean_distance(rows, False) == pytest.approx(0.3)
    assert mean_distance(rows, True) == 0.9


def test_load_questions_rejects_invalid_golden(tmp_path):
    bad = tmp_path / "g.json"
    bad.write_text(json.dumps({"questions": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalide"):
        load_questions(str(bad))


def test_write_json_atomic_roundtrip_and_no_leftover_tmp(tmp_path):
    target = tmp_path / "sub" / "r.json"
    write_json_atomic(str(target), {"é": "ع"})
    assert read_json(str(target)) == {"é": "ع"}
    assert not list((tmp_path / "sub").glob("*.tmp"))
