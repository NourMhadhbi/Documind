import json

from api.request_log import build_record, log_request

RESULT = {
    "language": "fr",
    "min_distance": 0.42,
    "refused": False,
    "refusal_reason": None,
    "latency_seconds": 12.5,
    "sources": ["tutorial/body.md", "tutorial/query-params.md"],
    "answer": "Réponse secrète",
}


def test_build_record_has_expected_fields_and_no_question_text():
    record = build_record("Comment créer une route POST ?", RESULT)
    assert record["language"] == "fr"
    assert record["n_sources"] == 2
    assert record["question_length"] == len("Comment créer une route POST ?")
    assert record["refused"] is False
    assert "Comment" not in json.dumps(record) and "secrète" not in json.dumps(record)


def test_log_request_appends_one_json_line_per_call(tmp_path):
    target = tmp_path / "logs" / "requests.jsonl"
    log_request("q1", RESULT, path=target)
    log_request("q2", {**RESULT, "refused": True, "language": None}, path=target)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["refused"] is True
    assert json.loads(lines[1])["language"] == "unknown"


def test_log_request_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log_request("q", RESULT, path=blocker / "sub" / "requests.jsonl")  # chemin impossible
