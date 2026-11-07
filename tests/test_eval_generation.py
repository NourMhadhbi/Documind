import pytest

from benchmarks.eval_generation import (
    evaluate_one,
    run_evaluation,
    select_questions,
    summarize,
    tracking_metrics,
)
from benchmarks.io_utils import read_json


def _q(id_, lang="en", expected=("tutorial/body.md",), trap=False, **extra):
    base = {
        "id": id_,
        "topic": "t",
        "lang": lang,
        "question": f"Q {id_}",
        "expected_sources": list(expected),
        "trap": trap,
    }
    base.update(extra)
    return base


def _result(
    answer="Use the body model. See tutorial/body.md.",
    sources=("tutorial/body.md",),
    **extra,
):
    base = {
        "answer": answer,
        "sources": list(sources),
        "contexts": ["ctx"],
        "refused": False,
        "refusal_reason": None,
        "latency_seconds": 12.5,
        "min_distance": 0.3,
        "llm_provider": "ollama",
        "llm_model": "m",
    }
    base.update(extra)
    return base


def test_select_questions_filters_and_limits():
    qs = [
        _q("a", "en"),
        _q("b", "fr"),
        _q("c", "ar", trap=True, expected=[]),
        _q("d", "en"),
    ]
    assert [q["id"] for q in select_questions(qs, langs=["en"])] == ["a", "d"]
    assert [q["id"] for q in select_questions(qs, ids=["b"])] == ["b"]
    assert [q["id"] for q in select_questions(qs, include_traps=False)] == [
        "a",
        "b",
        "d",
    ]
    assert [q["id"] for q in select_questions(qs, limit=2)] == ["a", "b"]


def test_evaluate_one_good_answer():
    row = evaluate_one(_q("a"), lambda text: _result())
    assert row["language_ok"] is True and row["source_ok"] is True
    assert row["refused"] is False and row["invalid_citations"] == [] and row["error"] is None


def test_evaluate_one_flags_wrong_language_missing_source_and_invented_citation():
    answer = "Voici la réponse, selon `docs/ru/docs/tutorial/body.md`, vous pouvez utiliser Pydantic pour valider."
    row = evaluate_one(_q("a"), lambda text: _result(answer=answer, sources=["tutorial/other.md"]))
    assert row["language_ok"] is False  # question anglaise, réponse française
    assert row["source_ok"] is False
    assert row["invalid_citations"] == ["docs/ru/docs/tutorial/body.md"]


def test_evaluate_one_detects_refusal_by_flag_or_by_wording():
    by_flag = evaluate_one(
        _q("t", trap=True, expected=[]), lambda t: _result(refused=True, answer="x")
    )
    by_text = evaluate_one(
        _q("t", trap=True, expected=[]),
        lambda t: _result(answer="I could not find information."),
    )
    assert by_flag["refused"] is True and by_text["refused"] is True
    assert by_flag["source_ok"] is None


def test_evaluate_one_records_errors_without_raising():
    def boom(text):
        raise RuntimeError("Ollama injoignable")

    row = evaluate_one(_q("a"), boom)
    assert row["error"] == "Ollama injoignable" and "answer" not in row


def test_summarize_rates_per_language():
    rows = [
        evaluate_one(_q("a", "en"), lambda t: _result()),
        evaluate_one(
            _q("b", "en"),
            lambda t: _result(
                answer="Voici la réponse, vous pouvez utiliser le modèle.",
                sources=["x.md"],
            ),
        ),
        evaluate_one(
            _q("t", "en", expected=[], trap=True),
            lambda t: _result(
                refused=True,
                answer="I could not find information.",
                latency_seconds=0.0,
            ),
        ),
        evaluate_one(_q("e", "fr"), lambda t: (_ for _ in ()).throw(RuntimeError("x"))),
    ]
    s = summarize(rows)
    assert s["en"]["n"] == 3 and s["en"]["n_errors"] == 0
    assert s["en"]["source_ok_rate"] == 0.5
    assert s["en"]["language_ok_rate"] == pytest.approx(2 / 3)
    assert s["en"]["trap_refusal_rate"] == 1.0 and s["en"]["wrong_refusal_rate"] == 0.0
    assert s["en"]["avg_latency_s"] == 12.5  # le refus sans LLM (latence 0) est ignoré
    assert s["fr"]["n_errors"] == 1 and s["fr"]["language_ok_rate"] is None
    assert s["overall"]["n"] == 4


def test_run_evaluation_writes_after_each_question_and_resumes(tmp_path):
    out = str(tmp_path / "gen.json")
    calls = []
    failures_left = {"Q b": 1}  # « b » échoue une seule fois

    def generate(text):
        calls.append(text)
        if failures_left.get(text, 0) > 0:
            failures_left[text] -= 1
            raise RuntimeError("timeout")
        return _result()

    questions = [_q("a"), _q("b")]
    run_evaluation(questions, generate, out, {"run_name": "t"}, progress=lambda m: None)
    first = read_json(out)
    assert [r["error"] for r in first["rows"]] == [None, "timeout"]

    calls.clear()
    run_evaluation(
        questions,
        generate,
        out,
        {"run_name": "t"},
        resume=True,
        progress=lambda m: None,
    )
    assert calls == ["Q b"]  # "a" déjà fait ; "b" (en erreur) est réessayé
    assert [r["error"] for r in read_json(out)["rows"]] == [None, None]


def test_tracking_metrics_flattens_groups():
    rows = [evaluate_one(_q("a", "en"), lambda t: _result())]
    metrics = tracking_metrics(summarize(rows))
    assert metrics["source_ok_en"] == 1.0 and metrics["source_ok_overall"] == 1.0
    assert metrics["latency_s_en"] == 12.5 and metrics["n_errors_overall"] == 0
