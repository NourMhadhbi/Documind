"""Tests des fonctions pures du script de comparaison (pas d'appel LLM)."""
import pytest
from benchmarks.compare_llms import (
    parse_candidate, looks_like_refusal, cites_expected_source, summarize,
)


def test_parse_candidate_keeps_colon_in_ollama_model_name():
    assert parse_candidate("ollama:qwen2.5:3b") == ("ollama", "qwen2.5:3b")


def test_parse_candidate_anthropic():
    assert parse_candidate("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")


@pytest.mark.parametrize("bad", ["qwen2.5", "gpt4:x", "ollama:", ""])
def test_parse_candidate_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_candidate(bad)


def test_refusal_detection():
    assert looks_like_refusal("Je ne sais pas, le contexte ne couvre pas ce sujet.")
    assert not looks_like_refusal("Utilisez @app.post() (tutorial/body.md).")


def test_citation_requires_text_and_sources():
    assert cites_expected_source("Voir tutorial/body.md", ["tutorial/body.md"], ["tutorial/body.md"])
    # cité dans le texte mais absent des sources récupérées -> hallucination de source
    assert not cites_expected_source("Voir tutorial/body.md", ["autre.md"], ["tutorial/body.md"])
    # source récupérée mais non citée dans la réponse
    assert not cites_expected_source("Réponse sans citation", ["tutorial/body.md"], ["tutorial/body.md"])


def test_summarize_ignores_errors_and_separates_trap():
    rows = [
        {"trap": False, "latency_seconds": 2.0, "cites_expected": True, "refused": False},
        {"trap": False, "latency_seconds": 4.0, "cites_expected": False, "refused": False},
        {"trap": True, "latency_seconds": 3.0, "cites_expected": False, "refused": True},
        {"trap": False, "error": "Ollama injoignable"},
    ]
    s = summarize(rows)
    assert s["errors"] == 1
    assert s["questions_ok"] == 3
    assert s["avg_latency_s"] == 3.0
    assert s["citation_rate"] == 0.5
    assert s["refuses_trap"] == 1.0