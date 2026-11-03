import pytest
from src.language import (
    REFUSAL_MESSAGES, answer_language_matches, detect_language, strip_code,
)


@pytest.mark.parametrize("text, expected", [
    ("How to declare path parameters in FastAPI?", "en"),
    ("What is the recipe for a chocolate cake?", "en"),
    ("Comment déclarer des paramètres de chemin dans FastAPI ?", "fr"),
    ("Qu'est-ce que l'injection de dépendances dans FastAPI ?", "fr"),
    ("Quelle est la recette du gâteau au chocolat ?", "fr"),
    ("كيف أعرّف معاملات المسار (path parameters) في FastAPI؟", "ar"),
    ("ما هو حقن التبعيات (dependency injection) في FastAPI؟", "ar"),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_detect_language_returns_none_without_letters():
    assert detect_language("") is None
    assert detect_language(None) is None
    assert detect_language("??? 123 !!!") is None


def test_refusal_messages_are_detected_as_their_own_language():
    """Garde-fou : un message de refus doit être reconnu comme étant dans sa langue
    (sinon l'évaluation de la langue des réponses se tromperait sur les refus)."""
    for lang, message in REFUSAL_MESSAGES.items():
        assert detect_language(message) == lang


def test_strip_code_removes_code_urls_and_file_names():
    text = "Voir `Path` et tutorial/path-params.md :\n```python\nfrom fastapi import Path\n```\nhttps://x.dev/a"
    cleaned = strip_code(text)
    assert "fastapi" not in cleaned and "path-params" not in cleaned and "https" not in cleaned
    assert "Voir" in cleaned


def test_answer_language_ignores_code_blocks():
    answer = "Pour déclarer un paramètre, utilisez `Path` :\n```python\nfrom fastapi import FastAPI, Path\nimport the_the_the\n```"
    assert answer_language_matches(answer, "fr") is True
    assert answer_language_matches(answer, "en") is False


def test_answer_language_none_when_only_code():
    assert answer_language_matches("```python\nprint(1)\n```", "en") is None


def test_arabic_answer_with_latin_technical_terms_is_arabic():
    answer = "لتعريف معاملات المسار في FastAPI يمكنك استخدام `Path` كما في المثال التالي."
    assert answer_language_matches(answer, "ar") is True
