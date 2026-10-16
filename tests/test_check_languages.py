"""Tests des fonctions pures de check_languages (aucun modèle chargé)."""
from benchmarks.check_languages import rank_of_expected, recall_by_language


def test_rank_of_expected_first_position():
    assert rank_of_expected(["a.md", "b.md"], ["a.md"]) == 1


def test_rank_of_expected_later_position():
    assert rank_of_expected(["x.md", "y.md", "a.md"], ["a.md"]) == 3


def test_rank_of_expected_absent():
    assert rank_of_expected(["x.md", "y.md"], ["a.md"]) is None


def test_recall_by_language_separates_languages():
    rows = [
        {"lang": "en", "rank": 1},
        {"lang": "en", "rank": 2},
        {"lang": "ar", "rank": None},
        {"lang": "ar", "rank": 4},
    ]
    result = recall_by_language(rows)
    assert result["en"] == 1.0
    assert result["ar"] == 0.5


def test_questions_are_balanced_across_languages():
    """Chaque langue doit avoir exactement les mêmes sujets, sinon la
    comparaison entre langues serait faussée."""
    from collections import Counter
    from benchmarks.check_languages import QUESTIONS

    counts = Counter(q["lang"] for q in QUESTIONS)
    assert set(counts) == {"en", "fr", "ar"}
    assert len(set(counts.values())) == 1  # même nombre de questions par langue
    assert all(q["expected"] for q in QUESTIONS)
