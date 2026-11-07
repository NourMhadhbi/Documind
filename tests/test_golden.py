"""Tests du validateur de jeu de référence, et du jeu livré lui-même."""

from pathlib import Path

import pytest

from benchmarks.validate_golden import (
    check_sources_exist,
    coverage_report,
    load_golden,
    validate_schema,
)

GOLDEN_PATH = Path(__file__).parent / "golden_qa.json"


def _q(**overrides):
    base = {
        "id": "t-en",
        "topic": "t",
        "lang": "en",
        "question": "Q ?",
        "expected_sources": ["a.md"],
        "trap": False,
    }
    base.update(overrides)
    return base


def test_valid_minimal_data_has_no_errors():
    assert validate_schema({"questions": [_q()]}) == []


def test_empty_or_missing_questions_is_an_error():
    assert validate_schema({"questions": []})
    assert validate_schema({})


def test_missing_fields_are_reported():
    bad = _q()
    del bad["lang"]
    errors = validate_schema({"questions": [bad]})
    assert any("champs manquants" in e and "lang" in e for e in errors)


def test_duplicate_ids_are_reported():
    errors = validate_schema({"questions": [_q(), _q()]})
    assert any("en double" in e for e in errors)


def test_invalid_language_is_reported():
    assert any("langue" in e for e in validate_schema({"questions": [_q(lang="de")]}))


def test_trap_must_have_no_expected_source():
    errors = validate_schema({"questions": [_q(trap=True, expected_sources=["a.md"])]})
    assert any("piège" in e for e in errors)


def test_normal_question_needs_an_expected_source():
    errors = validate_schema({"questions": [_q(expected_sources=[])]})
    assert any("au moins une source" in e for e in errors)


def test_empty_question_text_is_reported():
    assert any("question vide" in e for e in validate_schema({"questions": [_q(question="  ")]}))


def test_check_sources_exist_flags_missing_files(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    data = {"questions": [_q(expected_sources=["a.md", "ghost.md"])]}
    errors = check_sources_exist(data, str(tmp_path))
    assert len(errors) == 1 and "ghost.md" in errors[0]


def test_coverage_report_detects_topic_missing_a_language():
    data = {
        "questions": [
            _q(id="x-en", topic="x", lang="en"),
            _q(id="x-fr", topic="x", lang="fr"),
            _q(id="y-en", topic="y", lang="en"),
            _q(id="y-fr", topic="y", lang="fr"),
            _q(id="y-ar", topic="y", lang="ar"),
        ]
    }
    report = coverage_report(data)
    assert report["incomplete_topics"] == {"x": ["ar"]}
    assert report["per_lang"] == {"ar": 1, "en": 2, "fr": 2}


# ---- Le jeu livré -----------------------------------------------------------


def test_shipped_golden_file_is_structurally_valid():
    assert validate_schema(load_golden(str(GOLDEN_PATH))) == []


def test_shipped_golden_file_covers_every_language_evenly():
    report = coverage_report(load_golden(str(GOLDEN_PATH)))
    assert report["incomplete_topics"] == {}
    assert len(set(report["per_lang"].values())) == 1  # même nombre par langue
    assert all(n >= 1 for n in report["traps_per_lang"].values())  # au moins un piège par langue


@pytest.mark.skipif(not Path("data/raw").is_dir(), reason="data/raw absent (lance `dvc pull`)")
def test_shipped_golden_sources_exist_in_corpus():
    assert check_sources_exist(load_golden(str(GOLDEN_PATH)), "data/raw") == []
