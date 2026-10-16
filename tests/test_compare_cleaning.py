"""Tests des fonctions pures de compare_cleaning (aucun modèle chargé)."""
import numpy as np

from benchmarks.compare_cleaning import (
    legacy_clean_text, build_chunks_with, rank_for_question, summarize, _verdict,
)


def test_legacy_baseline_reproduces_the_known_underscore_bug():
    """La référence doit être fidèle à l'ancien code, défaut compris."""
    assert "itemid and userid" in legacy_clean_text("item_id and user_id")


def test_new_cleaning_does_not_have_that_bug():
    from src.ingestion import clean_text
    assert "item_id and user_id" in clean_text("item_id and user_id")


def test_rank_for_question_returns_position_of_expected_source():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    sources = ["a.md", "b.md", "c.md", "d.md"]
    assert rank_for_question(scores, sources, ["d.md"], depth=4) == 2   # b, d, c, a
    assert rank_for_question(scores, sources, ["b.md"], depth=4) == 1


def test_rank_for_question_none_when_outside_depth():
    scores = np.array([0.9, 0.8, 0.1])
    assert rank_for_question(scores, ["a", "b", "c"], ["c"], depth=2) is None


def test_summarize_computes_recall_and_mrr_per_language():
    rows = [
        {"lang": "en", "rank": 1}, {"lang": "en", "rank": 6},
        {"lang": "ar", "rank": None}, {"lang": "ar", "rank": 2},
    ]
    s = summarize(rows, k=5)
    assert s["en"]["recall"] == 0.5
    assert abs(s["en"]["mrr"] - (1 + 1 / 6) / 2) < 1e-9
    assert s["ar"]["recall"] == 0.5
    assert abs(s["ar"]["mrr"] - 0.25) < 1e-9


def test_build_chunks_with_uses_the_given_cleaning_function():
    docs = [{"filename": "x.md", "content": "hello world"}]
    chunks = build_chunks_with(lambda t, k: t.upper(), docs, 100, 10, True)
    assert chunks == [{"chunk_id": "x.md_0", "source": "x.md", "text": "HELLO WORLD"}]


def test_verdict_labels():
    assert _verdict(4, 2) == "mieux"
    assert _verdict(2, 4) == "moins bien"
    assert _verdict(3, 3) == "="
    assert _verdict(None, 5) == "mieux"      # absent -> trouvé
    assert _verdict(5, None) == "moins bien"  # trouvé -> absent


# ---------------------------------------------------------------
# Ajouts : cache d'embeddings, meilleur chunk, ablation
# ---------------------------------------------------------------

from benchmarks.compare_cleaning import best_expected, encode_with_cache, make_variant


class _CountingModel:
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_encode_with_cache_only_encodes_unseen_texts():
    model, cache = _CountingModel(), {}
    encode_with_cache(model, ["a", "bb", "a"], cache)       # "a" en double : encodé une fois
    assert model.encoded == ["a", "bb"]
    encode_with_cache(model, ["bb", "ccc"], cache)           # seul "ccc" est nouveau
    assert model.encoded == ["a", "bb", "ccc"]


def test_encode_with_cache_keeps_order_of_input_texts():
    model, cache = _CountingModel(), {}
    result = encode_with_cache(model, ["ccc", "a", "ccc"], cache)
    assert [row[0] for row in result] == [3.0, 1.0, 3.0]


def test_best_expected_returns_rank_and_chunk():
    chunks = [{"chunk_id": "a_0", "source": "a.md", "text": "A"},
              {"chunk_id": "b_0", "source": "b.md", "text": "B"},
              {"chunk_id": "b_1", "source": "b.md", "text": "B2"}]
    scores = np.array([0.1, 0.5, 0.9])                       # ordre : b_1, b_0, a_0
    best = best_expected(scores, chunks, ["b.md"], depth=3)
    assert best == {"rank": 1, "chunk_id": "b_1", "text": "B2"}
    assert best_expected(scores, chunks, ["zzz.md"], depth=3) is None


def test_ablation_variant_keeps_the_rule_that_was_disabled():
    raw = "Copy it.\n\n{* ../../docs_src/x.py hl[1] *}\n\nRun it."
    assert "docs_src" not in make_variant(None)(raw, True)
    assert "docs_src" in make_variant("RE_INCLUDE_DIRECTIVE")(raw, True)   # directive conservée


def test_ablation_does_not_leak_into_the_next_call():
    raw = "A\n\n{* x.py *}\n\nB"
    make_variant("RE_INCLUDE_DIRECTIVE")(raw, True)
    from src.ingestion import clean_text
    assert "x.py" not in clean_text(raw, True)                # règle bien réactivée après


def test_retained_variant_keeps_anchors_and_previous_variant_strips_them():
    from benchmarks.compare_cleaning import corrected_without_anchors
    raw = "# First Steps { #first-steps }"
    assert make_variant(None)(raw, True) == "First Steps { #first-steps }"
    assert corrected_without_anchors(raw, True) == "First Steps"