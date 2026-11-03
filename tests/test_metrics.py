from benchmarks.metrics import (
    aggregate_retrieval, cited_markdown_files, invalid_citations, looks_like_refusal,
    rank_of_expected, reciprocal_rank,
)


def test_rank_of_expected():
    assert rank_of_expected(["a.md", "b.md"], ["b.md"]) == 2
    assert rank_of_expected(["a.md"], ["z.md"]) is None


def test_reciprocal_rank_respects_k():
    assert reciprocal_rank(1, 5) == 1.0
    assert reciprocal_rank(4, 5) == 0.25
    assert reciprocal_rank(6, 5) == 0.0
    assert reciprocal_rank(None, 5) == 0.0


def test_aggregate_retrieval_ignores_traps_and_errors():
    rows = [
        {"lang": "en", "rank": 1, "trap": False}, {"lang": "en", "rank": None, "trap": False},
        {"lang": "ar", "rank": 3, "trap": False}, {"lang": "ar", "rank": None, "trap": True},
        {"lang": "ar", "rank": None, "trap": False, "error": "boom"},
    ]
    s = aggregate_retrieval(rows, k=5)
    assert s["en"] == {"n": 2, "recall": 0.5, "mrr": 0.5}
    assert s["ar"]["n"] == 1 and s["ar"]["recall"] == 1.0
    assert s["overall"]["n"] == 3


def test_looks_like_refusal_in_three_languages():
    assert looks_like_refusal("Je ne sais pas, le contexte ne couvre pas ce sujet.")
    assert looks_like_refusal("I could not find information on this topic.")
    assert looks_like_refusal("لم أجد معلومات حول هذا الموضوع في الوثائق.")
    assert not looks_like_refusal("Utilisez @app.post() (tutorial/body.md).")


def test_cited_markdown_files_unique_and_ordered():
    answer = "Voir [dependencies-with-yield.md] et `./tutorial/body.md`, puis dependencies-with-yield.md."
    assert cited_markdown_files(answer) == ["dependencies-with-yield.md", "tutorial/body.md"]


def test_invalid_citations_accepts_suffix_matches():
    sources = ["tutorial/dependencies/dependencies-with-yield.md", "tutorial/body.md"]
    assert invalid_citations("Voir dependencies-with-yield.md et tutorial/body.md", sources) == []


def test_invalid_citations_flags_invented_paths():
    """Cas réel observé : le modèle avait cité un chemin de traduction inexistant."""
    sources = ["tutorial/path-params.md"]
    answer = "Selon la source `docs/ru/docs/tutorial/path-params.md`, utilisez Path."
    assert invalid_citations(answer, sources) == ["docs/ru/docs/tutorial/path-params.md"]
