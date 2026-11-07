"""
Tests unitaires du Sprint 3.

Comme pour test_embeddings.py (Sprint 2), on mocke le modèle et le
vector store : un test unitaire doit être rapide et reproductible,
indépendant du téléchargement d'un vrai modèle ou de l'existence d'un
vrai index sur disque.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval import RetrievalError, reset_cache, retrieve
from src.vector_store import VectorStoreError


@pytest.fixture(autouse=True)
def clear_cache():
    """Réinitialise le cache singleton avant et après chaque test, pour
    qu'un test ne pollue pas l'état d'un autre (isolation stricte)."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------
# Validation des entrées
# ---------------------------------------------------------------


def test_retrieve_empty_question_raises():
    with pytest.raises(RetrievalError, match="vide"):
        retrieve("")


def test_retrieve_whitespace_only_question_raises():
    """Une question faite uniquement d'espaces doit être traitée comme
    vide, pas encodée telle quelle."""
    with pytest.raises(RetrievalError, match="vide"):
        retrieve("    \n\t  ")


def test_retrieve_invalid_top_k_raises():
    with pytest.raises(RetrievalError, match="top_k"):
        retrieve("Une question valide ?", top_k=0)


def test_retrieve_negative_top_k_raises():
    with pytest.raises(RetrievalError, match="top_k"):
        retrieve("Une question valide ?", top_k=-3)


# ---------------------------------------------------------------
# Comportement nominal (avec mocks)
# ---------------------------------------------------------------


def test_retrieve_returns_store_results():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    mock_store = MagicMock()
    mock_store.query.return_value = [
        {
            "text": "FastAPI route example",
            "source": "tutorial/first-steps.md",
            "score": 0.12,
        },
    ]

    with (
        patch("src.retrieval._get_model", return_value=mock_model),
        patch("src.retrieval._get_store", return_value=mock_store),
    ):
        results = retrieve("How to create a POST route?", top_k=3)

    assert len(results) == 1
    assert results[0]["source"] == "tutorial/first-steps.md"
    mock_store.query.assert_called_once()


def test_retrieve_normalizes_embeddings():
    """Vérifie que la question est bien encodée avec normalize_embeddings=True
    -- condition requise pour la cohérence avec la métrique cosinus de
    l'index (voir ADR 002). Un oubli ici casserait silencieusement la
    pertinence du retrieval, sans lever d'erreur -- ce test protège contre
    cette régression silencieuse."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_store = MagicMock()
    mock_store.query.return_value = []

    with (
        patch("src.retrieval._get_model", return_value=mock_model),
        patch("src.retrieval._get_store", return_value=mock_store),
    ):
        retrieve("Une question", top_k=5)

    call_kwargs = mock_model.encode.call_args.kwargs
    assert call_kwargs.get("normalize_embeddings") is True


def test_retrieve_empty_results_is_not_an_error():
    """Aucun résultat trouvé est un cas valide (ex: index vide ou
    question hors sujet), pas une exception à lever."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_store = MagicMock()
    mock_store.query.return_value = []

    with (
        patch("src.retrieval._get_model", return_value=mock_model),
        patch("src.retrieval._get_store", return_value=mock_store),
    ):
        results = retrieve("Question sans rapport avec le corpus", top_k=5)

    assert results == []


# ---------------------------------------------------------------
# Gestion d'erreurs
# ---------------------------------------------------------------


def test_retrieve_propagates_vector_store_error_as_retrieval_error():
    """Une VectorStoreError (erreur interne du Sprint 2) ne doit jamais
    fuiter telle quelle vers l'appelant -- elle doit être traduite en
    RetrievalError, pour que l'API (Sprint 6) n'ait besoin de connaître
    qu'un seul type d'erreur, indépendamment du backend vectoriel choisi."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_store = MagicMock()
    mock_store.query.side_effect = VectorStoreError("panne simulée de l'index")

    with (
        patch("src.retrieval._get_model", return_value=mock_model),
        patch("src.retrieval._get_store", return_value=mock_store),
    ):
        with pytest.raises(RetrievalError, match="recherche"):
            retrieve("Une question valide", top_k=5)


def test_retrieve_wraps_encoding_failure():
    """Un échec du modèle lors de l'encodage (ex: modèle corrompu) doit
    aussi être enveloppé proprement, jamais laissé remonter brut."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("modèle corrompu simulé")
    mock_store = MagicMock()

    with (
        patch("src.retrieval._get_model", return_value=mock_model),
        patch("src.retrieval._get_store", return_value=mock_store),
    ):
        with pytest.raises(RetrievalError, match="encodage"):
            retrieve("Une question valide", top_k=5)


# ---------------------------------------------------------------
# Pattern singleton : le modèle ne doit être chargé qu'une seule fois
# ---------------------------------------------------------------


def test_model_is_loaded_only_once_across_multiple_calls():
    """Vérifie le bénéfice du pattern singleton : 2 appels à retrieve()
    ne doivent déclencher qu'UNE SEULE instanciation de SentenceTransformer,
    pas une par appel (sinon la latence par question serait inacceptable
    en production)."""
    with patch("src.retrieval.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_instance

        mock_store = MagicMock()
        mock_store.query.return_value = []

        with patch("src.retrieval._get_store", return_value=mock_store):
            retrieve("Première question", top_k=1)
            retrieve("Deuxième question", top_k=1)

        mock_cls.assert_called_once()
