"""
Module de retrieval : transforme une question en langage naturel en une
liste de chunks pertinents, en s'appuyant sur le vector store construit
au Sprint 2 (voir docs/adr/002-vector-database-choice.md pour le choix
de ChromaDB et de la métrique cosinus).
"""

import logging
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from config import settings
from src.vector_store import VectorStore, VectorStoreError, get_vector_store

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Erreur métier du retrieval (question invalide, échec de recherche).
    Distincte de VectorStoreError pour ne pas fuiter de détail
    d'implémentation du vector store vers les couches supérieures
    (l'API du Sprint 6 ne doit connaître que RetrievalError)."""

    pass


# ================================================================
# Pattern singleton (module-level cache) pour le modèle et le store.
#
# Pourquoi : SentenceTransformer("all-MiniLM-L6-v2") charge ~90 Mo de
# poids en mémoire. Le recharger à CHAQUE appel de retrieve() (ce qui
# arriverait si on l'instanciait à l'intérieur de la fonction) ajouterait
# plusieurs centaines de millisecondes de latence par question -- un
# coût inacceptable pour un système conversationnel. On le charge une
# seule fois, au premier appel, et on le réutilise ensuite.
# ================================================================
_model: Optional[SentenceTransformer] = None
_store: Optional[VectorStore] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(
            "Chargement du modèle d'embedding (une seule fois):%s",
            settings.embedding_model_name,
        )
        _model = SentenceTransformer(
            settings.embedding_model_name, device=settings.embedding_device
        )
    return _model


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = get_vector_store(
            backend=settings.vector_store_backend,
            index_path=settings.index_path,
            collection_name=settings.vector_collection_name,
        )
    return _store


def reset_cache() -> None:
    """Réinitialise le modèle et le store mis en cache.
    Utile pour les tests (isolation entre cas de test) et pour forcer
    un rechargement après un changement de configuration en cours
    d'exécution (rare, mais utile en développement interactif)."""
    global _model, _store
    _model = None
    _store = None


def retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """
    Retourne les top_k chunks les plus pertinents pour une question.

    Args:
        question: question en langage naturel posée par l'utilisateur.
        top_k: nombre de chunks à retourner (doit être strictement positif).

    Returns:
        Liste de dicts {"text": ..., "source": ..., "score": ...},
        triés du plus pertinent au moins pertinent. Liste vide si
        l'index ne contient aucun résultat (cas valide, pas une erreur).

    Raises:
        RetrievalError: question vide/invalide, top_k invalide, ou
            échec de l'encodage/de la recherche.
    """
    if not question or not question.strip():
        raise RetrievalError("La question ne peut pas être vide")
    if top_k <= 0:
        raise RetrievalError("top_k doit être strictement positif ")

    question = question.strip()

    model = _get_model()
    store = _get_store()

    try:
        embedding = model.encode([question], normalize_embeddings=True)[0]
    except Exception as exc:
        raise RetrievalError(f"Échec de l'encodage de la question : {exc}") from exc

    try:
        results = store.query(embedding.tolist(), top_k=top_k)
    except VectorStoreError as exc:
        raise RetrievalError(f"Échec de la recherche dans l'index : {exc}") from exc

    if not results:
        logger.warning("Aucun résultat trouvé pour la question : '%s'", question)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    sample_questions = [
        "How do I create a POST route in FastAPI?",
        "How to declare path parameters?",
        "What is dependency injection in FastAPI?",
    ]

    for q in sample_questions:
        print(f"\nQuestion : {q}")
        try:
            hits = retrieve(q, top_k=3)
            for hit in hits:
                print(f"  [{hit['score']:.3f}] {hit['source']}")
                print(f"     {hit['text'][:360]}...")
        except RetrievalError as e:
            print(f"  Erreur : {e}")
