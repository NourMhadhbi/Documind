"""
Tests unitaires du Sprint 2.

Pourquoi des mocks : un test unitaire doit être rapide et reproductible.
Charger un vrai modèle sentence-transformers (~90 Mo) et une vraie base
ChromaDB à chaque exécution de pytest ralentirait toute ta suite de tests
et introduirait de la non-déterminisme. On teste ici la LOGIQUE de notre
code (validation, gestion d'erreurs), pas le comportement interne des
librairies tierces (déjà testées par leurs propres mainteneurs).
"""

import json
import pytest
from pathlib import Path

from src.embeddings import load_chunks, ChunksValidationError
from src.vector_store import ChromaVectorStore, VectorStoreError


# ---------------------------------------------------------------
# Tests de load_chunks() — validation du schéma d'entrée
# ---------------------------------------------------------------

def test_load_chunks_valid_file(tmp_path):
    """Un fichier bien formé doit être chargé sans erreur."""
    valid_data = [
        {"chunk_id": "a_0", "source": "a.md", "text": "Hello world"},
        {"chunk_id": "a_1", "source": "a.md", "text": "Second chunk"},
    ]
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(valid_data), encoding="utf-8")

    result = load_chunks(str(chunks_file))
    assert len(result) == 2
    assert result[0]["chunk_id"] == "a_0"


def test_load_chunks_missing_file():
    """Un chemin inexistant doit lever une erreur explicite, pas un
    FileNotFoundError brut incompréhensible pour l'appelant."""
    with pytest.raises(ChunksValidationError, match="Fichier introuvable"):
        load_chunks("data/processed/does_not_exist.json")


def test_load_chunks_empty_list(tmp_path):
    """Une liste vide est un cas invalide (rien à indexer)."""
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ChunksValidationError, match="liste non vide"):
        load_chunks(str(chunks_file))


def test_load_chunks_missing_required_field(tmp_path):
    """Un chunk sans le champ 'source' doit être détecté et rejeté
    explicitement, pas planter plus loin dans le pipeline avec un
    KeyError obscur au moment de l'indexation."""
    invalid_data = [{"chunk_id": "a_0", "text": "Hello"}]  # 'source' manquant
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(invalid_data), encoding="utf-8")

    with pytest.raises(ChunksValidationError, match="champs manquants"):
        load_chunks(str(chunks_file))


# ---------------------------------------------------------------
# Tests de ChromaVectorStore — validation défensive
# ---------------------------------------------------------------

def test_vector_store_add_mismatched_lengths(tmp_path):
    """add() doit rejeter des listes de longueurs incohérentes plutôt
    que d'insérer des données corrompues silencieusement."""
    store = ChromaVectorStore(str(tmp_path / "index"), "test_collection")

    with pytest.raises(VectorStoreError, match="incohérentes"):
        store.add(
            ids=["1", "2"],
            texts=["text1"],  # une seule entrée au lieu de deux
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"source": "a.md"}, {"source": "b.md"}],
        )


def test_vector_store_add_and_count(tmp_path):
    """Un cycle add() -> count() basique doit refléter le nombre réel
    de documents insérés."""
    store = ChromaVectorStore(str(tmp_path / "index"), "test_collection")
    store.add(
        ids=["1", "2"],
        texts=["text1", "text2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
    )
    assert store.count() == 2


def test_vector_store_query_invalid_top_k(tmp_path):
    """top_k négatif ou nul doit être rejeté explicitement."""
    store = ChromaVectorStore(str(tmp_path / "index"), "test_collection")
    with pytest.raises(VectorStoreError, match="top_k"):
        store.query(query_embedding=[0.1, 0.2], top_k=0)