# 📊 Benchmark des bases vectorielles — `benchmark_vectordb.py`

## À quoi sert ce script

Comparer ChromaDB, FAISS et Qdrant sur le **vrai corpus** du projet (documentation FastAPI), selon 2 dimensions :
1. **Performance brute** : vitesse d'indexation et de requête
2. **Qualité du retrieval** : la base retrouve-t-elle vraiment les bons passages ?

## Méthodologie (après correction d'un biais important)

Un premier essai de comparaison a révélé que les 3 bases utilisaient des **métriques de distance différentes** par défaut (ChromaDB : métrique non précisée ; FAISS : distance euclidienne ; Qdrant : cosinus), ce qui rendait toute comparaison de qualité invalide — chaque base pouvait classer les mêmes vecteurs différemment pour des raisons purement mathématiques, indépendamment de ses mérites réels.

**Correction appliquée** : toutes les bases utilisent désormais la similarité cosinus, et tous les embeddings (chunks et questions) sont normalisés (`normalize_embeddings=True`) avant comparaison — condition mathématique requise pour que la distance cosinus soit valide. Cette harmonisation a été validée en vérifiant que les 3 bases retournent des scores de qualité identiques sur un même jeu de données.

## Les métriques utilisées

| Métrique | Question à laquelle elle répond |
|---|---|
| **Temps d'indexation** | Combien de temps pour insérer tous les chunks ? |
| **Latence de requête** | Combien de temps pour répondre à une recherche ? |
| **Recall@k** | Le bon document est-il présent dans les k résultats retournés ? (0 à 1) |
| **MRR (Mean Reciprocal Rank)** | À quel rang apparaît le bon document ? (proche de 1.0 = en tête) |

## La vérité terrain (`BENCHMARK_QUERIES`)

5 questions représentatives, avec leur(s) source(s) attendue(s) vérifiée(s) manuellement dans `data/raw/` :

```python
{
    "query": "How do I create a POST route in FastAPI?",
    "expected_sources": ["tutorial/first-steps.md"],
}
```

## Historique des expériences (étude d'ablation)

Chaque ligne change **une seule variable** par rapport à la précédente, pour isoler son effet réel :

| Run | chunk_size / overlap | Blocs de code conservés | Métrique | Recall@k | MRR |
|---|---|---|---|---|---|
| Baseline | 500 / 50 | Non | ⚠️ incohérente entre bases | 0.60 | 0.50 |
| Exp1 | 800 / 100 | Non | ⚠️ incohérente entre bases | 0.60–0.80 (selon la base) | mixte |
| Exp2 | 800 / 100 | Non | ✅ cosinus harmonisée | 0.80 | 0.42 |
| **Exp3 (retenue)** | 800 / 100 | **Oui** | ✅ cosinus harmonisée | **1.00** | **0.73** |

**Résultat final (ChromaDB, FAISS, Qdrant — identique sur les 3)** :

```
Base        Index (s)   Latence (ms)   Recall@k    MRR
ChromaDB    0.300       3.57           1.00        0.73
FAISS       0.001       0.29           1.00        0.73
Qdrant      0.473       4.66           1.00        0.73
```

## Comment l'exécuter

```bash
pip install chromadb faiss-cpu qdrant-client sentence-transformers mlflow python-dotenv

python -m src.ingestion        # régénère chunks.json avec la config courante
python benchmarks/benchmark_vectordb.py
```

## Comment interpréter le résultat pour choisir

Recall@k et MRR étant **strictement identiques** sur les 3 bases (preuve que la méthodologie est fiable), la décision ne se fait pas sur la qualité mais sur des critères pratiques : simplicité d'intégration, métadonnées natives, absence de service à maintenir. Décision documentée dans `docs/adr/002-vector-database-choice.md`.

## Enseignements méthodologiques (au-delà du résultat final)

- **Isoler une seule variable à la fois** a permis de distinguer l'effet du `chunk_size`, de la conservation du code, et de la métrique — un changement groupé aurait rendu impossible de savoir lequel de ces facteurs expliquait l'amélioration.
- **Un Recall@k identique entre plusieurs outils est un signal de fiabilité méthodologique**, pas une coïncidence suspecte : sur les mêmes vecteurs et la même métrique, une recherche exacte doit mathématiquement retourner les mêmes voisins.
- **MRR = 0.73 malgré Recall@k = 1.00** signale une marge de progression : le bon document est toujours trouvé, mais pas toujours classé en tête. Point à réévaluer au Sprint 5 avec un jeu de questions plus large (RAGAS).

## Limites de ce benchmark

- Résultat spécifique à ce corpus (documentation FastAPI) — pas forcément généralisable à un autre type de contenu.