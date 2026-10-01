# ADR 002 — Choix de la base vectorielle

**Statut** : ✅ Validé
**Date** : Sprint 2
**Contexte Sprint** : Sprint 2

## Contexte

Le système RAG a besoin de stocker les embeddings des chunks de documentation et de pouvoir retrouver rapidement les plus pertinents pour une question donnée. Trois bases candidates : ChromaDB, FAISS, Qdrant.

## Options considérées

| Option | Avantages | Inconvénients |
|---|---|---|
| ChromaDB | Setup simple, persistance intégrée, métadonnées natives | Overhead supérieur à FAISS sur de très gros volumes |
| FAISS | Le plus rapide en recherche pure | Pas de métadonnées natives (mapping externe nécessaire) |
| Qdrant | Bon compromis perf/fonctionnalités, filtres avancés | Nécessite un service (même en mode `:memory:` pour les tests) |

## Méthode de décision

Benchmark exécuté via `benchmarks/benchmark_vectordb.py` sur le corpus réel (155 fichiers FastAPI, 1281 chunks après nettoyage), mesurant temps d'indexation, latence de requête (top-5), et qualité de retrieval (Recall@k, MRR) sur 5 questions avec vérité terrain vérifiée manuellement.

**Incident méthodologique corrigé en cours de route** : le premier essai comparait les 3 bases avec des métriques de distance différentes par défaut (euclidienne pour FAISS, cosinus pour Qdrant, non précisée pour ChromaDB), rendant la comparaison de qualité invalide. Corrigé en imposant la similarité cosinus partout et en normalisant systématiquement les embeddings — validé par l'obtention de scores identiques sur les 3 bases.

## Historique des expériences

| Run | chunk_size/overlap | Code conservé | Métrique | Recall@k | MRR |
|---|---|---|---|---|---|
| Baseline | 500/50 | Non | incohérente (bug) | 0.60 | 0.50 |
| Exp1 | 800/100 | Non | incohérente (bug) | 0.60–0.80 | mixte |
| Exp2 | 800/100 | Non | cosinus harmonisée | 0.80 | 0.42 |
| **Exp3 (retenue)** | 800/100 | **Oui** | cosinus harmonisée | **1.00** | **0.73** |

Chaque run ne change qu'une seule variable par rapport au précédent (étude d'ablation), pour isoler l'effet de chaque facteur :
- `chunk_size` 500→800 : améliore le Recall@k (chunks plus grands = moins de perte de contexte)
- Conservation des blocs de code : résout le dernier échec (les pages riches en exemples de code, comme `tutorial/first-steps.md` et `tutorial/body.md`, perdaient leur contenu le plus pertinent quand le code était supprimé au nettoyage)
- Harmonisation de la métrique : ne change pas la qualité réelle, mais rend la comparaison entre bases valide

## Résultats finaux (configuration retenue)

| Base | Index (s) | Latence (ms) | Recall@k | MRR |
|---|---|---|---|---|
| ChromaDB | 0.300 | 3.57 | 1.00 | 0.73 |
| FAISS | 0.001 | 0.29 | 1.00 | 0.73 |
| Qdrant | 0.473 | 4.66 | 1.00 | 0.73 |

Runs consultables : https://dagshub.com/mhadhbinouur/documind.mlflow/#/experiments/0

## Décision

**ChromaDB est retenu.**

À qualité de retrieval strictement identique entre les 3 bases (Recall@k=1.00, MRR=0.73 partout — confirmation que la méthodologie est fiable), le choix se fait sur des critères pratiques :
- Persistance et métadonnées natives, contrairement à FAISS qui exige un mapping externe position→source
- Pas de service à maintenir, contrairement à Qdrant en usage réel (le mode `:memory:` du benchmark ne reflète pas un déploiement de production)
- Latence de 3.57ms largement suffisante pour un usage conversationnel (un humain ne perçoit pas une différence de quelques millisecondes)

FAISS, bien que le plus rapide, apporterait une complexité d'intégration non justifiée par un gain de performance imperceptible à l'usage.

## Conséquences

- Le MRR de 0.73 (bon document pas toujours en 1ère position) est un point de vigilance, pas un défaut bloquant — à réévaluer au Sprint 5 avec RAGAS sur un jeu de questions plus large (20-30 questions).
- Ce choix pourra être révisé si le corpus grandit significativement (passage à un volume nécessitant une base distribuée comme Milvus).
- `KEEP_CODE_BLOCKS=True` est désormais la configuration de production — implique que `generation.py` (Sprint 4) recevra des chunks contenant du code, à prendre en compte dans la construction du prompt.