# 🔍 Retrieval — `src/retrieval.py`

> **Le retrieval, c'est quoi ?** L'étape qui transforme une question en langage naturel en une liste de passages de documentation pertinents, en comparant mathématiquement le "sens" de la question à celui de chaque chunk déjà indexé. C'est la moitié "R" de RAG (Retrieval-Augmented Generation) — la génération (Sprint 4) s'appuiera sur ce que cette brique retourne.

## Où ce module se situe dans l'architecture

```
Question utilisateur
        ↓
   retrieve()  ← CE MODULE
        ↓
   Encodage (même modèle qu'à l'indexation, Sprint 2)
        ↓
   Recherche dans l'index (via VectorStore, Sprint 2)
        ↓
   Liste de chunks pertinents + sources + scores
        ↓
   [Sprint 4] Génération LLM à partir de ces chunks
```

Ce module ne fait **aucun nouveau travail d'infrastructure** — il réutilise entièrement ce que le Sprint 2 a construit (`config.py`, `vector_store.py`) et se contente d'orchestrer : encoder la question, interroger l'index, retourner les résultats.

## Pourquoi un pattern singleton (`_get_model()` / `_get_store()`)

Le modèle `all-MiniLM-L6-v2` pèse environ 90 Mo et prend du temps à charger en mémoire. Si `retrieve()` le rechargeait à chaque appel, chaque question poserait une latence de plusieurs centaines de millisecondes rien que pour le chargement — inacceptable pour un système conversationnel où l'utilisateur attend une réponse rapide.

La solution : deux variables de module (`_model`, `_store`) servent de cache. Au premier appel de `retrieve()`, le modèle et le store sont chargés et stockés ; tous les appels suivants réutilisent ces mêmes instances. C'est vérifié explicitement par le test `test_model_is_loaded_only_once_across_multiple_calls` — pas juste supposé, **prouvé**.

```python
_model: Optional[SentenceTransformer] = None  # cache, vide au démarrage


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:  # chargé seulement si pas déjà en cache
        _model = SentenceTransformer(...)
    return _model
```

## Pourquoi `normalize_embeddings=True` est non-négociable ici

L'ADR 002 (Sprint 2) documente que l'index utilise la métrique **cosinus**, valide uniquement sur des vecteurs normalisés (norme = 1). Si la question était encodée sans cette normalisation :
- **Aucune erreur ne serait levée** — le code tournerait sans planter
- Mais les résultats de recherche seraient **subtilement faux**, le classement de pertinence serait corrompu silencieusement

C'est le type de bug le plus dangereux : invisible à l'exécution, visible seulement dans la qualité dégradée des réponses. C'est pourquoi `test_retrieve_normalizes_embeddings` vérifie explicitement cette condition — une régression future sur ce point serait immédiatement détectée par la suite de tests, avant de corrompre la qualité en production.

## La hiérarchie d'erreurs : pourquoi `RetrievalError` et pas `VectorStoreError`

```python
try:
    results = store.query(...)
except VectorStoreError as exc:
    raise RetrievalError(f"Échec de la recherche dans l'index : {exc}") from exc
```

Le Sprint 2 a défini `VectorStoreError` comme l'erreur du vector store. Le Sprint 3 la **traduit** en `RetrievalError` plutôt que de la laisser remonter telle quelle. Raison : la couche au-dessus (l'API du Sprint 6) ne doit avoir besoin de connaître **qu'un seul type d'erreur** pour tout ce qui touche à la recherche de documents — elle ne doit pas se soucier de savoir si c'est ChromaDB, FAISS ou Qdrant qui tourne dessous. C'est la même logique d'abstraction que `get_vector_store()` au Sprint 2, appliquée une couche plus haut.

## Validation défensive des entrées

```python
if not question or not question.strip():
    raise RetrievalError("La question ne peut pas être vide")
if top_k <= 0:
    raise RetrievalError("top_k doit être strictement positif")
```

Deux vérifications systématiques avant tout traitement coûteux (encodage, recherche) :
- Une question vide ou uniquement composée d'espaces produirait un embedding sans signification exploitable
- Un `top_k` nul ou négatif n'a aucun sens pour une recherche de "k plus proches voisins"

Principe appliqué : valider les entrées **avant** d'engager du travail coûteux, pas après avoir découvert l'erreur au milieu du traitement.

## Pourquoi une liste vide n'est pas une erreur

```python
if not results:
    logger.warning("Aucun résultat trouvé pour la question : '%s'", question)
return results
```

Si l'index ne contient rien de pertinent pour une question (hors sujet par rapport au corpus), ce n'est **pas un bug** — c'est un résultat légitime que l'API devra gérer différemment (par exemple répondre "Je n'ai pas trouvé d'information sur ce sujet dans la documentation"). On logge un avertissement pour garder une trace, sans lever d'exception qui interromprait le flux normalement.

## Comment tester ce module

**Tests unitaires** (rapides, aucune dépendance à un vrai modèle ou un vrai index) :
```bash
pytest tests/test_retrieval.py -v
```
10 tests couvrant : validation des entrées, comportement nominal, gestion des 2 types d'erreurs possibles, et la garantie du pattern singleton.

**Test manuel de bout en bout** (nécessite l'index réel du Sprint 2 déjà construit) :
```bash
python -m src.retrieval
```
Pose 3 questions réelles et affiche les chunks retrouvés avec leur score et leur source — permet de vérifier à l'œil que la pertinence est cohérente avec les résultats du benchmark (Recall@k=1.00, voir ADR 002).

## Limites connues de ce module, assumées pour ce sprint

- **Pas de cache de requêtes répétées** : poser deux fois la même question relance deux fois l'encodage et la recherche. Un cache de requêtes serait pertinent en production à forte charge, mais prématuré ici (Sprint 12 - observabilité/robustesse avancée, si le besoin se confirme).
- **Pas de retry automatique** en cas d'échec transitoire du vector store : à ajouter si le Sprint 11 (robustesse production) l'identifie comme nécessaire.
- **`top_k` n'a pas de borne supérieure** : rien n'empêche de demander `top_k=10000`. Pas un risque réel avec ChromaDB local, mais à surveiller si le backend change.

## Lien avec le Sprint 4 (génération)

Ce module retourne des dicts `{"text": ..., "source": ..., "score": ...}` — c'est exactement le format que `generation.py` (Sprint 4) consommera pour construire le prompt envoyé au LLM, avec citation des sources. Aucune transformation supplémentaire ne devrait être nécessaire entre les deux.