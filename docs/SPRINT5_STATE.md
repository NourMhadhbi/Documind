# État du projet DocuMind au Sprint 5



## Projet
Chatbot RAG sur la doc FastAPI (155 fichiers .md), pipeline MLOps/LLMOps. Tout en local (Ollama), **sans Claude API**.
Stack : Python, sentence-transformers (all-MiniLM-L6-v2), ChromaDB (cosinus, vecteurs normalisés), FastAPI (Sprint 6), DVC + DagsHub, MLflow (DagsHub), pytest, Docker.

## Sprints terminés (1 à 4)
- Ingestion : chunks 800/100, **code conservé**, **ancres `{ #...}` conservées** dans l'index (ADR 004, mesuré), retirées du prompt.
- Index : 1283 chunks (`python -m src.ingestion`, supprimer `data/index`, `python -m src.embeddings`).
- Retrieval (`src/retrieval.py`), génération (`src/generation.py`), abstraction LLM (`src/llm_provider.py`).
- Ollama : conteneur `documind-ollama`, port **11435**, modèle `llama3.2:3b` (ADR 003 **provisoire**, Claude non évalué, 5 questions seulement).
- ADR 002 (ChromaDB), 003 (LLM), 004 (nettoyage).

## Sprint 5 : déjà écrit et testé (150 passed, 1 skipped chez moi, tests avec mocks)
`src/language.py` (détection en/fr/ar, messages de refus), garde-fou `MAX_DISTANCE` dans `generation.py`
(désactivé par défaut), `benchmarks/` : `metrics.py`, `tracking.py` (MLflow), `io_utils.py`, `validate_golden.py`,
`eval_retrieval.py`, `calibrate_guard.py`, `eval_generation.py` (avec `--resume`), `tests/golden_qa.json` (18 questions).

## Reste à faire au Sprint 5
1. Elle doit écrire **12 questions de plus** (3 sujets x 3 langues + 3 pièges) en LISANT la doc (jamais d'après le retrieval).
2. `benchmarks/ragas_eval.py` + `requirements-eval.txt` (environnement séparé), ADR 005 (embeddings multilingues),
   006 (garde-fou), 007 (RAGAS), `README_EVALUATION.md`.
3. Essai embeddings multilingue (second index `data/index_multilingual`, vérifier `max_seq_length` du modèle vs chunks de 800 caractères).
4. Calibrer le garde-fou avec `calibrate_guard`. Essai RAGAS sur 2 questions, puis 10.

## Faits vérifiés sur RAGAS (ne pas redécouvrir)
- `pip install ragas` seul **casse l'import** (langchain-community trop récent). Pins qui fonctionnent (testés en Python 3.12) :
  `ragas==0.4.3`, `langchain-community<0.4`, `langchain<1`, `langchain-core<1`, `langchain-openai<1`.
- Avec Ollama : `AsyncOpenAI(api_key="ollama", base_url="http://localhost:11435/v1")` + `llm_factory(model, provider="openai", client=client)`.
  Un client **synchrone échoue** (TypeError). Il faut `await metric.ascore(...)`.
- Signatures : `Faithfulness.ascore(user_input, response, retrieved_contexts)` ; `ContextPrecisionWithoutReference.ascore(user_input, response, retrieved_contexts)` ;
  `ContextPrecisionWithReference.ascore(user_input, reference, retrieved_contexts)` ; `ContextRecall.ascore(user_input, retrieved_contexts, reference)`. Résultat : `.value`.
- Jamais testé contre un vrai juge Ollama. Prévoir un essai sur 2 questions (durée, sorties lisibles) avant tout le reste.

## Résultats mesurés à retenir
Retrieval (15 questions, index final) : anglais Recall@5 1.00, français 0.60, arabe 0.20. Le frein en arabe est le modèle d'embeddings anglais.
Génération : ~50 à 140 s par réponse (latence anormale, non diagnostiquée). Le modèle 3B invente parfois des sources et répond hors contexte.


