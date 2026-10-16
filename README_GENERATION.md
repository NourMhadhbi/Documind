# 🤖 Génération — `src/generation.py` + `src/llm_provider.py`

> Ce module complète le pipeline RAG : il prend les chunks retournés par `retrieve()` (Sprint 3), construit un prompt, interroge un LLM et retourne une réponse **sourcée**. Le fournisseur LLM (cloud Anthropic ou local Ollama) se choisit par configuration, sans toucher au code.

## Sommaire

1. [Architecture](#architecture)
2. [Décisions de conception](#décisions-de-conception)
3. [Configuration](#configuration)
4. [Tester le Sprint 4 (commandes)](#tester-le-sprint-4-commandes)
5. [Comparer les modèles](#comparer-les-modèles-adr-003)
6. [Dépannage](#dépannage)
7. [Limites connues](#limites-connues)

## Architecture

```
Question
   ↓
retrieve() [Sprint 3]
   ↓
_deduplicate_by_source()   1 chunk max par fichier (configurable)
   ↓
build_prompt()             contexte numéroté + question
   ↓
get_llm_provider(settings) ──► AnthropicProvider  (API cloud)
                           └─► OllamaProvider     (modèle local)
   ↓
{"answer", "sources", "chunks_used", "llm_provider", "llm_model", "latency_seconds"}
```

Fichiers du sprint :

```
documind/
├── config.py                        paramètres (remplace l'ancien)
├── .env.example                     modèle de .env (remplace l'ancien)
├── src/
│   ├── ingestion.py                 nettoyage corrigé + chunking (remplace l'ancien)
│   ├── llm_provider.py              abstraction + Anthropic + Ollama
│   └── generation.py                pipeline retrieval + génération
├── tests/
│   ├── test_generation.py           21 tests
│   ├── test_llm_provider.py         12 tests
│   ├── test_config.py               2 tests
│   ├── test_compare_llms.py         9 tests
│   ├── test_check_languages.py      5 tests
│   ├── test_clean_text.py           18 tests
│   └── test_compare_cleaning.py     13 tests
├── benchmarks/
│   ├── __init__.py                  fichier vide (voir ci-dessous)
│   ├── compare_llms.py              comparaison des modèles
│   ├── check_languages.py           retrieval en français / anglais / arabe
│   └── compare_cleaning.py          ancien vs nouveau nettoyage (en mémoire)
├── docs/adr/003-llm-choice.md       choix du LLM (provisoire)
├── docs/adr/004-text-cleaning.md    nettoyage : conserver les ancres de titre
└── README_GENERATION.md             ce fichier
```

`__init__.py` (vide) déclare un dossier comme package Python, ce qui permet `python -m benchmarks.compare_llms` et `from benchmarks... import ...`. Il n'a pas besoin de contenu.

## Décisions de conception

| Décision | Pourquoi |
|---|---|
| Abstraction `LLMProvider` | Même pattern que `VectorStore` (Sprint 2) : changer de modèle = changer la config, pas le code. Rend la comparaison possible. |
| Erreurs `LLMProviderError` → `GenerationError` | Les couches supérieures (API, Sprint 6) ne connaissent qu'un seul type d'erreur. |
| Déduplication par source avant le prompt | Observé au Sprint 3 : une page dense occupait tout le top-k. Configurable (`max_chunks_per_source`). |
| Prompt système séparé du contexte | Mesure anti prompt-injection de base : un chunk contenant « ignore les règles » reste une donnée. Traitement complet au Sprint 10. |
| Aucun appel LLM si aucun chunk | Évite un coût inutile et une réponse tirée des connaissances générales du modèle. |
| `temperature = 0.0` | Réponses factuelles et reproductibles. `None` = paramètre non envoyé (certains modèles récents le refusent). |
| Timeout Ollama configurable | Le 1er appel charge le modèle en RAM. Défaut 120 s dans `config.py` ; relevé à 360 s pendant le benchmark car la machine était lente. |
| `latency_seconds` et `llm_model` dans le résultat | Nécessaires pour comparer les modèles et pour MLflow (Sprint 5). |

## Langue des réponses (français, anglais, arabe)

Le chatbot répond dans la langue de la question. Deux mécanismes, dans `src/generation.py` :

- **Règle 6 du prompt système** : « réponds dans la même langue que la question, même si le contexte est en anglais ; garde en anglais le code, les noms de fonctions et de fichiers ».
- **Rappel explicite pour l'arabe** : `detect_language_hint()` repère l'écriture arabe (plages Unicode, sans bibliothèque ni aléatoire) et ajoute `LANGUE DE RÉPONSE : arabe` à la question. L'écriture non latine, avec un contexte entièrement en anglais, est le cas où un petit modèle risque le plus de répondre dans la mauvaise langue. Le français et l'anglais n'ont pas besoin de ce rappel.

**Deux limites à connaître, qui ne dépendent pas du prompt :**

1. **Le retrieval.** Le corpus est en anglais et l'index utilise `all-MiniLM-L6-v2`, un modèle surtout entraîné sur l'anglais. Une question en français ou en arabe peut retrouver de moins bons passages, et le LLM ne peut pas répondre correctement à partir de mauvais extraits. Mesure-le avec `python -m benchmarks.check_languages`. Si l'arabe ou le français sont nettement moins bons, la correction est un modèle d'embeddings multilingue : changer `EMBEDDING_MODEL_NAME`, supprimer `data/index`, reconstruire avec `python -m src.embeddings`, puis refaire le benchmark. Le modèle d'index et celui des questions doivent toujours être identiques.
2. **Le modèle de génération.** D'après leurs fiches, Qwen2.5 liste l'arabe parmi ses plus de 29 langues, alors que Llama 3.2 3B liste officiellement l'anglais, l'allemand, le français, l'italien, le portugais, l'hindi, l'espagnol et le thaï, sans l'arabe. Aucune mesure fiable de leur qualité en arabe sur ton cas n'existe : teste-le.

## Configuration

Les valeurs par défaut sont dans `config.py`. Tes valeurs réelles vont dans `.env` (jamais commité), qui l'emporte sur `config.py`. La clé Anthropic ne va **que** dans `.env`.

```bash
LLM_PROVIDER=ollama                      # ou anthropic
ANTHROPIC_API_KEY=                       # seulement si anthropic
LLM_MODEL_NAME=claude-sonnet-5

OLLAMA_BASE_URL=http://localhost:11435   # conteneur documind-ollama
OLLAMA_MODEL_NAME=llama3.2:3b            # nom EXACT affiché par `ollama list`
OLLAMA_TIMEOUT_SECONDS=360
```

Le port de `OLLAMA_BASE_URL` doit correspondre au côté hôte du compose (`"11435:11434"` pour `documind-ollama`). Le nom du modèle doit être exact, tag compris, sinon Ollama répond « model not found ».

`.env.example` contient exactement les variables de `Settings` ; `tests/test_config.py` échoue s'il dérive de `config.py`. Les noms de modèles Anthropic évoluent : vérifie la liste officielle avant de figer un choix.

## Tester le Sprint 4 (commandes)

Toutes les commandes se lancent **depuis la racine du projet** (`documind/`), PowerShell, environnement virtuel activé.

### 0. Préparation

```powershell
.venv\Scripts\Activate.ps1
python --version                         # doit afficher 3.13.x
pip install -r requirements.txt
pip install pytest
New-Item -ItemType File benchmarks/__init__.py -ErrorAction SilentlyContinue
```

Vérifie que ces fichiers existent : `pytest.ini`, `src/__init__.py`, `benchmarks/__init__.py`, `.env`.

### 1. Tests unitaires (aucun appel réseau, aucun coût)

```powershell
pytest tests -v
```

Résultat attendu, par fichier :

| Fichier | Tests | Ce qu'il vérifie |
|---|---|---|
| `test_generation.py` | 21 | déduplication, prompt, pas d'appel LLM sans chunk, propagation d'erreurs, instructions séparées du contexte, langue de réponse |
| `test_llm_provider.py` | 12 | Anthropic et Ollama (mockés), température optionnelle, timeout configurable, factory |
| `test_config.py` | 2 | `.env.example` synchronisé avec `config.py` |
| `test_compare_llms.py` | 9 | fonctions pures du script de comparaison |
| `test_check_languages.py` | 5 | fonctions pures du contrôle multilingue |
| `test_clean_text.py` | 18 | nettoyage Markdown : underscores, ancres, code, liens |
| `test_compare_cleaning.py` | 13 | classement, cache d'embeddings, ablation règle par règle |
| `test_retrieval.py` | 10 | Sprint 3 (non-régression) |
| `test_embeddings.py` | 7 | Sprint 2 (non-régression) |

Dernière ligne attendue : **`97 passed`** (90 pour les Sprints 3 et 4, plus les 7 du Sprint 2), si `tests/` ne contient que ces six fichiers. Si ton total diffère, vérifie qu'aucun fichier de test n'est manquant ou en trop ; ce qui compte est l'absence d'échec (`failed`).

Variantes utiles :

```powershell
pytest tests/test_generation.py -v           # un seul fichier
pytest tests -k "ollama" -v                  # seulement les tests dont le nom contient "ollama"
pytest tests -x                              # s'arrête au premier échec
pytest tests/test_config.py -v               # vérifie .env.example vs config.py
```

**Limite** : ces tests mockent le LLM et le retrieval. Ils prouvent que la logique est correcte, pas que Ollama ou Anthropic répondent. C'est le rôle des étapes 2 à 4.

### 2. Vérifier la configuration réelle (sans afficher la clé)

```powershell
python -c "from config import settings; print(settings.llm_provider, settings.ollama_base_url, settings.ollama_model_name, settings.ollama_timeout_seconds, 'clé:', 'présente' if settings.anthropic_api_key else 'absente')"
```

Une erreur `extra_forbidden` signifie qu'une variable de `.env` n'existe pas dans `config.py` : supprime-la de `.env` ou ajoute le champ.

### 3. Vérifier Ollama (conteneur Docker)

```powershell
docker ps                                                     # documind-ollama doit être "Up", port 0.0.0.0:11435->11434
docker exec documind-ollama ollama list                       # le modèle choisi doit apparaître
curl.exe http://localhost:11435/api/tags                      # doit renvoyer du JSON
docker exec documind-ollama ollama run llama3.2:3b "Réponds juste: OK"
```

Si le conteneur n'existe pas : `docker compose -f docker-compose.ollama.yml up -d`, puis `docker exec documind-ollama ollama pull llama3.2:3b`.

### 4. Tests réels, du plus petit au plus complet

```powershell
# a) Le fournisseur LLM seul, sans retrieval
python -c "from config import settings; from src.llm_provider import get_llm_provider; print(get_llm_provider(settings).complete('Réponds en un mot.', 'Dis OK'))"

# b) Le retrieval seul (Sprint 3), nécessite l'index du Sprint 2
python -m src.retrieval

# c) Le pipeline complet : retrieval + génération
python -m src.generation

# d) Le retrieval en trois langues (aucun LLM appelé)
python -m benchmarks.check_languages
```

### Mesurer l'effet du nettoyage (ancien contre corrigé)

`check_languages` seul ne peut pas dire si le nettoyage corrigé améliore le retrieval : il faudrait aussi l'ancien index, que tu as supprimé. `compare_cleaning` reconstruit **les deux versions en mémoire**, avec le même modèle, le même découpage (800/100) et les mêmes 15 questions, sans toucher à ton index ChromaDB :

```powershell
python -m benchmarks.compare_cleaning
```

Il affiche le rang de la bonne page pour chaque question, avec un verdict (`mieux`, `moins bien`, `=`), puis le Recall@5 et le MRR par langue.

Deux options pour comprendre un écart :

```powershell
python -m benchmarks.compare_cleaning --explain            # texte des chunks des questions dégradées
python -m benchmarks.compare_cleaning --ablate --explain   # + teste chaque règle séparément
```

Les variantes sont : A ancien nettoyage, B corrigé avec ancres supprimées (essai précédent), C **corrigé avec ancres conservées (configuration retenue, ADR 004)**. `--ablate` ajoute deux variantes de C, chacune avec **une seule règle désactivée** (directives `{* *}`, marqueurs `///`) : une variable à la fois, comme pour l'étude d'ablation du Sprint 2. Les embeddings sont mis en cache par texte de chunk, donc une variante ne ré-encode que les chunks qui diffèrent. Avec 5 questions par langue, une différence d'une seule question n'est pas significative : décide sur l'ensemble, pas sur un cas isolé.

### Comparer deux modèles d'embeddings sans toucher à l'index validé

Pour tester un modèle multilingue, construis un **second index** dans un autre dossier : l'index du Sprint 2 reste intact. Les variables d'environnement l'emportent sur `.env`.

```powershell
# Mesure de référence (index actuel)
python -m benchmarks.check_languages

# Vérifie la longueur maximale d'entrée du modèle candidat (en tokens)
python -c "from sentence_transformers import SentenceTransformer as S; print(S('paraphrase-multilingual-MiniLM-L12-v2').max_seq_length)"

# Second index avec le modèle candidat
$env:EMBEDDING_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"
$env:INDEX_PATH="data/index_multilingual"
python -m src.embeddings
python -m benchmarks.check_languages
Remove-Item Env:EMBEDDING_MODEL_NAME, Env:INDEX_PATH
```

Ajoute `data/index_*/` à `.gitignore` avant de commiter, sinon l'index binaire serait suivi par Git. Si `max_seq_length` est inférieur à la longueur des chunks (800 caractères, environ 200 tokens), la fin de chaque chunk n'est pas encodée : il faudra réduire `chunk_size` et refaire l'étude d'ablation du Sprint 2.

Pour vérifier la langue des réponses, pose la même question dans les trois langues :

```powershell
python -c "from src.generation import generate_answer as g; [print(q, '->', g(q)['answer'][:300], chr(10)) for q in ['How to declare path parameters?', 'Comment déclarer des paramètres de chemin ?', 'كيف أعرّف معاملات المسار؟']]"
```

Lis les réponses : le script ne juge pas la langue à ta place. Vérifie surtout que la réponse arabe est en arabe et que le code et les noms de fichiers restent en anglais.

Si (a) échoue, le problème vient d'Ollama ou de la config. Si (a) passe et (c) échoue, il vient du retrieval ou de l'index. Le premier appel d'un modèle est lent (chargement en RAM).

### 5. Fournisseur Anthropic (jamais validé en conditions réelles)

Les tests ne font que mocker Anthropic. Pour le valider, mets `ANTHROPIC_API_KEY` dans `.env` (compte API avec crédit, distinct de l'abonnement claude.ai), puis :

```powershell
$env:LLM_PROVIDER="anthropic"; python -m src.generation; Remove-Item Env:LLM_PROVIDER
```

Un test coûte quelques centimes de dollar. Une erreur « model not found » signifie que `LLM_MODEL_NAME` doit être corrigé d'après la documentation Anthropic.

### 6. Comparaison des modèles

Voir la section suivante.

### 7. Vérifications Git avant de commiter

```powershell
git status                               # .env et data/ ne doivent PAS apparaître
git check-ignore -v .env                 # doit citer la règle de .gitignore
git ls-files .env                        # ne doit rien afficher
git add src/ tests/ benchmarks/ docs/adr/003-llm-choice.md README_GENERATION.md config.py .env.example requirements.txt
git commit -m "Sprint 4: génération RAG, abstraction LLMProvider, comparaison Ollama, ADR 003 provisoire"
git push origin main
```

Vérifie que `benchmarks/results/llm_comparison.json` fait partie du commit : c'est la preuve de l'ADR.

## Comparer les modèles (ADR 003)

```powershell
python -m benchmarks.compare_llms --candidates ollama:qwen2.5:3b ollama:llama3.2:3b
```

Ajoute `anthropic:claude-sonnet-5` pour la référence cloud. Chaque candidat s'écrit `fournisseur:modèle`, nom exact. Le script pose 5 questions du benchmark du Sprint 2 et une question piège hors corpus, puis écrit les réponses complètes dans `benchmarks/results/llm_comparison.json`. Lis-les : les indicateurs (latence, citation, refus) sont des heuristiques.

Pour garder une trace complète du terminal : `python -m benchmarks.compare_llms ... *> resultat_llm.txt`.

Résultat actuel : voir `docs/adr/003-llm-choice.md` (décision provisoire en faveur de `llama3.2:3b` pour le développement local ; modèle de production non décidé, Claude non évalué).

## Dépannage

| Symptôme | Cause probable | Correction |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Lancé hors de la racine, ou `pytest.ini` / `src/__init__.py` manquant | Lancer depuis `documind/`, vérifier ces fichiers |
| `No module named 'benchmarks'` | `benchmarks/__init__.py` manquant | `New-Item -ItemType File benchmarks/__init__.py` |
| `extra_forbidden` au chargement | Variable de `.env` absente de `config.py` | Supprimer la variable ou ajouter le champ |
| `Impossible de joindre Ollama ... ollama serve` | Conteneur arrêté ou mauvais port | `docker ps`, vérifier `OLLAMA_BASE_URL` (11435) |
| `model 'xxx' not found` | Nom de modèle inexact | Copier le nom affiché par `ollama list` |
| `Ollama n'a pas répondu dans les N s` | Machine lente ou RAM saturée | Arrêter les autres conteneurs, baisser `LLM_MAX_TOKENS`, augmenter `OLLAMA_TIMEOUT_SECONDS` |
| `Internal Server Error ... dockerDesktopLinuxEngine` | Docker Desktop mal démarré | Quitter puis rouvrir Docker Desktop |
| `ANTHROPIC_API_KEY manquant` | Clé absente de `.env` | L'ajouter dans `.env` uniquement |

## Limites connues

- Les **citations écrites par le LLM dans sa réponse ne sont pas vérifiées** (observé : une source inexistante `docs/ru/...`). Seule la liste `sources` renvoyée par le code est fiable.
- Quand le retrieval se trompe (fréquent en français et en arabe), le modèle peut répondre quand même, avec aplomb, à partir de ses connaissances générales. Il n'existe pas encore de garde-fou de pertinence (Sprint 5).
- Le retrieval multilingue n'est **pas validé** : l'index est en anglais (voir « Langue des réponses »). Les tests vérifient les instructions de langue envoyées au LLM, pas la qualité des réponses en arabe.
- Les tests unitaires ne contactent ni Ollama ni Anthropic ; le fournisseur Anthropic n'a **jamais été appelé pour de vrai**.
- Les petits modèles locaux (3B) inventent des fonctions qui n'existent pas dans FastAPI : leurs réponses ne sont pas vérifiées tant que la fidélité n'est pas mesurée (Sprint 5, RAGAS).
- Latence mesurée anormalement élevée (50 à 140 s par réponse) : à diagnostiquer avant toute conclusion sur la vitesse des modèles.
- Pas de retry automatique (Sprint 11), pas de streaming (Sprint 6), protection anti prompt-injection de base seulement (Sprint 10), pas de suivi de coût (Sprint 12).
- Les indicateurs du script de comparaison sont des heuristiques, pas une évaluation RAGAS (Sprint 5).