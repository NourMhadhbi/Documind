# 📚 DocuMind — Assistant RAG avec pipeline MLOps/LLMOps

> Assistant conversationnel intelligent qui répond à des questions sur la documentation technique de FastAPI en citant systématiquement ses sources, construit avec un pipeline MLOps/LLMOps professionnel de bout en bout.

---

## Sommaire

- [Le problème](#-le-problème)
- [La solution](#-la-solution)
- [Architecture](#-architecture)
- [Stack technique](#-stack-technique)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Données](#-données)
- [Évaluation & qualité](#-évaluation--qualité)
- [Méthodologie de développement](#-méthodologie-de-développement)
- [Feuille de route (Sprints)](#-feuille-de-route-sprints)
- [Limites connues](#-limites-connues)
- [Compétences mises en pratique](#-compétences-mises-en-pratique)
- [Auteur](#-auteur)

---

## 🎯 Le problème

Chercher une information précise dans une documentation technique volumineuse (des centaines de pages) prend du temps. Poser la question directement à un modèle de langage (LLM) est plus rapide, mais pose deux problèmes :

- **Hallucination** : le LLM peut inventer une réponse plausible mais fausse si l'information ne fait pas partie de ses connaissances.
- **Absence de traçabilité** : impossible de savoir d'où vient une réponse générée, donc impossible de vérifier sa fiabilité.

## 💡 La solution

**DocuMind** utilise le **RAG (Retrieval-Augmented Generation)** : avant de répondre, le système recherche les passages les plus pertinents dans la vraie documentation, puis demande au LLM de générer sa réponse **à partir de ces passages uniquement**, en citant les sources utilisées.

```
Question → Recherche dans la doc réelle → Réponse générée à partir des passages trouvés + sources citées
```

Le corpus utilisé est la documentation officielle de [FastAPI](https://fastapi.tiangolo.com/) (155 fichiers Markdown), mais l'architecture est généralisable à n'importe quel corpus documentaire (documentation interne d'entreprise, contrats, base de connaissances support...).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SOURCE : Documentation FastAPI (155 fichiers .md)           │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  1. INGESTION & CHUNKING          (src/ingestion.py)          │
│     Lecture récursive des .md → découpage en chunks           │
│     (chunk_size / chunk_overlap paramétrables)                │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. EMBEDDINGS                    (src/embeddings.py)         │
│     sentence-transformers → vecteurs sémantiques               │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. INDEX VECTORIEL                (ChromaDB)                  │
│     Stockage + recherche par similarité                        │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
                Question utilisateur → embedding
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. RETRIEVAL (top-k)              (src/retrieval.py)          │
│     Récupère les chunks les plus pertinents                    │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. GÉNÉRATION LLM                 (src/generation.py)         │
│     Prompt = question + chunks → réponse + sources citées      │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. API                            (api/main.py — FastAPI)     │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
     Tests               Docker              CI/CD
  (pytest + RAGAS)                      (GitHub Actions)
        └───────────────────┼───────────────────┘
                            ↓
                       DÉPLOIEMENT
                            ↓
                       MONITORING
          (latence, coût tokens, feedback utilisateur)

┌─────────────────────────────────────────────────────────────┐
│  COUCHE TRANSVERSALE : DAGSHUB                                  │
│  • DVC   → versionne les documents, chunks et l'index vectoriel │
│  • MLflow → journalise chaque expérience (chunk_size, modèle    │
│             d'embedding, prompt) et ses scores de qualité        │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Stack technique

| Domaine | Outil | Rôle |
|---|---|---|
| Langage | Python 3.11+ | Développement du pipeline |
| Chunking | Python natif | Découpage des documents |
| Embeddings | sentence-transformers | Représentation vectorielle du texte |
| Base vectorielle | ChromaDB | Stockage et recherche de similarité |
| Génération | API LLM (Anthropic/OpenAI) ou Ollama | Génération de réponses |
| API | FastAPI | Exposition du service |
| Évaluation | RAGAS | Mesure objective de la qualité RAG |
| Versioning données | DVC + DagsHub | Traçabilité données/index |
| Experiment tracking | MLflow (via DagsHub) | Comparaison des configurations |
| Tests | pytest | Tests unitaires et de non-régression |
| Conteneurisation | Docker | Reproductibilité |
| CI/CD | GitHub Actions | Automatisation tests/build |

## 📂 Structure du projet

```
documind/
│
├── data/
│   ├── raw/                # documents source .md — versionnés via DVC
│   ├── processed/          # chunks.json généré par ingestion.py
│   └── index/               # index vectoriel ChromaDB — versionné via DVC
│
├── src/
│   ├── ingestion.py          # chargement + découpage des documents
│   ├── embeddings.py         # génération des embeddings + index
│   ├── retrieval.py          # recherche top-k de chunks pertinents
│   ├── generation.py         # appel LLM avec contexte récupéré
│   └── evaluate.py           # évaluation RAGAS + logging MLflow
│
├── api/
│   └── main.py                # API FastAPI (endpoint /chat)
│
├── tests/
│   ├── golden_qa.json         # jeu de questions/réponses de référence
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_regression.py     # non-régression qualité sur golden_qa
│
├── .github/workflows/
│   └── ci.yml                  # pipeline CI/CD GitHub Actions
│
├── Dockerfile
├── requirements.txt
├── .env.example                # modèle des variables d'environnement
├── .gitignore
├── dvc.yaml                     # pipeline DVC (à ajouter en Sprint 2+)
└── README.md
```

## ⚙️ Installation

```bash
# 1. Cloner le dépôt (hébergé sur DagsHub)
git clone https://dagshub.com/<TON_USERNAME>/documind.git
cd documind

# 2. Créer et activer un environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Récupérer les données versionnées (documents, chunks, index)
dvc pull

# 5. Configurer les variables d'environnement
cp .env.example .env
# puis éditer .env avec tes propres clés (DagsHub, LLM, etc.)
```

## ▶️ Utilisation

```bash
# Étape 1 : générer les chunks à partir de la documentation brute
python src/ingestion.py

# Étape 2 : construire l'index vectoriel
python src/embeddings.py

# Étape 3 : lancer l'API
uvicorn api.main:app --reload

# Étape 4 : interroger le système
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment créer une route POST en FastAPI ?"}'
```

Réponse attendue (exemple) :
```json
{
  "answer": "Pour créer une route POST en FastAPI, on utilise le décorateur @app.post(\"/chemin\")...",
  "sources": ["tutorial/first-steps.md", "tutorial/body.md"]
}
```

## 📊 Données

- **Source** : documentation officielle FastAPI, récupérée depuis le dépôt GitHub officiel (`docs/en/docs`)
- **Volume** : 155 fichiers Markdown
- **Format des chunks** : `{"chunk_id": "...", "source": "chemin/relatif/fichier.md", "text": "..."}`
- **Versioning** : les documents bruts, les chunks et l'index vectoriel sont versionnés avec **DVC**, stockés sur **DagsHub**, et référencés dans Git via des fichiers `.dvc` (pointeurs légers)

## ✅ Évaluation & qualité

Contrairement à un modèle de classification classique, un système RAG n'a pas de métrique unique du type "accuracy". La qualité est évaluée sur plusieurs dimensions distinctes, mesurées avec le framework **RAGAS** sur un jeu de **questions/réponses de référence** (`golden_qa.json`) construit et validé manuellement :

| Métrique | Ce qu'elle mesure |
|---|---|
| **Faithfulness** | La réponse générée est-elle fidèle aux documents récupérés (pas d'invention) ? |
| **Context precision / recall** | Les chunks récupérés sont-ils les bons ? |
| **Answer relevancy** | La réponse répond-elle réellement à la question posée ? |

Chaque expérience (variation de `chunk_size`, du modèle d'embedding, ou du prompt système) est journalisée dans **MLflow** (hébergé par DagsHub) afin de comparer objectivement les configurations et choisir la meilleure.

Un test de **non-régression** (`test_regression.py`) rejoue automatiquement le golden dataset à chaque modification du pipeline pour garantir qu'une amélioration côté code ne dégrade pas silencieusement la qualité des réponses.

## 🔄 Méthodologie de développement

Projet développé en méthodologie **Scrum**, avec des sprints hebdomadaires ayant chacun un objectif clair (*Sprint Goal*) et des critères de fin explicites (*Definition of Done*).

## 🗺️ Feuille de route (Sprints)

| Sprint | Objectif | Statut |
|---|---|---|
| 1 | Setup DagsHub + Ingestion & Chunking | ✅ terminé |
| 2 | Embeddings & Index vectoriel | ✅ terminé  |
| 3 | Retrieval | ⬜ à venir |
| 4 | Génération (pipeline RAG complet) | ⬜ à venir |
| 5 | Évaluation (RAGAS + MLflow) | ⬜ à venir |
| 6 | API FastAPI | ⬜ à venir |
| 7 | Tests de non-régression + Docker | ⬜ à venir |
| 8 | CI/CD (GitHub Actions) | ⬜ à venir |
| 9 | Monitoring + finalisation | ⬜ à venir |



## 🎓 Compétences mises en pratique

`RAG` · `LLMOps` · `Versioning de données (DVC)` · `Experiment tracking (MLflow)` · `Évaluation de systèmes génératifs (RAGAS)` · `Développement d'API (FastAPI)` · `Tests de non-régression` · `Conteneurisation (Docker)` · `CI/CD (GitHub Actions)` · `Méthodologie Scrum`

