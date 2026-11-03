# ADR 003 — Choix du modèle LLM pour la génération

**Statut** : 🟡 Provisoire (à confirmer après benchmark élargi et run de référence Claude)
**Date** : à compléter
**Contexte Sprint** : Sprint 4

## Contexte

La génération RAG doit produire une réponse fidèle aux extraits récupérés, citer ses sources, et refuser de répondre quand le contexte ne contient pas l'information. Deux familles de solutions : un modèle cloud (API Anthropic, payant) et un petit modèle local servi par Ollama (gratuit, privé, hors-ligne).

L'architecture (`src/llm_provider.py`) permet de changer de fournisseur par simple configuration : ce choix est **réversible**.

## Options considérées

| Option | Avantages | Inconvénients | Évalué ? |
|---|---|---|---|
| Claude Sonnet 5 (cloud) | Qualité et respect des consignes attendus plus élevés | Coût par token, dépendance réseau, données envoyées à un tiers | ❌ Non (coût du crédit API) |
| Qwen2.5 3B (Ollama) | Gratuit, privé, hors-ligne | Petit modèle : risque d'inventer | ✅ |
| Llama 3.2 3B (Ollama) | Gratuit, privé, hors-ligne | Même limite de taille | ✅ |

Claude n'a **pas** été évalué. Les conclusions ci-dessous comparent deux modèles locaux entre eux ; elles ne disent rien de leur distance par rapport à un modèle cloud.

## Méthode

`python -m benchmarks.compare_llms --candidates ollama:qwen2.5:3b ollama:llama3.2:3b`

Même retrieval, même prompt, même corpus ; seul le modèle change. 5 questions du benchmark du Sprint 2 + 1 question piège hors corpus (recette de gâteau). Un seul passage par question. Réponses complètes : `benchmarks/results/llm_comparison.json`.

## Résultats mesurés

| Indicateur | Qwen2.5 3B | Llama 3.2 3B |
|---|---|---|
| Questions traitées / erreurs | 6 / 0 | 5 / **1 (timeout 360 s)** |
| Latence moyenne | 89,9 s | 85,9 s |
| Citation d'une source attendue | 0 / 5 | 2 / 4 évaluées |
| Refus sur la question piège | ✅ | ✅ |

## Lecture qualitative des réponses

| Question | Qwen2.5 3B | Llama 3.2 3B |
|---|---|---|
| Route POST | Confus : mélange GET et POST, `Request` utilisé sans import | Correct (`@app.post("/items/")`) |
| Path parameters | Enum et `response_model` incohérents | `Path(...)` correct, mais `Annotated[Path(...), UserID]` invalide ; cite `path-params.md` |
| Validation Pydantic | Modèle correct, puis exemple hors sujet (headers) | Modèle correct, mais invente `.extra()` et utilise `Depends()` à tort |
| Dependency injection | Invente un décorateur `@app.dependencies` | Fidèle au contexte, cite `dependencies/index.md` |
| Code de statut HTTP | Répond via `Response(status_code=...)` | Timeout |
| Question piège | Refus concis | Refus correct, plus bavard |

Langue : Qwen répond en anglais ou en français selon la question, Llama en français. Le prompt ne précise pas la langue de réponse.

## Limites de cette expérience

- **5 questions, un seul passage** : une différence de 1 ou 2 réponses n'est pas statistiquement significative.
- **Latence anormale** : 50 à 140 s par réponse est très lent pour un 3B, ce qui suggère un problème d'infrastructure (mémoire, deux conteneurs Ollama) plutôt qu'une propriété des modèles. Le timeout de Llama est donc probablement non représentatif. Les latences ne sont pas comparables.
- **Citation** : l'indicateur exige le nom exact du fichier dans le texte ; c'est une heuristique.
- **Fidélité non mesurée** : on voit les sources, pas les chunks envoyés. Une bonne réponse peut venir de la mémoire du modèle plutôt que du contexte (mesure prévue au Sprint 5, RAGAS).
- Les deux modèles **inventent des fonctions** qui n'existent pas dans FastAPI : aucun n'est fiable pour de la documentation technique sans vérification.
- Quantification des deux modèles non vérifiée (`ollama show <modèle>`).
- **Langues non évaluées** : le benchmark est en anglais et en français. Le chatbot doit répondre aussi en arabe ; or, d'après leurs fiches, Llama 3.2 ne liste pas l'arabe parmi ses langues officielles, alors que Qwen2.5 le liste. Cet avantage éventuel de Qwen n'est pas mesuré.

## Observations complémentaires (test de génération en trois langues)

Modèle Ollama configuré, retrieval sur l'index à ancres conservées (ADR 004).

- **La règle de langue fonctionne** : les réponses sont en anglais, en français et en arabe selon la question.
- **Question française sur les path parameters** : le retrieval n'a pas trouvé la bonne page (absente du top-5, voir `benchmarks/check_languages.py`). Le modèle a répondu qu'on ne lui avait pas fourni le fichier `tutorial/dependencies/sub-dependencies.md`, puis a décrit les path parameters, probablement depuis ses connaissances générales : cela viole la règle 2 du prompt (ne répondre qu'avec le contexte).
- **Question arabe** : réponse en arabe, mais qui cite `docs/ru/docs/tutorial/path-params.md`, un nom qui ne correspond pas au nommage des sources du corpus (`tutorial/path-params.md`). Les citations écrites par le modèle ne sont pas fiables ; seule la liste `sources` renvoyée par le code l'est.
- **Conséquence** : un mauvais retrieval produit une réponse fausse mais d'apparence assurée. Pistes (Sprint 5) : améliorer le retrieval en français et en arabe, et ajouter un garde-fou « contexte trop peu pertinent, donc pas de réponse », à calibrer sur les distances mesurées plutôt qu'à deviner.

## Décision (provisoire)

- **Modèle local de développement et de test : `llama3.2:3b`**, tant que l'arabe n'est pas évalué ; si l'arabe est une exigence, cette décision est à réviser en comparant les deux modèles sur des questions arabes. Il fait un peu mieux sur les 4 questions comparables et cite ses sources, pour un refus de la question piège équivalent. Avantage faible, à confirmer.
- **Modèle de production : non décidé.** Aucun modèle local évalué n'est assez fiable pour la production sur ce corpus, et la référence Claude manque pour savoir de combien.
- **Qwen2.5 3B écarté** pour ce projet, sauf si le benchmark élargi inverse le classement.

## Actions de suivi

- [ ] Corriger la latence (arrêter le conteneur HM-Retail pendant les tests, limiter `LLM_MAX_TOKENS`, surveiller `docker stats`)
- [ ] Ajouter au prompt : répondre dans la langue de la question, ne citer que des fonctions présentes dans les extraits
- [ ] Élargir le benchmark : au moins 15 questions, 2 passages, 3 ou 4 questions pièges
- [ ] Run de référence Claude (coût estimé de l'ordre de quelques centimes, à vérifier)
- [ ] Vérifier la quantification des deux modèles
- [ ] Évaluer les réponses en arabe et en français (`python -m benchmarks.check_languages`, puis comparer les deux modèles sur des questions arabes)
- [ ] Réviser cet ADR avec les nouveaux chiffres ; statut ✅ seulement à ce moment-là

## Conséquences

- Les réponses d'un 3B doivent être présentées comme **non vérifiées** tant que la faithfulness n'est pas mesurée (Sprint 5).
- Un usage hybride reste possible : local pour développer sans coût, cloud pour la production.
- Le choix étant réversible, il peut être révisé sans toucher à `generation.py`.