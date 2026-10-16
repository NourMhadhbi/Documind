# ADR 004 — Nettoyage du texte : conserver les ancres de titre

**Statut** : ✅ Retenu (sur 15 questions, une exécution ; à reconfirmer avec un benchmark plus large)
**Date** : à compléter
**Contexte Sprint** : Sprint 4

## Contexte

La sortie réelle du retrieval a révélé trois défauts dans le nettoyage du Sprint 1 :
`user_id` devenait `userid`, les blocs de code étaient abîmés (indentation perdue,
fences cassées), et des éléments propres à MkDocs restaient dans le texte
(`{ #ancre }`, `{* fichier.py *}`, `/// tip`). Le nettoyage a été réécrit (le code est
mis à l'abri avant de traiter le texte courant). Question : cette réécriture
améliore-t-elle le retrieval ?

## Méthode

`python -m benchmarks.compare_cleaning --ablate --explain` : mêmes documents, même
modèle (`all-MiniLM-L6-v2`), mêmes chunks (800/100, code conservé), mêmes 15 questions
(5 sujets × anglais, français, arabe). Seule la fonction de nettoyage change. Tout est
fait en mémoire ; chaque variante désactive **une seule** règle.

Variantes : A = ancien nettoyage ; B = corrigé, ancres supprimées ; C = B sans
suppression des directives `{* *}` ; D = **B sans suppression des ancres** ;
E = B sans suppression des marqueurs `///`.

## Résultats mesurés (Recall@5 / MRR)

| Langue | A ancien | B corrigé | C sans directives | **D ancres conservées** | E sans `///` |
|---|---|---|---|---|---|
| Anglais | 1.00 / 0.58 | 0.60 / 0.38 | 0.60 / 0.35 | **1.00 / 0.70** | 0.60 / 0.41 |
| Français | 0.20 / 0.24 | 0.40 / 0.31 | 0.40 / 0.30 | **0.60 / 0.35** | 0.40 / 0.28 |
| Arabe | 0.20 / 0.04 | 0.20 / 0.07 | 0.20 / 0.07 | 0.20 / **0.22** | 0.20 / 0.07 |

Rangs de la bonne page pour les deux questions anglaises les plus touchées :

| Question | A | B | C | D | E |
|---|---|---|---|---|---|
| Route POST | 1 | 16 | >20 | 2 | 10 |
| Path parameters | 4 | >20 | 7 | 1 | 8 |

Nombre de chunks : A 1349, B 1237, C 1281, D 1283, E 1249.

## Lecture

- Supprimer les ancres est la cause de la dégradation : B, C et E perdent deux
  questions anglaises, D les retrouve.
- Garder les directives `{* *}` (C) ou les marqueurs `///` (E) ne change rien.
- Le chunk gagnant de l'ancien test commençait par
  `Step 3: create a path operation { #step-3-create-a-path-operation }` : les mots du
  titre y figuraient deux fois. Cohérent avec l'idée que l'ancre renforce le titre, mais
  **ce mécanisme n'est pas démontré**.

## Décision

Configuration retenue = variante D : toutes les corrections **sauf** la suppression des
ancres (`clean_text(..., strip_anchors=False)`, valeur par défaut). Les ancres restent dans
le texte **indexé**. Elles sont retirées au moment de construire le prompt
(`generation.build_prompt`), car un modèle les avait recopiées dans une réponse.

## Conséquences et limites

- 15 questions, une seule exécution : la direction est cohérente sur trois langues, mais
  l'écart reste indicatif. À reconfirmer avec le jeu élargi du Sprint 5.
- **L'arabe reste faible** (Recall@5 0.20, quatre questions sur cinq absentes du top-20 quelle
  que soit la variante) : le frein est le modèle d'embeddings anglais, pas le nettoyage.
  Piste suivante : un modèle multilingue sur un second index (attention à sa longueur
  maximale d'entrée par rapport aux chunks de 800 caractères).
- Piste non testée : préfixer chaque chunk par le titre de la page et de la section, pour
  porter le sujet explicitement au lieu de s'appuyer sur un effet de bord des ancres.
- Les ancres sont des identifiants en anglais : leur effet sur des questions en français ou
  en arabe n'est pas expliqué.
- Les exemples de code des pages sont stockés dans `docs_src/`, absent du corpus : seules les
  directives qui y renvoient (supprimées) étaient présentes.
- Index et chunks sont à régénérer après tout changement de nettoyage
  (`python -m src.ingestion`, suppression de `data/index`, `python -m src.embeddings`).
