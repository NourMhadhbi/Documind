# ADR : Pourquoi ces métriques et ces seuils

- Statut : brouillon
- Date : AAAA-MM-JJ

## Contexte
Ce qu'on veut surveiller (disponibilité, latence, qualité) et pourquoi.

## Mesures de base (à remplir avec TES chiffres)
| Indicateur | Valeur mesurée | Comment (script, nombre de requêtes) |
|---|---|---|
| Latence p95 de génération | | |
| Taux de refus (questions normales) | | |
| Taux de refus (pièges) | | |
| Distance médiane (normales / pièges) | | |

## Seuils d'alerte retenus
| Alerte | Seuil | Justification (lien avec la mesure de base) | Action quand elle se déclenche |
|---|---|---|---|
| HighErrorRate | | | |
| HighRefusalRate | | | |
| HighGenerationLatencyP95 | | | |

## Choix de vie privée
Le texte des questions n'est pas journalisé : seulement langue, longueur, distances, refus, latence.

## Conséquences et limites
Volume de trafic faible, référence = baseline du Sprint 5, à réexaminer quand le volume augmente.
