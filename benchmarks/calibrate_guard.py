"""
Aide à calibrer le garde-fou de pertinence (MAX_DISTANCE) à partir d'un fichier
produit par benchmarks.eval_retrieval.

    python -m benchmarks.calibrate_guard benchmarks/results/retrieval_baseline.json

Principe. Le garde-fou refuse de répondre quand la distance du MEILLEUR chunk dépasse un
seuil. Une question « doit être refusée » si c'est un piège hors corpus, ou si le
retrieval a raté la bonne page (répondre depuis un mauvais contexte donne une réponse
fausse mais assurée). Les autres « doivent être traitées ». Le script cherche le seuil
qui sépare le mieux ces deux groupes, et dit honnêtement s'ils se séparent mal.

Il n'écrit rien : tu décides, puis tu mets MAX_DISTANCE dans .env.
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

from benchmarks.io_utils import read_json


def labelled_distances(rows: List[Dict]) -> Tuple[List[float], List[float]]:
    """(distances des questions à traiter, distances des questions à refuser)."""
    answerable, unanswerable = [], []
    for row in rows:
        if row.get("error") or row.get("top1_distance") is None:
            continue
        should_refuse = bool(row["trap"]) or row.get("rank") is None
        (unanswerable if should_refuse else answerable).append(row["top1_distance"])
    return answerable, unanswerable


def separation_auc(answerable: List[float], unanswerable: List[float]) -> Optional[float]:
    """Probabilité qu'une question à refuser ait une distance plus grande qu'une question
    à traiter (1.0 = séparation parfaite, 0.5 = aucune)."""
    if not answerable or not unanswerable:
        return None
    wins = 0.0
    for bad in unanswerable:
        for good in answerable:
            wins += 1.0 if bad > good else 0.5 if bad == good else 0.0
    return wins / (len(answerable) * len(unanswerable))


def evaluate_threshold(
    threshold: float, answerable: List[float], unanswerable: List[float]
) -> Dict:
    kept = sum(1 for d in answerable if d <= threshold) / len(answerable)
    refused = sum(1 for d in unanswerable if d > threshold) / len(unanswerable)
    return {
        "threshold": threshold,
        "answerable_kept": kept,
        "unanswerable_refused": refused,
        "balanced": (kept + refused) / 2,
    }


def candidate_thresholds(values: List[float]) -> List[float]:
    ordered = sorted(set(values))
    mids = [(a + b) / 2 for a, b in zip(ordered, ordered[1:], strict=False)]
    return [ordered[0] - 1e-6] + mids + [ordered[-1] + 1e-6]


def best_threshold(answerable: List[float], unanswerable: List[float]) -> Dict:
    """Seuil de meilleure exactitude équilibrée. À égalité, on préfère ne PAS refuser de
    bonnes questions, puis le seuil le plus grand."""
    candidates = [
        evaluate_threshold(t, answerable, unanswerable)
        for t in candidate_thresholds(answerable + unanswerable)
    ]
    return max(
        candidates,
        key=lambda c: (round(c["balanced"], 9), c["answerable_kept"], c["threshold"]),
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", help="fichier benchmarks/results/retrieval_*.json")
    args = parser.parse_args()

    data = read_json(args.results)
    meta = data.get("meta", {})
    answerable, unanswerable = labelled_distances(data["rows"])
    print(f"Modèle : {meta.get('embedding_model')} | index : {meta.get('index_path')}")
    print(
        f"Questions à traiter : {len(answerable)} | à refuser (pièges ou page ratée) : {len(unanswerable)}"
    )

    if not answerable or not unanswerable:
        print("\nImpossible de calibrer : il faut au moins une question de chaque groupe.")
        return

    for label, values in (("à traiter", answerable), ("à refuser", unanswerable)):
        print(
            f"  distances {label} : min {min(values):.3f} | moyenne {sum(values) / len(values):.3f} | max {max(values):.3f}"
        )

    auc = separation_auc(answerable, unanswerable)
    best = best_threshold(answerable, unanswerable)
    print(f"\nSéparation (AUC) : {auc:.2f}   (1.00 parfaite, 0.50 aucune)")
    print(
        f"Meilleur seuil : {best['threshold']:.3f} -> bonnes questions conservées "
        f"{best['answerable_kept']:.0%}, refus corrects {best['unanswerable_refused']:.0%}"
    )

    print("\nSeuils candidats (autour du meilleur) :")
    print(f"{'Seuil':<9}{'Conservées':<12}{'Refus corrects':<16}{'Équilibre':<10}")
    all_thresholds = candidate_thresholds(answerable + unanswerable)
    index = min(
        range(len(all_thresholds)),
        key=lambda i: abs(all_thresholds[i] - best["threshold"]),
    )
    for t in all_thresholds[max(0, index - 4) : index + 5]:
        r = evaluate_threshold(t, answerable, unanswerable)
        marker = "  <- retenu" if abs(t - best["threshold"]) < 1e-12 else ""
        print(
            f"{t:<9.3f}{r['answerable_kept']:<12.0%}{r['unanswerable_refused']:<16.0%}{r['balanced']:<10.2f}{marker}"
        )

    print("\nAVERTISSEMENTS")
    if len(answerable) < 10 or len(unanswerable) < 5:
        print(
            "  - Très peu de données : le seuil est fragile. Ajoute des questions au jeu de référence."
        )
    if auc is not None and auc < 0.7:
        print(
            "  - Séparation faible (AUC < 0.70) : ce garde-fou refuserait de bonnes questions ou en laisserait"
            " passer de mauvaises. Ne l'active pas sans amélioration du retrieval."
        )
    print(
        "  - Le seuil est calibré sur ces questions : vérifie-le sur des questions NON utilisées ici."
    )
    print("  - Recalibre dès que le modèle d'embeddings ou l'index change.")
    print(f"\nSi tu l'adoptes, ajoute dans .env :  MAX_DISTANCE={best['threshold']:.2f}")


if __name__ == "__main__":
    main()
