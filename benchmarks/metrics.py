"""
Métriques d'évaluation pures (bibliothèque standard seulement) : aucun modèle,
aucun accès réseau. Partagées par les scripts d'évaluation.
"""

import re
from typing import Dict, List, Optional


def rank_of_expected(sources: List[str], expected: List[str]) -> Optional[int]:
    """Rang (à partir de 1) du premier résultat dont la source est attendue, ou None."""
    for rank, source in enumerate(sources, start=1):
        if source in expected:
            return rank
    return None


def reciprocal_rank(rank: Optional[int], k: int) -> float:
    """1/rang si la bonne source est dans le top-k, sinon 0."""
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def aggregate_retrieval(rows: List[Dict], k: int) -> Dict:
    """Recall@k et MRR@k par langue et au total, sur les lignes NORMALES
    (les pièges n'ont pas de source attendue, donc pas de rappel)."""
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        if row.get("trap") or row.get("error"):
            continue
        buckets.setdefault(row["lang"], []).append(row)
        buckets.setdefault("overall", []).append(row)

    summary = {}
    for lang, items in buckets.items():
        hits = sum(1 for r in items if r["rank"] is not None and r["rank"] <= k)
        summary[lang] = {
            "n": len(items),
            "recall": hits / len(items),
            "mrr": sum(reciprocal_rank(r["rank"], k) for r in items) / len(items),
        }
    return summary


# Marqueurs de refus (heuristique transparente, à compléter si besoin).
_REFUSAL_MARKERS = (
    # français
    "ne sais pas",
    "n'ai pas trouvé",
    "pas trouvé",
    "pas d'information",
    "aucune information",
    "ne contient pas",
    "pas mentionn",
    "impossible de répondre",
    # anglais
    "don't know",
    "do not know",
    "could not find",
    "couldn't find",
    "no information",
    "not mentioned",
    "does not contain",
    "cannot answer",
    "can't answer",
    # arabe
    "لم أجد",
    "لا أعرف",
    "لا توجد معلومات",
    "غير مذكور",
    "لا يحتوي",
    "لا أستطيع",
)


def looks_like_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


_MD_REFERENCE = re.compile(r"[\w./\-]+\.md\b")


def cited_markdown_files(answer: str) -> List[str]:
    """Noms de fichiers .md cités dans le TEXTE d'une réponse, sans doublons."""
    seen: List[str] = []
    for match in _MD_REFERENCE.findall(answer or ""):
        name = match[2:] if match.startswith("./") else match
        if name not in seen:
            seen.append(name)
    return seen


def invalid_citations(answer: str, sources: List[str]) -> List[str]:
    """Fichiers cités par le modèle qui NE correspondent à aucune source réellement
    fournie. Une citation est valide si elle est égale à une source ou en est la fin
    (`path-params.md` pour `tutorial/path-params.md`). Un chemin comme
    `docs/ru/docs/tutorial/path-params.md` est invalide : c'est une invention."""
    invalid = []
    for cited in cited_markdown_files(answer):
        if not any(source == cited or source.endswith("/" + cited) for source in sources):
            invalid.append(cited)
    return invalid
