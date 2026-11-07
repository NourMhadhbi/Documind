"""
Valide le jeu de questions de référence (tests/golden_qa.json).

Un jeu d'évaluation faux fausse toutes les mesures qui en dépendent : on
vérifie donc sa structure, que chaque source attendue existe vraiment dans le
corpus, et que les trois langues sont couvertes de façon équilibrée.

    python -m benchmarks.validate_golden
    python -m benchmarks.validate_golden --min 30

IMPORTANT : les sources attendues se vérifient en LISANT les pages de la
documentation. Ne les déduis pas des résultats du retrieval : tu mesurerais
alors le système avec ses propres réponses, et il aurait toujours raison.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

LANGS = {"en", "fr", "ar"}
REQUIRED_FIELDS = {"id", "topic", "lang", "question", "expected_sources", "trap"}
DEFAULT_PATH = "tests/golden_qa.json"


def load_golden(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: Dict) -> List[str]:
    """Retourne la liste des erreurs de structure (liste vide = valide)."""
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        return ["'questions' doit être une liste non vide"]

    errors: List[str] = []
    seen_ids = set()
    for index, q in enumerate(questions):
        if not isinstance(q, dict):
            errors.append(f"#{index}: doit être un objet")
            continue
        label = q.get("id", f"#{index}")
        missing = REQUIRED_FIELDS - q.keys()
        if missing:
            errors.append(f"{label}: champs manquants {sorted(missing)}")
            continue
        if q["id"] in seen_ids:
            errors.append(f"{label}: id en double")
        seen_ids.add(q["id"])
        if q["lang"] not in LANGS:
            errors.append(f"{label}: langue '{q['lang']}' invalide (attendu : {sorted(LANGS)})")
        if not isinstance(q["question"], str) or not q["question"].strip():
            errors.append(f"{label}: question vide")
        if not isinstance(q["trap"], bool):
            errors.append(f"{label}: 'trap' doit être true ou false")
        if not isinstance(q["expected_sources"], list):
            errors.append(f"{label}: 'expected_sources' doit être une liste")
        elif q["trap"] is True and q["expected_sources"]:
            errors.append(f"{label}: une question piège ne doit avoir aucune source attendue")
        elif q["trap"] is False and not q["expected_sources"]:
            errors.append(f"{label}: une question normale doit avoir au moins une source attendue")
    return errors


def check_sources_exist(data: Dict, data_dir: str) -> List[str]:
    """Erreurs pour chaque source attendue qui n'existe pas dans le corpus."""
    errors = []
    for q in data.get("questions", []):
        for source in q.get("expected_sources", []):
            if not (Path(data_dir) / source).is_file():
                errors.append(f"{q.get('id')}: source introuvable dans {data_dir} : {source}")
    return errors


def coverage_report(data: Dict) -> Dict:
    """Compte par langue et repère les sujets qui n'existent pas dans les 3 langues."""
    per_lang: Dict[str, int] = {lang: 0 for lang in sorted(LANGS)}
    traps_per_lang: Dict[str, int] = {lang: 0 for lang in sorted(LANGS)}
    langs_by_topic: Dict[str, set] = {}
    for q in data.get("questions", []):
        lang = q.get("lang")
        if lang not in LANGS:
            continue
        if q.get("trap"):
            traps_per_lang[lang] += 1
        else:
            per_lang[lang] += 1
            langs_by_topic.setdefault(q.get("topic", "?"), set()).add(lang)
    incomplete = {t: sorted(LANGS - langs) for t, langs in langs_by_topic.items() if langs != LANGS}
    return {
        "per_lang": per_lang,
        "traps_per_lang": traps_per_lang,
        "topics": len(langs_by_topic),
        "incomplete_topics": incomplete,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--min",
        type=int,
        default=30,
        help="nombre de questions visé (avertissement seulement)",
    )
    args = parser.parse_args()

    data = load_golden(args.path)
    errors = validate_schema(data)
    if not errors:
        if Path(args.data_dir).is_dir():
            errors += check_sources_exist(data, args.data_dir)
        else:
            print(
                f"AVERTISSEMENT : {args.data_dir} introuvable, existence des sources non vérifiée "
                "(lance `dvc pull` ou ajuste --data-dir)."
            )

    total = len(data.get("questions", []))
    report = coverage_report(data) if not errors or total else {}
    print(f"{total} questions dans {args.path}")
    if report:
        print(f"  questions normales par langue : {report['per_lang']}")
        print(f"  questions pièges par langue   : {report['traps_per_lang']}")
        for topic, missing in report["incomplete_topics"].items():
            print(f"  AVERTISSEMENT : sujet '{topic}' sans langue(s) {missing}")
    if total < args.min:
        print(
            f"  AVERTISSEMENT : {total} questions, objectif {args.min} (ajoute des sujets lus dans la doc)."
        )

    if errors:
        print("\nERREURS :")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nJeu de référence valide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
