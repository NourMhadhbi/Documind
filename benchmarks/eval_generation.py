"""
Évalue la GÉNÉRATION sur le jeu de référence, SANS modèle juge : tout est déterministe.

    python -m benchmarks.eval_generation --run-name llama --limit 6 --resume --mlflow

Mesures (par langue) :
  - langue de la réponse conforme à celle de la question (prose seulement, code ignoré)
  - bonne source présente dans les sources réellement fournies au modèle
  - refus sur les questions pièges, et refus à tort sur les questions normales
  - fichiers .md cités dans le texte mais absents des sources (citations inventées)
  - latence de l'appel LLM

Chaque question prend 1 à 2 minutes sur une machine lente : le fichier de résultats est
réécrit après CHAQUE question, et --resume reprend là où l'exécution s'est arrêtée (les
questions en erreur sont réessayées). Le fichier contient aussi les contextes et réponses :
c'est l'entrée de l'étape RAGAS (benchmarks.ragas_eval).
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from benchmarks.eval_retrieval import load_questions
from benchmarks.io_utils import read_json, write_json_atomic
from benchmarks.metrics import invalid_citations, looks_like_refusal
from src.language import answer_language_matches

DEFAULT_GOLDEN = "tests/golden_qa.json"
RESULTS_DIR = "benchmarks/results"


def select_questions(questions: List[Dict], limit: Optional[int] = None, langs: Optional[List[str]] = None,
                     ids: Optional[List[str]] = None, include_traps: bool = True) -> List[Dict]:
    selected = [q for q in questions
                if (not langs or q["lang"] in langs)
                and (not ids or q["id"] in ids)
                and (include_traps or not q["trap"])]
    return selected[:limit] if limit else selected


def evaluate_one(question: Dict, generate_fn: Callable[[str], Dict]) -> Dict:
    """Génère la réponse à une question et calcule ses indicateurs. Une panne est
    enregistrée dans la ligne (champ `error`), jamais propagée."""
    row = {"id": question["id"], "lang": question["lang"], "topic": question["topic"],
           "trap": question["trap"], "question": question["question"],
           "reference": question.get("reference"), "expected_sources": question["expected_sources"],
           "error": None}
    try:
        result = generate_fn(question["question"])
    except Exception as exc:  # noqa: BLE001 : on veut continuer l'évaluation
        row["error"] = str(exc)
        return row

    answer = result["answer"]
    sources = result.get("sources", [])
    row.update({
        "answer": answer,
        "sources": sources,
        "contexts": result.get("contexts", []),
        "refused": bool(result.get("refused")) or looks_like_refusal(answer),
        "refusal_reason": result.get("refusal_reason"),
        "latency_seconds": result.get("latency_seconds", 0.0),
        "min_distance": result.get("min_distance"),
        "llm_provider": result.get("llm_provider"),
        "llm_model": result.get("llm_model"),
        "language_ok": answer_language_matches(answer, question["lang"]),
        "source_ok": None if question["trap"] else bool(set(question["expected_sources"]) & set(sources)),
        "invalid_citations": invalid_citations(answer, sources),
    })
    return row


def _rate(values: List[Optional[bool]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return sum(1 for v in known if v) / len(known) if known else None


def summarize(rows: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = {"overall": rows}
    for row in rows:
        groups.setdefault(row["lang"], []).append(row)

    summary = {}
    for name, items in groups.items():
        ok = [r for r in items if not r.get("error")]
        normal = [r for r in ok if not r["trap"]]
        traps = [r for r in ok if r["trap"]]
        answered = [r for r in normal if not r["refused"]]
        latencies = [r["latency_seconds"] for r in ok if r.get("latency_seconds")]
        summary[name] = {
            "n": len(items),
            "n_errors": len(items) - len(ok),
            "language_ok_rate": _rate([r["language_ok"] for r in ok]),
            "source_ok_rate": _rate([r["source_ok"] for r in normal]),
            "trap_refusal_rate": _rate([r["refused"] for r in traps]),
            "wrong_refusal_rate": _rate([r["refused"] for r in normal]),
            "invalid_citation_rate": (sum(1 for r in answered if r["invalid_citations"]) / len(answered)
                                      if answered else None),
            "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        }
    return summary


def run_evaluation(questions: List[Dict], generate_fn: Callable[[str], Dict], output_path: str,
                   meta: Dict, resume: bool = False, progress: Callable[[str], None] = print) -> List[Dict]:
    done: Dict[str, Dict] = {}
    if resume and Path(output_path).exists():
        done = {r["id"]: r for r in read_json(output_path).get("rows", []) if not r.get("error")}

    rows: List[Dict] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        if question["id"] in done:
            rows.append(done[question["id"]])
            progress(f"[{index}/{total}] {question['id']} : déjà fait")
            continue
        started = time.perf_counter()
        row = evaluate_one(question, generate_fn)
        rows.append(row)
        status = f"erreur : {row['error']}" if row["error"] else ("refus" if row["refused"] else "répondu")
        progress(f"[{index}/{total}] {question['id']} : {status} ({time.perf_counter() - started:.0f} s)")
        write_json_atomic(output_path, {"meta": meta, "rows": rows, "summary": summarize(rows)})

    write_json_atomic(output_path, {"meta": meta, "rows": rows, "summary": summarize(rows)})
    return rows


def tracking_metrics(summary: Dict[str, Dict]) -> Dict[str, Optional[float]]:
    names = {"language_ok_rate": "language_ok", "source_ok_rate": "source_ok",
             "trap_refusal_rate": "trap_refusal", "wrong_refusal_rate": "wrong_refusal",
             "invalid_citation_rate": "invalid_citation", "avg_latency_s": "latency_s", "n_errors": "n_errors"}
    metrics: Dict[str, Optional[float]] = {}
    for group, values in summary.items():
        for key, label in names.items():
            metrics[f"{label}_{group}"] = values[key]
    return metrics


def _fmt(value: Optional[float], percent: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if percent else f"{value:.0f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--limit", type=int, default=None, help="nombre maximal de questions")
    parser.add_argument("--lang", default=None, help="langues séparées par des virgules (ex: en,fr)")
    parser.add_argument("--ids", default=None, help="identifiants séparés par des virgules")
    parser.add_argument("--no-traps", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    from config import settings
    from src.generation import generate_answer

    questions = select_questions(
        load_questions(args.golden), args.limit,
        args.lang.split(",") if args.lang else None,
        args.ids.split(",") if args.ids else None,
        include_traps=not args.no_traps)
    if not questions:
        print("Aucune question sélectionnée.")
        return

    meta = {
        "run_name": args.run_name, "golden": args.golden, "n_questions": len(questions),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.ollama_model_name if settings.llm_provider == "ollama" else settings.llm_model_name,
        "embedding_model": settings.embedding_model_name, "index_path": settings.index_path,
        "max_distance": settings.max_distance, "retrieval_top_k": settings.retrieval_top_k,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output = args.output or f"{RESULTS_DIR}/generation_{args.run_name}.json"
    print(f"{len(questions)} questions | {meta['llm_provider']}:{meta['llm_model']} | garde-fou : {meta['max_distance']}")
    rows = run_evaluation(questions, generate_answer, output, meta, resume=args.resume)
    summary = summarize(rows)

    print(f"\n{'Groupe':<9}{'n':<4}{'Langue':<8}{'Source':<8}{'Pièges':<8}{'Refus':<7}{'Cit.inv.':<9}{'Latence':<8}{'Err'}")
    for group in sorted(summary, key=lambda name: (name == "overall", name)):
        v = summary[group]
        print(f"{group:<9}{v['n']:<4}{_fmt(v['language_ok_rate']):<8}{_fmt(v['source_ok_rate']):<8}"
              f"{_fmt(v['trap_refusal_rate']):<8}{_fmt(v['wrong_refusal_rate']):<7}"
              f"{_fmt(v['invalid_citation_rate']):<9}{_fmt(v['avg_latency_s'], False) + ' s':<8}{v['n_errors']}")
    print("\nLangue = réponse dans la langue de la question | Source = bonne page parmi les sources fournies |"
          "\nPièges = refus des questions hors corpus | Refus = refus à tort | Cit.inv. = citations de fichiers inventées")
    print(f"Résultats : {output}")

    if args.mlflow:
        from benchmarks.tracking import log_run
        log_run("generation_eval", args.run_name, meta, tracking_metrics(summary), artifacts=[output])
        print("Run journalisé dans MLflow.")


if __name__ == "__main__":
    main()
