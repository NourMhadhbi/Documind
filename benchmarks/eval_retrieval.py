"""
Évalue le RETRIEVAL sur le jeu de référence (tests/golden_qa.json), par langue.
Aucun LLM n'est appelé : quelques secondes par configuration.

    python -m benchmarks.eval_retrieval --run-name baseline --mlflow

Pour comparer un autre modèle d'embeddings SANS toucher à l'index validé, construis un
second index, puis lance l'évaluation avec les mêmes variables d'environnement :

    $env:EMBEDDING_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"
    $env:INDEX_PATH="data/index_multilingual"
    python -m src.embeddings
    python -m benchmarks.eval_retrieval --run-name multilingual --mlflow
    Remove-Item Env:EMBEDDING_MODEL_NAME, Env:INDEX_PATH

Le fichier de résultats contient, pour chaque question, la distance du meilleur chunk :
benchmarks.calibrate_guard s'en sert pour calibrer le garde-fou de pertinence.
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List

from benchmarks.io_utils import write_json_atomic
from benchmarks.metrics import aggregate_retrieval, rank_of_expected
from benchmarks.validate_golden import load_golden, validate_schema

DEFAULT_GOLDEN = "tests/golden_qa.json"
RESULTS_DIR = "benchmarks/results"


def load_questions(path: str) -> List[Dict]:
    data = load_golden(path)
    errors = validate_schema(data)
    if errors:
        raise ValueError("Jeu de référence invalide : " + " ; ".join(errors[:5]))
    return data["questions"]


def run_retrieval(questions: List[Dict], retrieve_fn: Callable, top_k: int) -> List[Dict]:
    """Une ligne par question. Une panne sur une question est enregistrée, pas fatale."""
    rows = []
    for q in questions:
        row = {"id": q["id"], "lang": q["lang"], "topic": q["topic"], "trap": q["trap"],
               "question": q["question"], "rank": None, "top1_distance": None,
               "sources": [], "error": None}
        try:
            chunks = retrieve_fn(q["question"], top_k)
        except Exception as exc:  # noqa: BLE001 : on veut continuer l'évaluation
            row["error"] = str(exc)
            rows.append(row)
            continue
        row["sources"] = [c["source"] for c in chunks]
        distances = [c["score"] for c in chunks if c.get("score") is not None]
        row["top1_distance"] = min(distances) if distances else None
        if not q["trap"]:
            row["rank"] = rank_of_expected(row["sources"], q["expected_sources"])
        rows.append(row)
    return rows


def tracking_metrics(summary: Dict, n_errors: int) -> Dict[str, float]:
    metrics = {"n_errors": float(n_errors)}
    for lang, values in summary.items():
        metrics[f"recall_{lang}"] = values["recall"]
        metrics[f"mrr_{lang}"] = values["mrr"]
    return metrics


def mean_distance(rows: List[Dict], trap: bool):
    values = [r["top1_distance"] for r in rows if r["trap"] == trap and r["top1_distance"] is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--note", default="", help="texte libre enregistré avec le run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mlflow", action="store_true", help="journalise le run dans MLflow/DagsHub")
    args = parser.parse_args()

    from config import settings
    from src.retrieval import retrieve

    questions = load_questions(args.golden)
    rows = run_retrieval(questions, lambda text, k: retrieve(text, top_k=k), args.top_k)
    summary = aggregate_retrieval(rows, args.top_k)
    n_errors = sum(1 for r in rows if r["error"])

    meta = {
        "run_name": args.run_name, "note": args.note, "top_k": args.top_k, "golden": args.golden,
        "embedding_model": settings.embedding_model_name, "index_path": settings.index_path,
        "collection": settings.vector_collection_name, "n_questions": len(questions),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output = args.output or f"{RESULTS_DIR}/retrieval_{args.run_name}.json"
    write_json_atomic(output, {"meta": meta, "rows": rows, "summary": summary})

    print(f"\nRetrieval | modèle : {meta['embedding_model']} | index : {meta['index_path']} | top-{args.top_k}")
    print(f"{'Langue':<10}{'n':<5}{'Recall@k':<11}{'MRR@k':<8}")
    for lang in sorted(summary, key=lambda name: (name == "overall", name)):
        v = summary[lang]
        print(f"{lang:<10}{v['n']:<5}{v['recall']:<11.2f}{v['mrr']:<8.2f}")
    misses = [r["id"] for r in rows if not r["trap"] and not r["error"] and r["rank"] is None]
    print(f"\nQuestions dont la bonne page est absente du top-{args.top_k} : {misses or 'aucune'}")
    d_normal, d_trap = mean_distance(rows, False), mean_distance(rows, True)
    if d_normal is not None and d_trap is not None:
        print(f"Distance moyenne du meilleur chunk : questions normales {d_normal:.3f} | pièges {d_trap:.3f}")
    if n_errors:
        print(f"ATTENTION : {n_errors} question(s) en erreur (voir le fichier de résultats).")
    print(f"Résultats : {output}")

    if args.mlflow:
        from benchmarks.tracking import log_run
        params = {k: v for k, v in meta.items() if k not in ("created_at",)}
        log_run("retrieval_eval", args.run_name, params, tracking_metrics(summary, n_errors), artifacts=[output])
        print("Run journalisé dans MLflow.")


if __name__ == "__main__":
    main()
