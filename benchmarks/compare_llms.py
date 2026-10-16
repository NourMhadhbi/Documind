"""
Comparaison objective de plusieurs LLM sur le pipeline RAG complet
(retrieval + génération). Alimente docs/adr/003-llm-choice.md avec des
résultats mesurés plutôt que des opinions.

Usage (depuis la racine du projet) :
    python -m benchmarks.compare_llms \
        --candidates anthropic:claude-sonnet-5 ollama:qwen2.5:3b ollama:llama3.2:3b

Chaque candidat s'écrit "fournisseur:modèle". Les prérequis : index du
Sprint 2 construit, ANTHROPIC_API_KEY dans .env pour Anthropic, serveur
Ollama lancé (avec le modèle téléchargé) pour Ollama.

Ce script mesure des indicateurs OBJECTIFS et simples (latence, citation
de la bonne source, refus sur une question piège). Il ne remplace pas la
lecture humaine des réponses, ni l'évaluation RAGAS prévue au Sprint 5.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import settings
from src.generation import generate_answer, GenerationError

logger = logging.getLogger(__name__)

# Questions du benchmark du Sprint 2 (mêmes vérités terrain) + une question
# PIÈGE hors corpus : le retrieval renverra quand même des chunks (un
# index renvoie toujours ses plus proches voisins), donc seul le LLM peut
# refuser de répondre. C'est le test qui distingue le mieux un petit
# modèle (tendance à inventer) d'un grand modèle (respecte la règle 2).
QUESTIONS: List[Dict] = [
    {"query": "How do I create a POST route in FastAPI?",
     "expected_sources": ["tutorial/first-steps.md", "tutorial/body.md"], "trap": False},
    {"query": "How to declare path parameters?",
     "expected_sources": ["tutorial/path-params.md"], "trap": False},
    {"query": "How to validate request body with Pydantic?",
     "expected_sources": ["tutorial/body.md"], "trap": False},
    {"query": "What is dependency injection in FastAPI?",
     "expected_sources": ["tutorial/dependencies/index.md"], "trap": False},
    {"query": "How to return a custom HTTP status code?",
     "expected_sources": ["tutorial/response-status-code.md"], "trap": False},
    {"query": "What is the recipe for a chocolate cake?",
     "expected_sources": [], "trap": True},
]

# Heuristique volontairement simple et transparente : à vérifier à la main.
REFUSAL_MARKERS = [
    "ne sais pas", "pas d'information", "n'ai pas trouvé", "ne contient pas",
    "pas mentionn", "aucune information", "don't know", "not contain",
    "no information", "cannot answer", "impossible de répondre",
]


def parse_candidate(spec: str) -> Tuple[str, str]:
    """'ollama:qwen2.5:3b' -> ('ollama', 'qwen2.5:3b'). Découpe au PREMIER
    ':' seulement, car les noms de modèles Ollama contiennent eux-mêmes ':'."""
    if ":" not in spec:
        raise ValueError(f"Candidat invalide '{spec}' : attendu 'fournisseur:modèle'")
    provider, model = spec.split(":", 1)
    if provider not in ("anthropic", "ollama") or not model:
        raise ValueError(f"Candidat invalide '{spec}' : fournisseur anthropic|ollama + modèle")
    return provider, model


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def cites_expected_source(answer: str, sources: List[str], expected: List[str]) -> bool:
    """Vrai si une source attendue est citée dans le TEXTE de la réponse
    (la règle 4 du prompt exige de citer) ET fait partie des sources."""
    return any(exp in answer and exp in sources for exp in expected)


def summarize(rows: List[Dict]) -> Dict:
    normal = [r for r in rows if not r["trap"] and not r.get("error")]
    traps = [r for r in rows if r["trap"] and not r.get("error")]
    latencies = [r["latency_seconds"] for r in rows if not r.get("error")]
    return {
        "questions_ok": len([r for r in rows if not r.get("error")]),
        "errors": len([r for r in rows if r.get("error")]),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "citation_rate": round(sum(r["cites_expected"] for r in normal) / len(normal), 2) if normal else None,
        "refuses_trap": (sum(r["refused"] for r in traps) / len(traps)) if traps else None,
    }


def run_candidate(provider: str, model: str) -> List[Dict]:
    settings.llm_provider = provider
    if provider == "anthropic":
        settings.llm_model_name = model
    else:
        settings.ollama_model_name = model

    rows = []
    for q in QUESTIONS:
        row = {"query": q["query"], "trap": q["trap"]}
        try:
            result = generate_answer(q["query"])
            row.update(
                answer=result["answer"],
                sources=result["sources"],
                latency_seconds=result["latency_seconds"],
                cites_expected=cites_expected_source(result["answer"], result["sources"], q["expected_sources"]),
                refused=looks_like_refusal(result["answer"]),
            )
        except GenerationError as exc:
            row["error"] = str(exc)
            logger.error("[%s:%s] %s -> %s", provider, model, q["query"], exc)
        rows.append(row)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", required=True, help="ex: anthropic:claude-sonnet-5 ollama:qwen2.5:3b")
    parser.add_argument("--output", default="benchmarks/results/llm_comparison.json")
    args = parser.parse_args()

    all_results = {}
    for spec in args.candidates:
        provider, model = parse_candidate(spec)
        print(f"\n=== {spec} ===")
        rows = run_candidate(provider, model)
        all_results[spec] = {"summary": summarize(rows), "rows": rows}
        print(json.dumps(all_results[spec]["summary"], ensure_ascii=False, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats détaillés (réponses complètes) : {out}")

    print("\n" + "=" * 78)
    print(f"{'Candidat':<30}{'Latence (s)':<14}{'Citations':<12}{'Refus piège':<14}{'Erreurs':<8}")
    print("-" * 78)
    for spec, data in all_results.items():
        s = data["summary"]
        print(f"{spec:<30}{str(s['avg_latency_s']):<14}{str(s['citation_rate']):<12}"
              f"{str(s['refuses_trap']):<14}{s['errors']:<8}")
    print("\nLis aussi les réponses complètes dans le JSON : les indicateurs ci-dessus sont des heuristiques.")


if __name__ == "__main__":
    main()