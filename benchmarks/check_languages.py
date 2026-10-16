"""
Vérifie si le RETRIEVAL fonctionne quand la question n'est pas en anglais.

Le corpus (documentation FastAPI) est en anglais et l'index a été construit
avec all-MiniLM-L6-v2, un modèle d'embeddings surtout entraîné sur l'anglais.
Une question en français ou en arabe risque donc de retrouver de moins bons
passages, quelle que soit la qualité du LLM qui répond ensuite.

Ce script n'appelle AUCUN LLM : il pose la même question en anglais,
français et arabe, et regarde si la bonne page est dans le top-k.

    python -m benchmarks.check_languages
"""

from typing import Dict, List, Optional

from src.retrieval import retrieve, RetrievalError

TOP_K = 5

# Les 5 sujets du benchmark du Sprint 2, chacun dans trois langues (15 questions).
QUESTIONS: List[Dict] = [
    # 1. Route POST
    {"lang": "en", "query": "How do I create a POST route in FastAPI?",
     "expected": ["tutorial/first-steps.md", "tutorial/body.md"]},
    {"lang": "fr", "query": "Comment créer une route POST dans FastAPI ?",
     "expected": ["tutorial/first-steps.md", "tutorial/body.md"]},
    {"lang": "ar", "query": "كيف أنشئ مسار POST في FastAPI؟",
     "expected": ["tutorial/first-steps.md", "tutorial/body.md"]},
    # 2. Paramètres de chemin
    {"lang": "en", "query": "How to declare path parameters in FastAPI?",
     "expected": ["tutorial/path-params.md"]},
    {"lang": "fr", "query": "Comment déclarer des paramètres de chemin dans FastAPI ?",
     "expected": ["tutorial/path-params.md"]},
    {"lang": "ar", "query": "كيف أعرّف معاملات المسار (path parameters) في FastAPI؟",
     "expected": ["tutorial/path-params.md"]},
    # 3. Validation du corps de requête
    {"lang": "en", "query": "How to validate request body with Pydantic?",
     "expected": ["tutorial/body.md"]},
    {"lang": "fr", "query": "Comment valider le corps d'une requête avec Pydantic ?",
     "expected": ["tutorial/body.md"]},
    {"lang": "ar", "query": "كيف أتحقق من صحة جسم الطلب باستخدام Pydantic؟",
     "expected": ["tutorial/body.md"]},
    # 4. Injection de dépendances
    {"lang": "en", "query": "What is dependency injection in FastAPI?",
     "expected": ["tutorial/dependencies/index.md"]},
    {"lang": "fr", "query": "Qu'est-ce que l'injection de dépendances dans FastAPI ?",
     "expected": ["tutorial/dependencies/index.md"]},
    {"lang": "ar", "query": "ما هو حقن التبعيات (dependency injection) في FastAPI؟",
     "expected": ["tutorial/dependencies/index.md"]},
    # 5. Code de statut HTTP
    {"lang": "en", "query": "How to return a custom HTTP status code?",
     "expected": ["tutorial/response-status-code.md"]},
    {"lang": "fr", "query": "Comment retourner un code de statut HTTP personnalisé ?",
     "expected": ["tutorial/response-status-code.md"]},
    {"lang": "ar", "query": "كيف أُرجع رمز حالة HTTP مخصصًا؟",
     "expected": ["tutorial/response-status-code.md"]},
]


def rank_of_expected(sources: List[str], expected: List[str]) -> Optional[int]:
    """Rang (à partir de 1) de la première source attendue, ou None si absente."""
    for rank, source in enumerate(sources, start=1):
        if source in expected:
            return rank
    return None


def recall_by_language(rows: List[Dict]) -> Dict[str, float]:
    """Proportion de questions dont la bonne page est dans le top-k, par langue."""
    totals: Dict[str, int] = {}
    hits: Dict[str, int] = {}
    for row in rows:
        lang = row["lang"]
        totals[lang] = totals.get(lang, 0) + 1
        if row["rank"] is not None:
            hits[lang] = hits.get(lang, 0) + 1
    return {lang: hits.get(lang, 0) / total for lang, total in totals.items()}


def main() -> None:
    rows = []
    for q in QUESTIONS:
        try:
            chunks = retrieve(q["query"], top_k=TOP_K)
        except RetrievalError as exc:
            print(f"[{q['lang']}] erreur : {exc}")
            continue
        sources = [c["source"] for c in chunks]
        rows.append({**q, "sources": sources, "rank": rank_of_expected(sources, q["expected"])})

    print(f"\n{'Langue':<8}{'Rang de la bonne page':<24}Question")
    print("-" * 78)
    for row in rows:
        rank = row["rank"] if row["rank"] is not None else f"absente du top-{TOP_K}"
        print(f"{row['lang']:<8}{str(rank):<24}{row['query']}")

    print("\nRecall@%d par langue :" % TOP_K)
    for lang, value in sorted(recall_by_language(rows).items()):
        print(f"  {lang} : {value:.2f}")
    print("\n5 questions par langue : indicatif. Compare les deux index sur les MÊMES questions.")


if __name__ == "__main__":
    main()
