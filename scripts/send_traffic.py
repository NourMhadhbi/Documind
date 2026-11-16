"""
Envoie des questions variées (EN/FR/AR + pièges hors sujet) à l'API pour alimenter
les dashboards et le journal de drift (Sprint 9).

Attention : avec Qwen2.5 3B en local, une requête prend plusieurs secondes à
dizaines de secondes. Commence par --n 10, puis augmente.

Usage :
  python scripts/send_traffic.py --n 30
  python scripts/send_traffic.py --n 100 --url http://127.0.0.1:8000 --seed 7
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = {
    "en": [
        "How do I create a POST route in FastAPI?",
        "How do I declare path parameters?",
        "How do query parameters work?",
        "How do I validate a request body with Pydantic?",
        "What is dependency injection in FastAPI?",
        "How do I return an error with HTTPException?",
    ],
    "fr": [
        "Comment créer une route POST avec FastAPI ?",
        "Comment déclarer un paramètre de chemin ?",
        "Comment utiliser les paramètres de requête ?",
        "Comment valider un corps de requête avec Pydantic ?",
        "Comment fonctionne l'injection de dépendances ?",
        "Comment renvoyer une erreur avec HTTPException ?",
    ],
    "ar": [
        "كيف أنشئ مسار POST في FastAPI؟",
        "كيف أعرّف معاملات المسار في FastAPI؟",
        "كيف تعمل معاملات الاستعلام في FastAPI؟",
        "كيف أتحقق من صحة جسم الطلب باستخدام Pydantic؟",
        "ما هو حقن التبعيات في FastAPI؟",
    ],
}

# Questions hors sujet : le garde-fou doit les refuser.
TRAPS = [
    "What is the capital of France?",
    "Who won the football World Cup in 2018?",
    "Quelle est la recette de la ratatouille ?",
    "Qui a écrit Les Misérables ?",
    "ما هي عاصمة تونس؟",
    "كيف أطبخ الكسكسي؟",
]


def build_pool(trap_ratio: float, rng: random.Random, n: int) -> list[str]:
    normal = [q for questions in QUESTIONS.values() for q in questions]
    pool = []
    for _ in range(n):
        source = TRAPS if rng.random() < trap_ratio else normal
        pool.append(rng.choice(source))
    return pool


def post_question(url: str, question: str, api_key: str | None, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        url.rstrip("/") + "/chat",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=10, help="nombre de requêtes")
    parser.add_argument("--trap-ratio", type=float, default=0.2, help="part de questions pièges")
    parser.add_argument("--pause", type=float, default=0.5, help="secondes entre deux requêtes")
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=None, help="en-tête X-API-Key (Sprint 10)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    questions = build_pool(args.trap_ratio, rng, args.n)
    ok = refused = errors = 0

    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            data = post_question(args.url, question, args.api_key, args.timeout)
            ok += 1
            refused += bool(data.get("refused"))
            status = "refus" if data.get("refused") else "réponse"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors += 1
            status = f"erreur ({exc})"
        print(f"[{index}/{args.n}] {status} en {time.perf_counter() - started:.1f} s")
        time.sleep(args.pause)

    print(f"\nTerminé : {ok} réussies dont {refused} refus, {errors} erreurs.")
    return 0 if errors < args.n else 1


if __name__ == "__main__":
    sys.exit(main())
