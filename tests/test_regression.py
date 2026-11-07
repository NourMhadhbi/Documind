"""
Test de régression : compare le retrieval actuel à la baseline figée du Sprint 5.

- La logique de comparaison est testée en permanence (CI incluse).
- La comparaison réelle ne tourne que si l'index ET la baseline existent.
- Ne jamais écraser le fichier de baseline : le run de contrôle utilise un autre nom.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from config import settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
BASELINE = RESULTS / "retrieval_sprint5-baseline.json"
CHECK_RUN = "regression-check"
TOLERANCE = 0.02  # baisse absolue tolérée (2 points)
METRICS = ("recall@5", "mrr")


def _index_exists() -> bool:
    path = Path(settings.index_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def extract_metrics(data: dict) -> dict:
    """
    Retourne {langue: {"recall@5": x, "mrr": y}}.

    À ADAPTER à la structure réelle de tes fichiers retrieval_*.json
    (ouvre benchmarks/results/retrieval_sprint5-baseline.json pour vérifier).
    Hypothèse ici : data["per_language"][langue][métrique].
    """
    return data["per_language"]


def find_regressions(baseline: dict, current: dict, tolerance: float = TOLERANCE) -> list[str]:
    problems = []
    for lang, base_vals in baseline.items():
        if lang not in current:
            problems.append(f"{lang}: langue absente du run actuel")
            continue
        for metric in METRICS:
            base, cur = base_vals[metric], current[lang][metric]
            if cur < base - tolerance:
                problems.append(f"{lang}/{metric}: {cur:.3f} < baseline {base:.3f} - {tolerance}")
    return problems


def test_comparison_logic():
    base = {"en": {"recall@5": 0.90, "mrr": 0.80}}
    assert find_regressions(base, {"en": {"recall@5": 0.89, "mrr": 0.80}}) == []
    assert find_regressions(base, {"en": {"recall@5": 0.70, "mrr": 0.80}})
    assert find_regressions(base, {})


@pytest.mark.skipif(
    not (_index_exists() and BASELINE.exists()),
    reason="Index ou baseline Sprint 5 absent (normal en CI).",
)
def test_retrieval_does_not_regress():
    subprocess.run(
        [sys.executable, "-m", "benchmarks.eval_retrieval", "--run-name", CHECK_RUN],
        cwd=ROOT,
        check=True,
    )
    baseline = extract_metrics(json.loads(BASELINE.read_text(encoding="utf-8")))
    current = extract_metrics(
        json.loads((RESULTS / f"retrieval_{CHECK_RUN}.json").read_text(encoding="utf-8"))
    )
    problems = find_regressions(baseline, current)
    assert not problems, "Régression détectée :\n" + "\n".join(problems)
