"""
Rapport de drift DocuMind (Sprint 9) avec Evidently.

Compare les requêtes récentes (logs/requests.jsonl) à une référence :
  - par défaut : la baseline du Sprint 5 (langue et distance des questions de référence) ;
  - ou --reference-from-logs N : les N premières requêtes du journal.

Limite : la baseline du Sprint 5 ne contient que la langue et la distance. Avec
--reference-from-logs, la comparaison porte aussi sur le refus, la latence et le
nombre de sources.

Usage :
  python -m monitoring.drift_report
  python -m monitoring.drift_report --reference-from-logs 50

Evidently est dans requirements-monitoring.txt (pas dans l'image de l'API).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_LOGS = "logs/requests.jsonl"
DEFAULT_BASELINE = "benchmarks/results/retrieval_sprint5-baseline.json"
DEFAULT_OUTPUT_DIR = "monitoring/reports"
COLUMNS = ["language", "min_distance", "refused", "latency_seconds", "n_sources"]
MIN_ROWS_WARNING = 30


def load_logs(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # une ligne corrompue ne doit pas bloquer le rapport
    frame = pd.DataFrame(rows)
    if "ts" in frame:
        frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    return frame


def load_baseline(path: str) -> pd.DataFrame:
    """Construit la référence depuis un fichier retrieval_*.json de benchmarks/."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for row in data.get("rows", []):
        if row.get("error") or row.get("top1_distance") is None:
            continue
        rows.append(
            {
                "language": row.get("lang") or row.get("language") or "unknown",
                "min_distance": row["top1_distance"],
            }
        )
    return pd.DataFrame(rows)


def split_reference_from_logs(frame: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return frame.iloc[:n].reset_index(drop=True), frame.iloc[n:].reset_index(drop=True)


def common_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    return [c for c in COLUMNS if c in reference.columns and c in current.columns]


def prepare(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame[columns].copy()
    for column in ("language", "refused"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    return frame


def summarize(frame: pd.DataFrame, name: str) -> str:
    parts = [f"{name} : {len(frame)} lignes"]
    if "language" in frame:
        parts.append("langues " + json.dumps(frame["language"].value_counts().to_dict()))
    if "min_distance" in frame:
        parts.append(f"distance médiane {frame['min_distance'].median():.3f}")
    if "refused" in frame:
        parts.append(f"taux de refus {frame['refused'].astype(str).eq('True').mean():.1%}")
    if "latency_seconds" in frame:
        parts.append(f"latence médiane {frame['latency_seconds'].median():.1f} s")
    return " | ".join(parts)


def build_html_report(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:  # Evidently récent (API Report([...]).run(...))
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(output))
    except ImportError:  # anciennes versions d'Evidently
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(output))


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=DEFAULT_LOGS)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--reference-from-logs", type=int, default=None, metavar="N")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if not Path(args.logs).is_file():
        print(f"Journal introuvable : {args.logs}. Envoie du trafic avec scripts/send_traffic.py.")
        return 1
    logs = load_logs(args.logs)
    if logs.empty:
        print("Le journal est vide.")
        return 1

    if args.reference_from_logs:
        reference, current = split_reference_from_logs(logs, args.reference_from_logs)
    else:
        if not Path(args.baseline).is_file():
            print(f"Baseline introuvable : {args.baseline}")
            return 1
        reference, current = load_baseline(args.baseline), logs

    columns = common_columns(reference, current)
    if reference.empty or current.empty or not columns:
        print("Pas assez de données communes pour comparer (référence ou journal trop court).")
        return 1
    reference, current = prepare(reference, columns), prepare(current, columns)

    print(summarize(reference, "Référence"))
    print(summarize(current, "Actuel   "))
    if min(len(reference), len(current)) < MIN_ROWS_WARNING:
        print(f"AVERTISSEMENT : moins de {MIN_ROWS_WARNING} lignes d'un côté, résultat indicatif.")

    output = Path(args.output_dir) / f"drift_{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}.html"
    try:
        build_html_report(reference, current, output)
    except ImportError:
        print("Evidently n'est pas installé : pip install -r requirements-monitoring.txt")
        return 1
    print(f"Rapport : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
