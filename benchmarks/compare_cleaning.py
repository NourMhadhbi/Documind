"""
Compare l'ANCIEN nettoyage (Sprint 1) et le nettoyage CORRIGÉ sur les mêmes
questions, avec le même modèle d'embeddings et le même découpage (800/100).

Tout se fait en mémoire : ton index ChromaDB n'est ni lu ni modifié. La seule
variable qui change entre les deux colonnes est la fonction de nettoyage.

    python -m benchmarks.compare_cleaning
    python -m benchmarks.compare_cleaning --explain          # montre les chunks qui ont changé le classement
    python -m benchmarks.compare_cleaning --ablate --explain # teste aussi chaque règle séparément
    python -m benchmarks.compare_cleaning --model paraphrase-multilingual-MiniLM-L12-v2

Les embeddings sont mis en cache par texte de chunk : une variante ne ré-encode
que les chunks qui diffèrent des précédentes.
"""

import argparse
import re
import sys
from typing import Callable, Dict, List, Optional
from unittest.mock import patch

import numpy as np

from benchmarks.check_languages import QUESTIONS, rank_of_expected
from src import ingestion
from src.ingestion import chunk_text, load_markdown_files, clean_text as new_clean_text

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
KEEP_CODE_BLOCKS = True
K = 5  # même top-k que le système réel

# ----------------------------------------------------------------------
# Ancien nettoyage (Sprint 1), conservé UNIQUEMENT comme référence de mesure.
# Ne pas l'utiliser ailleurs : il contient les défauts corrigés depuis.
# ----------------------------------------------------------------------
_L_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_L_INLINE_CODE = re.compile(r"`([^`]+)`")
_L_IMAGE = re.compile(r"!\[.*?\]\(.*?\)")
_L_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_L_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_L_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")
_L_ADMONITION = re.compile(r"^!!!\s*\w+.*$", re.MULTILINE)
_L_HTML_TAG = re.compile(r"<[^>]+>")
_L_HR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_L_MULTI_SPACE = re.compile(r"[ \t]+")
_L_MULTI_BLANK = re.compile(r"\n{3,}")


def legacy_clean_text(text: str, keep_code_blocks: bool = False) -> str:
    if not text:
        return ""
    if not keep_code_blocks:
        text = _L_CODE_BLOCK.sub("", text)
    text = _L_INLINE_CODE.sub(r"\1", text)
    text = _L_IMAGE.sub("", text)
    text = _L_LINK.sub(r"\1", text)
    text = _L_HEADER.sub("", text)
    text = _L_BOLD_ITALIC.sub(r"\2", text)
    text = _L_ADMONITION.sub("", text)
    text = _L_HTML_TAG.sub(" ", text)
    text = _L_HR.sub("", text)
    text = _L_MULTI_SPACE.sub(" ", text)
    text = _L_MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------
# Fonctions de mesure
# ----------------------------------------------------------------------

def build_chunks_with(clean_fn: Callable, documents: List[Dict], chunk_size: int,
                      chunk_overlap: int, keep_code_blocks: bool) -> List[Dict]:
    chunks = []
    for doc in documents:
        cleaned = clean_fn(doc["content"], keep_code_blocks)
        if not cleaned:
            continue
        for i, piece in enumerate(chunk_text(cleaned, chunk_size, chunk_overlap)):
            chunks.append({"chunk_id": f"{doc['filename']}_{i}", "source": doc["filename"], "text": piece})
    return chunks


def rank_for_question(scores: np.ndarray, sources: List[str], expected: List[str],
                      depth: int) -> Optional[int]:
    """Rang (1 = meilleur) du premier chunk dont la source est attendue,
    parmi les `depth` chunks les mieux classés ; None si absent."""
    order = np.argsort(-scores, kind="stable")[:depth]
    return rank_of_expected([sources[i] for i in order], expected)


def best_expected(scores: np.ndarray, chunks: List[Dict], expected: List[str],
                  depth: int) -> Optional[Dict]:
    """Premier chunk (dans le classement) dont la source est attendue :
    {"rank", "chunk_id", "text"}, ou None s'il n'est pas dans le top-`depth`."""
    order = np.argsort(-scores, kind="stable")[:depth]
    for rank, idx in enumerate(order, start=1):
        if chunks[idx]["source"] in expected:
            return {"rank": rank, "chunk_id": chunks[idx]["chunk_id"], "text": chunks[idx]["text"]}
    return None


def encode_with_cache(model, texts: List[str], cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Encode seulement les textes jamais vus ; les autres viennent du cache."""
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        vectors = model.encode(missing, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        for text, vector in zip(missing, vectors):
            cache[text] = vector
    return np.array([cache[t] for t in texts])


def evaluate_variant(chunks: List[Dict], model, questions: List[Dict], depth: int,
                     cache: Optional[Dict[str, np.ndarray]] = None) -> List[Dict]:
    cache = cache if cache is not None else {}
    doc_emb = encode_with_cache(model, [c["text"] for c in chunks], cache)
    q_emb = model.encode([q["query"] for q in questions], normalize_embeddings=True)
    rows = []
    for q, vec in zip(questions, q_emb):
        scores = doc_emb @ np.asarray(vec)  # cosinus (vecteurs normalisés)
        best = best_expected(scores, chunks, q["expected"], depth)
        top3 = [chunks[i]["chunk_id"] for i in np.argsort(-scores, kind="stable")[:3]]
        rows.append({"lang": q["lang"], "query": q["query"],
                     "rank": best["rank"] if best else None, "best": best, "top3": top3})
    return rows


# Règles du nettoyage corrigé qu'on peut désactiver une à une (ablation).
NEVER_MATCHES = re.compile(r"(?!x)x")
ABLATIONS = {
    "sans-directives": "RE_INCLUDE_DIRECTIVE",   # garde les {* fichier.py *}
    "sans-///": "RE_ADMONITION_SLASHES",         # garde les marqueurs /// tip
}


def corrected_without_anchors(text: str, keep_code_blocks: bool = False) -> str:
    """Le nettoyage corrigé AVEC suppression des ancres (essai précédent, variante B)."""
    return new_clean_text(text, keep_code_blocks, strip_anchors=True)


def make_variant(disabled_regex: Optional[str]) -> Callable:
    """Nettoyage retenu (ancres conservées), avec éventuellement UNE règle désactivée."""
    if disabled_regex is None:
        return new_clean_text

    def clean_without(text: str, keep_code_blocks: bool = False) -> str:
        with patch.object(ingestion, disabled_regex, NEVER_MATCHES):
            return new_clean_text(text, keep_code_blocks)
    return clean_without


def summarize(rows: List[Dict], k: int = K) -> Dict[str, Dict[str, float]]:
    """Recall@k et MRR (sur la profondeur analysée) par langue."""
    acc: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = acc.setdefault(r["lang"], {"n": 0, "hits": 0, "rr": 0.0})
        d["n"] += 1
        if r["rank"] is not None and r["rank"] <= k:
            d["hits"] += 1
        if r["rank"] is not None:
            d["rr"] += 1.0 / r["rank"]
    return {lang: {"recall": d["hits"] / d["n"], "mrr": d["rr"] / d["n"]} for lang, d in acc.items()}


def _fmt_rank(rank: Optional[int], depth: int) -> str:
    return f">{depth}" if rank is None else str(rank)


def _verdict(old: Optional[int], new: Optional[int]) -> str:
    big = 10 ** 6
    o, n = (old or big), (new or big)
    return "mieux" if n < o else ("moins bien" if n > o else "=")


def _one_line(text: str, width: int = 320) -> str:
    return " / ".join(text.split("\n"))[:width]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--depth", type=int, default=20, help="profondeur de classement analysée")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--explain", action="store_true", help="affiche les chunks des questions dégradées")
    parser.add_argument("--ablate", action="store_true", help="teste aussi chaque règle séparément")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    docs = load_markdown_files(args.data_dir)

    variants = [("A", "ancien nettoyage (Sprint 1)", legacy_clean_text),
                ("B", "corrigé, ancres supprimées (essai précédent)", corrected_without_anchors),
                ("C", "corrigé, ancres conservées (RETENU, ADR 004)", make_variant(None))]
    if args.ablate:
        for label, (key, regex) in zip("DE", ABLATIONS.items()):
            variants.append((label, f"C {key}", make_variant(regex)))

    cache: Dict[str, np.ndarray] = {}
    results: Dict[str, List[Dict]] = {}
    for label, description, fn in variants:
        chunks = build_chunks_with(fn, docs, CHUNK_SIZE, CHUNK_OVERLAP, KEEP_CODE_BLOCKS)
        print(f"\n[{label}] {description} : {len(chunks)} chunks")
        results[label] = evaluate_variant(chunks, model, QUESTIONS, args.depth, cache)

    labels = [v[0] for v in variants]
    print(f"\nModèle : {args.model} | chunks {CHUNK_SIZE}/{CHUNK_OVERLAP} | code conservé | top-{K}")
    for label, description, _ in variants:
        print(f"  {label} = {description}")

    print("\nRang de la bonne page (1 = meilleur) :")
    print(f"{'Lang':<6}" + "".join(f"{l:<7}" for l in labels) + f"{'A>C':<12}Question")
    print("-" * (6 + 7 * len(labels) + 12 + 50))
    for i, q in enumerate(QUESTIONS):
        ranks = [results[l][i]["rank"] for l in labels]
        print(f"{q['lang']:<6}" + "".join(f"{_fmt_rank(r, args.depth):<7}" for r in ranks)
              + f"{_verdict(ranks[0], ranks[2]):<12}{q['query'][:50]}")

    print("\nRecall@5 / MRR par langue :")
    print(f"{'Lang':<6}" + "".join(f"{l + ' rec/mrr':<14}" for l in labels))
    summaries = {l: summarize(results[l]) for l in labels}
    for lang in sorted(summaries[labels[0]]):
        print(f"{lang:<6}" + "".join(
            f"{summaries[l][lang]['recall']:.2f}/{summaries[l][lang]['mrr']:.2f}".ljust(14) for l in labels))

    if args.explain:
        print("\n" + "=" * 90 + "\nQUESTIONS ENCORE DÉGRADÉES (A -> C) : chunks concernés\n" + "=" * 90)
        for i, q in enumerate(QUESTIONS):
            a, c = results["A"][i], results["C"][i]
            if _verdict(a["rank"], c["rank"]) != "moins bien":
                continue
            print(f"\n[{q['lang']}] {q['query']}")
            for label, row in (("A", a), ("C", c)):
                best = row["best"]
                if best:
                    print(f"  {label} : meilleur chunk attendu au rang {best['rank']} ({best['chunk_id']})")
                    print(f"      {_one_line(best['text'])}")
                else:
                    print(f"  {label} : aucun chunk attendu dans le top-{args.depth}")
                print(f"      top-3 : {row['top3']}")

    print("\n5 questions par langue : indicatif. Une différence d'une seule question n'est pas significative.")


if __name__ == "__main__":
    main()