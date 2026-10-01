import json
import time
import logging
from typing import List, Dict
import mlflow
import os
from dotenv import load_dotenv
import sys

from sentence_transformers import SentenceTransformer
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
mlflow.set_experiment("vector_db_benchmark") 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_PATH = "data/processed/chunks.json"
TOP_K = 5

# ================================================================
# Jeu de requêtes de benchmark AVEC vérité terrain (ground truth).
# "expected_sources" doit être vérifié/ajusté par toi en lisant la
# vraie doc FastAPI dans data/raw/.
# ================================================================
BENCHMARK_QUERIES: List[Dict] = [
    {
        "query": "How do I create a POST route in FastAPI?",
        "expected_sources": ["tutorial/first-steps.md"],
    },
    {
        "query": "How to declare path parameters?",
        "expected_sources": ["tutorial/path-params.md"],
    },
    {
        "query": "How to validate request body with Pydantic?",
        "expected_sources": ["tutorial/body.md"],
    },
    {
        "query": "What is dependency injection in FastAPI?",
        "expected_sources": ["tutorial/dependencies/index.md"],
    },
    {
        "query": "How to return a custom HTTP status code?",
        "expected_sources": ["tutorial/response-status-code.md"],
    },
]


def load_chunks(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_chunks(chunks: List[Dict], model: SentenceTransformer):
    """
    Encode une seule fois tous les chunks (partagé par les 3 backends,
    pour une comparaison juste), et extrait au passage ids/textes/sources
    une seule fois -> évite de recalculer ces listes dans chaque fonction
    benchmark_* (principe DRY : Don't Repeat Yourself).
    """
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    sources = [c["source"] for c in chunks]

    t0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True)
    encode_time = time.perf_counter() - t0

    return embeddings, encode_time, texts, ids, sources


# ================================================================
# Métriques de QUALITÉ du retrieval
# ================================================================

def recall_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """1.0 si au moins une source attendue apparaît parmi les résultats
    retournés, sinon 0.0."""
    return 1.0 if any(src in retrieved_sources for src in expected_sources) else 0.0


def reciprocal_rank(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """1/rang du premier résultat correct trouvé (1.0 si en 1ère position,
    0.5 si en 2ème, ..., 0.0 si absent des résultats)."""
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in expected_sources:
            return 1.0 / rank
    return 0.0


def evaluate_quality(backend_name: str, query_fn, model: SentenceTransformer,
                      queries: List[Dict], top_k: int) -> Dict:
    """
    Exécute toutes les BENCHMARK_QUERIES sur un backend donné et calcule
    Recall@k et MRR moyens.

    `query_fn` est une fonction (query_embedding, top_k) -> List[str des sources
    retournées, dans l'ordre de pertinence] — chaque backend fournit la sienne
    (adaptateur), ce qui permet à cette fonction de rester générique et
    réutilisable pour ChromaDB, FAISS et Qdrant sans dupliquer la logique
    de calcul des métriques.
    """
    recalls, rr_scores = [], []
    details = []  # (corrigé : une seule orthographe, cohérente partout)

    for item in queries:
        query_text = item["query"]
        expected = item["expected_sources"]

        query_embedding = model.encode([query_text])[0]
        retrieved_sources = query_fn(query_embedding, top_k)

        r = recall_at_k(retrieved_sources, expected)
        rr = reciprocal_rank(retrieved_sources, expected)
        recalls.append(r)
        rr_scores.append(rr)

        details.append({
            "query": query_text,
            "expected": expected,
            "retrieved": retrieved_sources,
            "recall": r,
            "reciprocal_rank": rr,
        })
        logger.info("[%s] '%s' -> recall=%.1f, RR=%.2f | attendu=%s | trouvé=%s",
                     backend_name, query_text, r, rr, expected, retrieved_sources)

    return {
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "mean_reciprocal_rank": sum(rr_scores) / len(rr_scores),
        "details": details,
    }


# ================================================================
# Backends : chacun construit son index, définit son propre query_fn
# (adaptateur), et mesure temps d'indexation + latence + qualité.
# ================================================================

def benchmark_chromadb(chunks, embeddings, texts, ids, sources, model) -> Dict:
    import chromadb

    client = chromadb.PersistentClient(path="data/index/bench_chroma")
    collection = client.get_or_create_collection("bench_chroma", metadata={"hnsw:space": "cosine"})

    metadatas = [{"source": s} for s in sources]

    t0 = time.perf_counter()
    collection.add(ids=ids, documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    index_time = time.perf_counter() - t0

    def query_fn(query_embedding, top_k):
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        return [m["source"] for m in results["metadatas"][0]]

    latencies = []
    for item in BENCHMARK_QUERIES:
        q_emb = model.encode([item["query"]])[0]
        t0 = time.perf_counter()
        query_fn(q_emb, TOP_K)
        latencies.append(time.perf_counter() - t0)

    quality = evaluate_quality("ChromaDB", query_fn, model, BENCHMARK_QUERIES, TOP_K)

    return {
        "index_time": index_time,
        "avg_query_latency": sum(latencies) / len(latencies),
        "quality": quality,
    }


def benchmark_faiss(chunks, embeddings, texts, ids, sources, model) -> Dict:
    import faiss
    import numpy as np

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]

    # FAISS ne connaît que des vecteurs et des indices numériques (0, 1, 2...)
    # -> on garde nous-mêmes le mapping position -> source pour pouvoir
    # retrouver la source après une recherche.
    sources_lookup = sources

    index = faiss.IndexFlatL2(dimension)  # recherche par distance euclidienne

    t0 = time.perf_counter()
    index.add(embeddings_np)
    index_time = time.perf_counter() - t0

    def query_fn(query_embedding, top_k):
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = index.search(query_np, top_k)
        return [sources_lookup[i] for i in indices[0]]

    latencies = []
    for item in BENCHMARK_QUERIES:
        q_emb = model.encode([item["query"]])[0]
        t0 = time.perf_counter()
        query_fn(q_emb, TOP_K)
        latencies.append(time.perf_counter() - t0)

    quality = evaluate_quality("FAISS", query_fn, model, BENCHMARK_QUERIES, TOP_K)

    return {
        "index_time": index_time,
        "avg_query_latency": sum(latencies) / len(latencies),
        "quality": quality,
    }


def benchmark_qdrant(chunks, embeddings, texts, ids, sources, model) -> Dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(":memory:")
    collection_name = "bench_qdrant"
    dimension = len(embeddings[0])

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"source": sources[i]})
        for i in range(len(chunks))
    ]

    t0 = time.perf_counter()
    client.upsert(collection_name=collection_name, points=points)
    index_time = time.perf_counter() - t0

    def query_fn(query_embedding, top_k):
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        )
        return [point.payload["source"] for point in results.points]

    latencies = []
    for item in BENCHMARK_QUERIES:
        q_emb = model.encode([item["query"]])[0]
        t0 = time.perf_counter()
        query_fn(q_emb, TOP_K)
        latencies.append(time.perf_counter() - t0)

    quality = evaluate_quality("Qdrant", query_fn, model, BENCHMARK_QUERIES, TOP_K)

    return {
        "index_time": index_time,
        "avg_query_latency": sum(latencies) / len(latencies),
        "quality": quality,
    }

def print_failures(results: Dict[str, Dict]) -> None:
    """Affiche les questions dont la bonne source n'a pas été trouvée."""
    print("\n" + "=" * 90)
    print("QUESTIONS EN ÉCHEC (recall = 0)")
    print("=" * 90)
    # Les 3 bases donnent les mêmes résultats, on lit celles de ChromaDB
    for d in results["ChromaDB"]["quality"]["details"]:
        if d["recall"] == 0.0:
            print(f"\nQuestion : {d['query']}")
            print(f"  Attendu : {d['expected']}")
            print(f"  Trouvé  : {sorted(set(d['retrieved']))}")

def print_comparison(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 90)
    print("COMPARAISON DES BASES VECTORIELLES — PERFORMANCE ET QUALITÉ")
    print("=" * 90)
    header = f"{'Base':<12}{'Index (s)':<12}{'Latence (ms)':<15}{'Recall@k':<12}{'MRR':<10}"
    print(header)
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<12}{r['index_time']:<12.3f}"
              f"{r['avg_query_latency']*1000:<15.2f}"
              f"{r['quality']['mean_recall_at_k']:<12.2f}"
              f"{r['quality']['mean_reciprocal_rank']:<10.2f}")
    print("\nRappel : Recall@k et MRR proches de 1.0 = meilleure qualité de retrieval.")

def log_to_mlflow(backend_name: str, results: Dict) -> None:
    with mlflow.start_run(run_name=f"benchmark_{backend_name.lower()}"):
        mlflow.log_param("backend", backend_name)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL_NAME)
        mlflow.log_param("top_k", TOP_K)

        mlflow.log_metric("index_time_seconds", results["index_time"])
        mlflow.log_metric("avg_query_latency_ms", results["avg_query_latency"] * 1000)
        mlflow.log_metric("mean_recall_at_k", results["quality"]["mean_recall_at_k"])
        mlflow.log_metric("mean_reciprocal_rank", results["quality"]["mean_reciprocal_rank"])

if __name__ == "__main__":
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    logger.info("Chunks chargés : %d", len(chunks))

    embeddings, encode_time, texts, ids, sources = encode_chunks(chunks, model)
    logger.info("Encodage de tous les chunks : %.2fs", encode_time)

    results = {
        "ChromaDB": benchmark_chromadb(chunks, embeddings, texts, ids, sources, model),
        "FAISS": benchmark_faiss(chunks, embeddings, texts, ids, sources, model),
        "Qdrant": benchmark_qdrant(chunks, embeddings, texts, ids, sources, model),
    }

    print_comparison(results)
    print_failures(results)
 #  on logge chaque backend comme une run séparée
    for backend_name, backend_results in results.items():
        log_to_mlflow(backend_name, backend_results)
        logger.info("Résultats de %s loggés dans MLflow", backend_name)