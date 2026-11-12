from prometheus_client import Counter, Histogram

RAG_REQUESTS = Counter(
    "documind_requests_total",
    "Requêtes /chat par langue et statut de refus",
    ["language", "refused"],
)

RAG_MIN_DISTANCE = Histogram(
    "documind_min_distance",
    "Distance du meilleur chunk",
    buckets=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
)

RAG_GENERATION_SECONDS = Histogram(
    "documind_generation_seconds",
    "Latence du pipeline RAG",
    buckets=[1, 2, 5, 10, 20, 40, 80, 180],
)
