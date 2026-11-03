from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Literal, Optional

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8")

    raw_data_dir:str="data/raw"
    chunks_path:str="data/processed/chunks.json"
    index_path:str="data/index"

    chunk_size:int=800
    chunk_overlap:int=100
    min_chunk_length:int=20
     # ---- Embeddings ----
    embedding_model_name:str="all-MiniLM-L6-v2"
    embedding_batch_size:int=64
    embedding_device:Literal["cpu","cuda"]="cpu"

    # ---- Vector store ----
    # Choix tracé et justifié dans docs/adr/002-vector-database-choice.md
    vector_store_backend:Literal["chroma","faiss","qdrant"]="chroma"
    vector_collection_name:str="documind_chunks"

     # ---- Retrieval ----
    retrieval_top_k: int = 5
 
    # ---- Génération : choix du fournisseur LLM ----
    # "anthropic" (cloud, qualité maximale) ou "ollama" (local, gratuit,
    # hors-ligne). Décision à documenter dans docs/adr/003-llm-choice.md
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
 
    # Anthropic
    llm_model_name: str = "claude-sonnet-5"
    anthropic_api_key: str = ""
 
    # Ollama (local)
    ollama_base_url: str = "http://localhost:11435"
    ollama_model_name: str = "qwen2.5:3b"  # ou "llama3.2:3b"
    # Le 1er appel charge le modèle en RAM (plusieurs dizaines de secondes
    # sur CPU) : un timeout trop court ferait échouer la 1ère question.
    ollama_timeout_seconds: int = 360
 
    # Communs aux deux fournisseurs
    llm_max_tokens: int = 1024
    # None = ne pas envoyer le paramètre (certains modèles récents le refusent)
    llm_temperature: Optional[float] = None
 
    dedupe_chunks_by_source: bool = True
    max_chunks_per_source: int = 1

    dagshub_token: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""   
    mlflow_tracking_password: str = ""  
    max_distance: float = 0.96



settings= Settings()
    
