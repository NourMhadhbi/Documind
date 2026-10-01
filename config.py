from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Literal

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

    dagshub_token: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""   
    mlflow_tracking_password: str = ""  



settings= Settings()
    
