import json
import logging
from pathlib import Path
from typing import List,Dict
from sentence_transformers import SentenceTransformer
from config import settings
from src.vector_store import get_vector_store, VectorStoreError

logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger=logging.getLogger(__name__)

class ChunksValidationError(Exception):
    """Levée si le fichier chunks.json ne respecte pas le schéma attendu.
    Sécurité/robustesse."""
    pass
REQUIRED_CHUNK_FIELDS={"chunk_id","source","text"}

def load_chunks(path:str)->List[Dict]:
    """charge et valide les chunks depuis le fichier json généré par ingestion.py"""
    chunks_file=Path(path)
    if not chunks_file.exists():
        raise ChunksValidationError(
            f"Fichier introuvable:{path}. As-tu lancé ingestion.py d'abord"
              )
        
    with open(chunks_file,"r",encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data,list) or len(data) == 0:
        raise ChunksValidationError("chunks.json doit contenir une liste non vide")

    for i,chunk in enumerate(data):
        missing = REQUIRED_CHUNK_FIELDS - chunk.keys()
        if missing:
            raise ChunksValidationError (
                f"Chunk #{i} invalide, champs manquants : {missing}"
            )

    logger.info("chunks.json validé:%d chunks conformes au schéma" ,len(data))
    return data
def encode_chunks(chunks:List[Dict],model:SentenceTransformer,
                  batch_size:int )->List[List[float]]:
    """Encode tous les chunks en batch (plus rapide et plus économe en
    mémoire qu'encoder un par un)."""
    texts=[c["text"] for c in chunks]
    embeddings = model.encode(
        texts,batch_size=batch_size,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    return embeddings.tolist()

def build_index()->None:
    logger.info("Chargement du modéle d'embedding :%s",settings.embedding_model_name)
    model=SentenceTransformer(settings.embedding_model_name,device=settings.embedding_device)
    chunks=load_chunks(settings.chunks_path)
    logger.info("Encodage de %d chunks (batch_size=%d)",len(chunks),settings.embedding_batch_size)
    embeddings = encode_chunks(chunks, model, settings.embedding_batch_size)
    store=get_vector_store(
        backend=settings.vector_store_backend,
        index_path=settings.index_path,
        collection_name=settings.vector_collection_name,
    )
    ids=[c["chunk_id"] for c in chunks]
    texts=[c["text"]for c in chunks]
    metadatas=[{"source":c["source"]} for c in chunks]

    try:
        store.add(ids=ids,texts=texts,embeddings=embeddings,metadatas=metadatas)
    except VectorStoreError:
        logger.exception("Échec de la construction de l'index")
        raise
    logger.info("Index construit avec succès. Total indexé : %d chunks", store.count())

if __name__=="__main__":
    build_index()
    
    


