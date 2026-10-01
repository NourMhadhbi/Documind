"""
Abstraction du vector store.
 
Pourquoi une interface abstraite (ABC) ici :
- Le reste du projet (retrieval.py, generation.py, l'API) ne doit JAMAIS
  importer ChromaDB directement. Il doit dépendre de l'interface
  `VectorStore`, pas d'une implémentation précise.
"""

import logging
from abc import ABC,abstractmethod
from typing import List,Dict,Any
logger =logging.getLogger(__name__)

class VectorStoreError(Exception):
    pass

class VectorStore(ABC):
    """Interface commune que toute base vectorielle doit respecter."""
    @abstractmethod
    def add(self,ids:List[str],texts:List[str],embeddings:List[List[float]],metadatas:List[Dict[str,Any]])->None:
        """Insère des chunks avec leurs embeddings pré-calculés et métadonnées."""
        raise NotImplementedError
    @abstractmethod
    def query (self,query_embedding:List[float],top_k:int=5)->List[Dict[str,Any]]:
        """Retourne les top_k chunks les plus proches, chacun avec au
        minimum : {"text": ..., "source": ..., "score": ...}."""
        raise NotImplementedError

    @abstractmethod
    def count (self)->int:
        """Nombre de chunks actuellement indexés"""
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    """Implémentation concrète basée sur ChromaDB(persistance locale sur disque ) """

    def __init__(self,index_path:str,collection_name:str):
        import chromadb

        if not index_path or not collection_name:
            raise VectorStoreError("index_path et collection_name sont requis")

        self._client=chromadb.PersistentClient(path=index_path)
        self._collection=self._client.get_or_create_collection(collection_name)
        logger.info("ChromaVectorStore initialisé (path=%s,collection=%s)",index_path,collection_name)

    def add(self,ids,texts,embeddings,metadatas)->None:

        if not (len(ids)==len(texts)==len(embeddings)==len(metadatas)):
            raise VectorStoreError(
                f"Longueurs incohérentes ;ids={len(ids)},texts={len(texts)},"
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"

            )
        if len(ids)==0:
            logger.warning("add() appelé avec une liste vide,aucune insertion effectuées")
            return
        try:
            self._collection.add(
                ids=ids,documents=texts,embeddings=embeddings,metadatas=metadatas
            )
        except Exception as exc:
            raise VectorStoreError(f"Échec de l'insertion dans chromaDB:{exc}") from exc

    def query (self,query_embedding,top_k=5)->List[Dict[str,Any]]:
        if top_k <= 0:
            raise VectorStoreError("top_k doit être strictement positif ")

        try:
            results=self._collection.query(
                query_embeddings=[query_embedding],n_results=top_k
            )
        except Exception as exc:
            raise VectorStoreError(f"Échec de la requête ChromaDB : {exc}") from exc

        output=[]
        docs=results.get("documents",[[]])[0]
        metas=results.get("metadatas",[[]])[0]
        distances=results.get("distances",[[]])[0]
        for text,meta,dist in zip(docs,metas,distances):
            output.append({
                "text":text,
                "source":meta.get("source","unknown"),
                "score":dist,
            })
        return output

    def count(self)->int:
        return self._collection.count()
    
def get_vector_store(backend:str,index_path:str,collection_name:str)-> VectorStore:
        
        if backend == "chroma":
            return ChromaVectorStore(index_path,collection_name)

        raise VectorStoreError(f"Backend vector store inconnu ou non implémenté:{backend }")











