import os
import json
from pathlib import Path 
from typing import List, Dict
def load_markdown_files(source_dir: str)-> List[Dict]:
    """
    Parcourt source_dir récursivement, lit chaque fichier .md,
    retourne une Liste de dicts : {"filename": ..., "content": ...}
    """
    files= []
    for file in Path(source_dir).rglob("*.md"):
        with open(file,"r",encoding="utf-8") as f:
            content=f.read()
        relative_path = file.relative_to(source_dir)
        files.append({
            "filename": str(relative_path).replace("\\", "/"),
            "content":content
        })
    return files
def chunk_text(text:str,chunk_size:int=500,chunk_overlap:int=50)->List[str]:
    """
    Découpe un texte en chunks de chunk_size caractères,
    avec un chevauchement de chunk_overlap caractères entre chunks consécutifs.
    """
    chunks= []
    i=0
    while i<len(text):
        chunk=text[i:i+chunk_size]
        chunks.append(chunk)
        i+=chunk_size-chunk_overlap

    return chunks

def build_chunks(documents:List[Dict],chunk_size:int,chunk_overlap:int)->List[Dict]:
    """
    Pour chaque document, applique chunk_text() et garde la métadonnée
    de la source (filename) avec chaque chunk.
    Retourne une Liste de dicts : {"chunk_id": ..., "source": ..., "text": ...}
    """
    chunks= []
    for doc in documents:
        filename=doc["filename"]
        content=doc["content"]

        text_chunks=chunk_text(content,chunk_size,chunk_overlap)

        for index,chunk in enumerate(text_chunks):
            chunks.append({
                "chunk_id":f"{filename}_{index}",
                "source":filename,
                "text":chunk  
                              })
    return chunks

if __name__=="__main__":
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    docs=load_markdown_files("data/raw")
    print(f"Documents chargés: {len (docs)}")

    chunks=build_chunks(docs,CHUNK_SIZE,CHUNK_OVERLAP)
    print(f"Chunks générés :{len (chunks)}")

    os.makedirs("data/processed",exist_ok=True)   
    with open("data/processed/chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)  
    print("Sauvegardé:data/processed/chunks.json")    
