import os
import json
import re
from pathlib import Path 
from typing import List, Dict

RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
RE_INLINE_CODE = re.compile(r"`([^`]+)`")
RE_IMAGE = re.compile(r"!\[.*?\]\(.*?\)")
RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
RE_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")
RE_ADMONITION = re.compile(r"^!!!\s*\w+.*$", re.MULTILINE)  # ex: "!!! note"
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HR = re.compile(r"^-{3,}\s*$", re.MULTILINE)  # séparateurs "---"
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

def clean_text(text:str,keep_code_blocks:bool=False)->str:
    """
    Nettoie le texte brut d'un document Markdown avant chunking.
 
    Retire la mise en forme Markdown (titres, gras, liens, images, HTML)
    qui n'apporte aucune information sémantique et dégraderait la qualité
    des embeddings, tout en préservant le sens du texte.
 
    Args:
        text: contenu brut du fichier Markdown.
        keep_code_blocks: si True, conserve les blocs de code ```...```
            (utile si on veut que le RAG puisse citer des exemples de code).
            Si False (défaut), les blocs de code sont supprimés car ils
            polluent l'embedding sémantique du texte explicatif.
 
    Returns:
        Texte nettoyé, prêt pour le chunking.
    """
    if not text:
        return""
    if not keep_code_blocks:
        text=RE_CODE_BLOCK.sub("",text)
    text=RE_INLINE_CODE.sub(r"\1",text)
    text=RE_IMAGE.sub("",text)
    text=RE_LINK.sub(r"\1",text)
     # 4. Titres Markdown (#, ##, ###...) -> on garde le texte du titre,
    #    on retire juste les symboles '#'
    text = RE_HEADER.sub("", text)
 
    # 5. Gras/italique (**texte**, *texte*, __texte__) -> texte brut
    text = RE_BOLD_ITALIC.sub(r"\2", text)
 
    # 6. Admonitions MkDocs (!!! note, !!! warning...) -> supprimées
    text = RE_ADMONITION.sub("", text)
 
    # 7. Balises HTML éventuelles (<div>, <br>...) -> supprimées
    text = RE_HTML_TAG.sub(" ", text)
 
    # 8. Séparateurs horizontaux (---) -> supprimés
    text = RE_HR.sub("", text)
 
    # 9. Normalisation des espaces : espaces/tabs multiples -> un seul
    text = RE_MULTI_SPACE.sub(" ", text)
    # lignes vides multiples -> une seule ligne vide (lisibilité)
    text = RE_MULTI_BLANK_LINES.sub("\n\n", text)
 
    return text.strip()

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

def build_chunks(documents:List[Dict],chunk_size:int,chunk_overlap:int,keep_code_blocks:bool=False)->List[Dict]:
    """
    Pour chaque document, applique chunk_text() et garde la métadonnée
    de la source (filename) avec chaque chunk.
    Retourne une Liste de dicts : {"chunk_id": ..., "source": ..., "text": ...}
    """
    chunks= []
    for doc in documents:
        filename=doc["filename"]
        cleaned_content=clean_text(doc["content"],keep_code_blocks=keep_code_blocks)

        if not cleaned_content:
            continue

        text_chunks=chunk_text(cleaned_content,chunk_size,chunk_overlap)

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
    KEEP_CODE_BLOCKS = False
    docs=load_markdown_files("data/raw")
    print(f"Documents chargés: {len (docs)}")

    chunks=build_chunks(docs,CHUNK_SIZE,CHUNK_OVERLAP, KEEP_CODE_BLOCKS)
    print(f"Chunks générés :{len (chunks)}")

    os.makedirs("data/processed",exist_ok=True)   
    with open("data/processed/chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)  
    print("Sauvegardé:data/processed/chunks.json")    
