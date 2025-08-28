import chromadb
from google.genai import types

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, genai_client, EMBEDDING_MODEL_ID

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def embed_document(text: str):
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    return result.embeddings[0].values

def embed_query(text: str):
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    return result.embeddings[0].values

def add_to_chroma(chunks):
    for i, chunk in enumerate(chunks):
        embedding = embed_document(chunk)
        doc_id = f"doc_{i}"
        metadata = {
            "source": "uploaded_file"
        }
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id] 
        )

# def add_to_chroma(chunks, username):
#     for i, chunk in enumerate(chunks):
#         embedding = embed_document(chunk)
#         doc_id = f"{username}_{i}"
#         metadata = {
#             "username": username,
#             "source": "uploaded_file"
#         }
#         collection.add(
#             documents=[chunk],
#             embeddings=[embedding],
#             metadatas=[metadata],
#             ids=[doc_id] 
#         )

# def query_chroma(query, top_k=2, username=None):
#     query_embedding = embed_query(query)
#     filters = {"username": username} if username else {}
#     results= collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         where=filters
#     )
#     return results['documents'][0]

def query_chroma(query, top_k=2, username=None):
    query_embedding = embed_query(query)
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }
    if username:
        query_args["where"] = {"username": {"$eq": username}}
    
    results = collection.query(**query_args)
    return results['documents'][0]
