import os
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "style_profiles")

# Initialize Embeddings
embedder = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

multilingual_embedder = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def get_embedder(lang_code: str):
    return multilingual_embedder if lang_code not in ["en"] else embedder

# Initialize Qdrant Client (in memory if url is localhost and not running)
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # Check if collection exists, if not create it
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
except Exception as e:
    # Fallback to in-memory Qdrant for development
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def ingest_document_chunks(chunks, profile: dict, lang_code: str, doc_name: str, org_id: str):
    points = []
    emb = get_embedder(lang_code)
    
    for chunk in chunks:
        vector = emb.embed_query(chunk.content)
        payload = {
            **profile,
            "chunk_text": chunk.content,
            "section_type": chunk.section_type,
            "doc_name": doc_name,
            "org_id": org_id
        }
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload
            )
        )
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return [p.id for p in points]

def retrieve_similar_styles(query_text: str, lang_code: str = "en", genre_filter: str = None, top_k: int = 5):
    emb = get_embedder(lang_code)
    vector = emb.embed_query(query_text)
    
    filters = None
    if genre_filter:
        filters = {
            "must": [
                {
                    "key": "genre",
                    "match": {"value": genre_filter}
                }
            ]
        }
        
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=filters,
        limit=top_k,
        with_payload=True
    )
    return results
