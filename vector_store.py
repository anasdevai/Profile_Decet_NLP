import os
import warnings
from uuid import uuid4
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Profile")

_qdrant = None
_embedder_en = None
_embedder_multi = None

def get_qdrant_client():
    global _qdrant
    if _qdrant is None:
        try:
            # Use a short timeout for the cloud check to prevent hanging
            _qdrant = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY, 
                check_compatibility=False,
                timeout=10
            )
            # Try to check existence, if it fails or timeouts, fallback
            if not _qdrant.collection_exists(collection_name=COLLECTION_NAME):
                _qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Qdrant connection failed, falling back to memory: {e}")
            _qdrant = QdrantClient(":memory:", check_compatibility=False)
            if not _qdrant.collection_exists(collection_name=COLLECTION_NAME):
                _qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
    return _qdrant

def _init_embedder_en():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def _init_embedder_multi():
    return HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def get_embedder(lang_code: str):
    global _embedder_en, _embedder_multi
    if lang_code not in ["en"]:
        if _embedder_multi is None:
            _embedder_multi = _init_embedder_multi()
        return _embedder_multi
    if _embedder_en is None:
        _embedder_en = _init_embedder_en()
    return _embedder_en

# Silence known qdrant compatibility warning noise.
warnings.filterwarnings(
    "ignore",
    message="Failed to obtain server version. Unable to check client-server compatibility.*",
    category=UserWarning,
)

def ingest_document_chunks(chunks, profile: dict, lang_code: str, doc_name: str, org_id: str):
    client = get_qdrant_client()
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
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return [p.id for p in points]

def retrieve_similar_styles(query_text: str, lang_code: str = "en", genre_filter: str = None, top_k: int = 5):
    client = get_qdrant_client()
    emb = get_embedder(lang_code)
    vector = emb.embed_query(query_text)
    
    filters = None
    if genre_filter:
        filters = Filter(
            must=[
                FieldCondition(
                    key="genre",
                    match=MatchValue(value=genre_filter),
                )
            ]
        )
        
    if hasattr(client, "search"):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=filters,
            limit=top_k,
            with_payload=True
        )

    if hasattr(client, "query_points"):
        query_res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=filters,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(query_res, "points", None)
        if points is not None:
            return points
        if isinstance(query_res, list):
            return query_res
        return []

    return []
