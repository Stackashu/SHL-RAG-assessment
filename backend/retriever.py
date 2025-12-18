import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import dotenv

dotenv.load_dotenv() 

COLLECTION_NAME = "shl"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

def search(query: str, top_k=5):
    vector = model.encode(query).tolist()

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )

    # 🔥 IMPORTANT: return FULL payload (including text)
    return [hit.payload for hit in hits]
