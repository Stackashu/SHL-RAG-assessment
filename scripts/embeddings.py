import os
import json
import uuid
import dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --------------------------------------------------
# Paths & env
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

dotenv.load_dotenv(os.path.join(ROOT_DIR, ".env"))

DATA_FILE = os.path.join(ROOT_DIR, "data", "cleaned_catalog.json")
COLLECTION_NAME = "shl"
BATCH_SIZE = 50

# --------------------------------------------------
# Load LOCAL embedding model (NO API, NO LIMITS)
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Qdrant client
# --------------------------------------------------
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),      
    api_key=os.getenv("QDRANT_API_KEY"),  
    timeout=60
)

# --------------------------------------------------
# Extract only DESCRIPTION from text
# --------------------------------------------------
def extract_description(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "").strip()

    if "Description:" not in text:
        return ""

    # Take everything after 'Description:'
    after_desc = text.split("Description:", 1)[1]

    # Stop at first newline
    description = after_desc.split("\n", 1)[0]

    return description.strip()

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Load data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Embed ONLY description-rich text
    texts = [
        extract_description(item.get("text", "")) or item.get("text", "")
        for item in data
    ]

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    vector_size = embeddings.shape[1]

    # --------------------------------------------------
    # SAFE collection recreation (works in ALL versions)
    # --------------------------------------------------
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # --------------------------------------------------
    # Prepare points
    # --------------------------------------------------
    points = []
    for i, vec in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "name": data[i].get("name", ""),
                    "url": data[i].get("url", ""),
                    "test_type": data[i].get("test_type", ""),
                    "duration": data[i].get("duration", ""),
                    "adaptive_support": data[i].get("adaptive_support", ""),
                    "remote_support": data[i].get("remote_support", ""),
                    "description": extract_description(data[i].get("text", "")),
                }
            )
        )

    # --------------------------------------------------
    # Batch upload
    # --------------------------------------------------
    total = len(points)
    for i in range(0, total, BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Uploaded {i + len(batch)} / {total}")

    print("✅ Local embeddings uploaded to Qdrant successfully")

# --------------------------------------------------
if __name__ == "__main__":
    main()


# this if for cleaning the embeddings 

# from qdrant_client import QdrantClient
# import os
# import dotenv

# dotenv.load_dotenv(".env")

# client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY")
# )

# client.delete_collection("shl")
# print("✅ Old collection deleted")
