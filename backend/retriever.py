import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss.index"
META_PATH = "data/metadata.pkl"

# Lazy-loaded globals
_model = None
_index = None
_metadata = None

SOFT_SKILL_KEYWORDS = [
    "collaborate", "communication", "stakeholder",
    "personality", "behavior", "team", "leadership"
]

TECH_SKILL_KEYWORDS = [
    "java", "python", "sql", "javascript",
    "coding", "developer", "programming"
]


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_index():
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    return _index


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)
    return _metadata


def detect_query_intent(query: str):
    q = query.lower()
    soft = any(k in q for k in SOFT_SKILL_KEYWORDS)
    tech = any(k in q for k in TECH_SKILL_KEYWORDS)
    return soft, tech


def search(query: str, top_k=10):
    model = get_model()
    index = get_index()
    metadata = get_metadata()

    # Embed query
    q_vec = model.encode([query]).astype("float32")

    # Search FAISS
    D, I = index.search(q_vec, 8)
    candidates = [metadata[i] for i in I[0]]

    wants_soft, wants_tech = detect_query_intent(query)

    final = []
    soft_count = 0
    tech_count = 0

    for item in candidates:
        if len(final) >= top_k:
            break

        t = item.get("test_type")

        if t == "P" and wants_soft and soft_count < 3:
            final.append(item)
            soft_count += 1
        elif t == "K" and wants_tech and tech_count < 5:
            final.append(item)
            tech_count += 1
        elif not wants_soft and not wants_tech:
            final.append(item)

    # Fallback
    if len(final) < 5:
        for item in candidates:
            if item not in final:
                final.append(item)
            if len(final) >= 5:
                break

    return final
