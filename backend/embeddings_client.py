from sentence_transformers import SentenceTransformer

# Load model ONCE (important for performance)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    """
    Generate embedding for a single query text
    using the SAME model used during indexing.
    """
    model = get_model()
    embedding = model.encode([text])[0]
    return embedding.tolist()
