"""Optional semantic layer. Falls back gracefully if sentence-transformers is missing."""
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def available():
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False

def encode(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    v = _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)
