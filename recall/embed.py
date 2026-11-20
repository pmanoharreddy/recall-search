"""Optional semantic layer. Falls back gracefully if sentence-transformers is missing."""
from importlib.util import find_spec

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def available():
    """Check installation without loading the model runtime during keyword-only work."""
    return find_spec("sentence_transformers") is not None


def fragmented_identifier(text):
    """A lone ASCII identifier split mostly into tiny model tokens needs literal evidence."""
    if _model is None or len(text) < 6 or not text.isascii() or not text.isalnum():
        return False
    tokens = _model.tokenizer.tokenize(text)
    return len(tokens) >= 5 and len(tokens) / len(text) >= 0.6

def encode(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    v = _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)
