import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEncoder:
    """
    Singleton wrapper class around SentenceTransformer to ensure the model is loaded exactly once.
    """
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

def get_sentence_embeddings(texts):
    """
    Computes semantic vector embeddings using a local SentenceTransformer model.
    """
    model = LocalEncoder.get_model()
    return model.encode(texts, convert_to_numpy=True)

def compute_cosine_similarity(vec_a, vec_b):
    """
    Computes cosine similarity between two numeric vectors.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)
