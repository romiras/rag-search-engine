from typing import List, Union
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        # Load the sentence transformer model directly
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.model.encode(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms


if __name__ == "__main__":
    # POC Test
    embedder = Embedder()
    test_texts = ["Hello world", "Artificial Intelligence is cool"]
    vecs = embedder.embed(test_texts)
    print(f"Vectors shape: {vecs.shape}")
    print(f"Vector 0 sample (first 5): {vecs[0][:5]}")
    print(f"Vector 0 norm: {np.linalg.norm(vecs[0])}")
