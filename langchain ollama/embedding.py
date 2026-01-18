"""
Embedding model loader and logic.
Uses Ollama embeddings with mxbai-embed-large model.
"""

from langchain_ollama import OllamaEmbeddings

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embedding_model() -> OllamaEmbeddings:
    """
    Initialize and return the Ollama embedding model.
    
    Returns:
        OllamaEmbeddings: Configured embedding model instance
    """
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    return embeddings


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    embeddings = get_embedding_model()
    return embeddings.embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple text documents.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = get_embedding_model()
    return embeddings.embed_documents(texts)


if __name__ == "__main__":
    # Test embedding
    test_text = "This is a test sentence for embedding."
    embedding = embed_text(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
