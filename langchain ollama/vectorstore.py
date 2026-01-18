"""
Vector database storage and loading using ChromaDB.
Handles persistent storage of document embeddings.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from embedding import get_embedding_model


def get_vectorstore(
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Get or create a ChromaDB vector store.
    
    Args:
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store instance
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    
    return vectorstore


def add_documents(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of documents to add
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        Updated Chroma vector store instance
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    vectorstore.add_documents(documents)
    return vectorstore


def create_vectorstore_from_documents(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Create a new vector store from documents.
    
    Args:
        documents: List of documents to embed
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        New Chroma vector store instance
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    
    return vectorstore


def similarity_search(
    query: str,
    k: int = 5,
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[Document]:
    """
    Perform similarity search on the vector store.
    
    Args:
        query: Query string
        k: Number of results to return
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        List of similar documents
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    return vectorstore.similarity_search(query, k=k)


def similarity_search_with_score(
    query: str,
    k: int = 5,
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[tuple[Document, float]]:
    """
    Perform similarity search with relevance scores.
    
    Args:
        query: Query string
        k: Number of results to return
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        List of (document, score) tuples
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    return vectorstore.similarity_search_with_score(query, k=k)


def get_collection_stats(
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> dict:
    """
    Get statistics about the vector store collection.
    
    Args:
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection statistics
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    collection = vectorstore._collection
    
    return {
        "name": collection.name,
        "count": collection.count(),
    }


def delete_collection(
    persist_directory: str | Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> None:
    """
    Delete a collection from the vector store.
    
    Args:
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    vectorstore.delete_collection()


if __name__ == "__main__":
    # Test vector store
    stats = get_collection_stats()
    print(f"Collection: {stats['name']}")
    print(f"Document count: {stats['count']}")