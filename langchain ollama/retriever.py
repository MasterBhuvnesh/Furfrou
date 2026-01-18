"""
Retrieval logic for fetching relevant documents.
Handles query embedding and similarity search.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import (
    RETRIEVER_K,
    SEARCH_TYPE,
    MMR_FETCH_K,
    MMR_LAMBDA_MULT,
)
from vectorstore import get_vectorstore


def get_retriever(
    search_type: str = SEARCH_TYPE,
    k: int = RETRIEVER_K,
    **kwargs,
) -> BaseRetriever:
    """
    Create and return a retriever from the vector store.
    
    Args:
        search_type: Type of search ("similarity" or "mmr")
        k: Number of documents to retrieve
        **kwargs: Additional arguments for the retriever
        
    Returns:
        Configured retriever instance
    """
    vectorstore = get_vectorstore()
    
    search_kwargs = {"k": k}
    
    if search_type == "mmr":
        search_kwargs["fetch_k"] = kwargs.get("fetch_k", MMR_FETCH_K)
        search_kwargs["lambda_mult"] = kwargs.get("lambda_mult", MMR_LAMBDA_MULT)
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    
    return retriever


def retrieve_documents(
    query: str,
    k: int = RETRIEVER_K,
    search_type: str = SEARCH_TYPE,
) -> list[Document]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: User query string
        k: Number of documents to retrieve
        search_type: Type of search ("similarity" or "mmr")
        
    Returns:
        List of relevant documents
    """
    retriever = get_retriever(search_type=search_type, k=k)
    return retriever.invoke(query)


def retrieve_with_context(
    query: str,
    k: int = RETRIEVER_K,
) -> str:
    """
    Retrieve documents and format them as context string.
    
    Args:
        query: User query string
        k: Number of documents to retrieve
        
    Returns:
        Formatted context string from retrieved documents
    """
    documents = retrieve_documents(query, k=k)
    
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def retrieve_by_volume(
    query: str,
    volume_name: str,
    k: int = RETRIEVER_K,
) -> list[Document]:
    """
    Retrieve documents filtered by volume name.
    
    Args:
        query: User query string
        volume_name: Name of the volume to filter by
        k: Number of documents to retrieve
        
    Returns:
        List of relevant documents from the specified volume
    """
    vectorstore = get_vectorstore()
    
    # Use filter to limit to specific volume
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={"source_file": volume_name},
    )
    
    return results


if __name__ == "__main__":
    # Test retrieval
    test_query = "What happens in the first chapter?"
    print(f"Query: {test_query}")
    print("-" * 50)
    
    docs = retrieve_documents(test_query, k=3)
    if docs:
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
    else:
        print("No documents found. Make sure to ingest some volumes first.")
