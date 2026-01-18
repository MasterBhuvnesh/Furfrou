"""
Text splitting logic for document chunking.
Splits documents into smaller chunks for embedding.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create and return a text splitter instance.
    
    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    return splitter


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of split Document objects
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    split_docs = splitter.split_documents(documents)
    
    # Add chunk index metadata
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_index"] = i
    
    return split_docs


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split a text string into smaller chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)


if __name__ == "__main__":
    # Test splitting
    test_text = """
    Chapter 1: The Beginning
    
    It was a dark and stormy night. The protagonist stood at the edge of the cliff,
    looking out over the vast ocean below. The waves crashed against the rocks with
    tremendous force, sending spray high into the air.
    
    "This is where it all begins," she whispered to herself.
    
    Chapter 2: The Journey
    
    The next morning brought clear skies and a fresh breeze. Our hero set out on
    the long road ahead, not knowing what adventures awaited. The path wound through
    dense forests and over rolling hills.
    """
    
    chunks = split_text(test_text, chunk_size=200, chunk_overlap=50)
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
