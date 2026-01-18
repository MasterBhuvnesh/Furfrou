"""
Document loaders for PDF and TXT files.
Handles loading of light novel volumes.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a single PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects (one per page)
    """
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source_file"] = Path(file_path).name
        doc.metadata["file_type"] = "pdf"
    
    return documents


def load_text(file_path: str | Path) -> list[Document]:
    """
    Load a single text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Document objects
    """
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source_file"] = Path(file_path).name
        doc.metadata["file_type"] = "txt"
    
    return documents


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a document based on its file extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of Document objects
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == ".pdf":
        return load_pdf(path)
    elif extension in [".txt", ".md"]:
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_directory(directory_path: str | Path, glob_pattern: str = "**/*.pdf") -> list[Document]:
    """
    Load all documents from a directory matching the pattern.
    
    Args:
        directory_path: Path to the directory
        glob_pattern: Glob pattern for file matching
        
    Returns:
        List of Document objects from all matching files
    """
    loader = DirectoryLoader(
        str(directory_path),
        glob=glob_pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    return loader.load()


def get_all_pdf_files(directory: str | Path) -> list[Path]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Path objects for PDF files
    """
    directory = Path(directory)
    return list(directory.glob("*.pdf"))


def get_all_text_files(directory: str | Path) -> list[Path]:
    """
    Get all text files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Path objects for text files
    """
    directory = Path(directory)
    txt_files = list(directory.glob("*.txt"))
    md_files = list(directory.glob("*.md"))
    return txt_files + md_files


if __name__ == "__main__":
    from config import PDF_DIR
    
    # List available PDFs
    pdfs = get_all_pdf_files(PDF_DIR)
    print(f"Found {len(pdfs)} PDF files in {PDF_DIR}")
    for pdf in pdfs:
        print(f"  - {pdf.name}")
