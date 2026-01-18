"""
Data ingestion pipeline for light novel volumes.
Handles loading, splitting, embedding, and storing documents.
"""

import json
from datetime import datetime
from pathlib import Path

from config import PDF_DIR, REGISTRY_FILE
from loaders import load_document, get_all_pdf_files, get_all_text_files
from splitter import split_documents
from vectorstore import add_documents, get_collection_stats


def load_registry() -> dict:
    """
    Load the registry of processed volumes.
    
    Returns:
        Dictionary containing processed volume information
    """
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict) -> None:
    """
    Save the registry to file.
    
    Args:
        registry: Registry dictionary to save
    """
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def is_volume_processed(filename: str) -> bool:
    """
    Check if a volume has already been processed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if already processed, False otherwise
    """
    registry = load_registry()
    return filename in registry and registry[filename].get("status") == "embedded"


def ingest_file(file_path: str | Path, force: bool = False) -> dict:
    """
    Ingest a single file into the vector store.
    
    Args:
        file_path: Path to the file to ingest
        force: If True, re-ingest even if already processed
        
    Returns:
        Dictionary with ingestion results
    """
    file_path = Path(file_path)
    filename = file_path.name
    
    # Check if already processed
    if not force and is_volume_processed(filename):
        return {
            "filename": filename,
            "status": "skipped",
            "message": "Already processed. Use force=True to re-ingest."
        }
    
    try:
        # Load the document
        print(f"Loading {filename}...")
        documents = load_document(file_path)
        print(f"  Loaded {len(documents)} pages/sections")
        
        # Split into chunks
        print("  Splitting into chunks...")
        chunks = split_documents(documents)
        print(f"  Created {len(chunks)} chunks")
        
        # Add to vector store
        print("  Embedding and storing...")
        add_documents(chunks)
        print("  Done!")
        
        # Update registry
        registry = load_registry()
        registry[filename] = {
            "status": "embedded",
            "chunks": len(chunks),
            "pages": len(documents),
            "last_updated": datetime.now().isoformat(),
            "file_path": str(file_path),
        }
        save_registry(registry)
        
        return {
            "filename": filename,
            "status": "success",
            "chunks": len(chunks),
            "pages": len(documents),
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "message": str(e),
        }


def ingest_directory(
    directory: str | Path = PDF_DIR,
    force: bool = False,
) -> list[dict]:
    """
    Ingest all files from a directory.
    
    Args:
        directory: Directory containing files to ingest
        force: If True, re-ingest all files
        
    Returns:
        List of ingestion results for each file
    """
    directory = Path(directory)
    results = []
    
    # Get all supported files
    pdf_files = get_all_pdf_files(directory)
    text_files = get_all_text_files(directory)
    all_files = pdf_files + text_files
    
    if not all_files:
        print(f"No files found in {directory}")
        return results
    
    print(f"Found {len(all_files)} files to process")
    print("-" * 50)
    
    for file_path in all_files:
        result = ingest_file(file_path, force=force)
        results.append(result)
        
        if result["status"] == "success":
            print(f"✓ {result['filename']}: {result['chunks']} chunks")
        elif result["status"] == "skipped":
            print(f"○ {result['filename']}: Skipped (already processed)")
        else:
            print(f"✗ {result['filename']}: Error - {result.get('message', 'Unknown')}")
    
    return results


def get_ingestion_status() -> dict:
    """
    Get the current ingestion status.
    
    Returns:
        Dictionary with status information
    """
    registry = load_registry()
    stats = get_collection_stats()
    
    return {
        "volumes_processed": len(registry),
        "total_chunks_in_db": stats.get("count", 0),
        "collection_name": stats.get("name", "Unknown"),
        "volumes": registry,
    }


def clear_and_reingest(directory: str | Path = PDF_DIR) -> list[dict]:
    """
    Clear the registry and re-ingest all files.
    
    Args:
        directory: Directory containing files to ingest
        
    Returns:
        List of ingestion results
    """
    # Clear registry
    save_registry({})
    
    # Re-ingest all
    return ingest_directory(directory, force=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            status = get_ingestion_status()
            print("Ingestion Status:")
            print(f"  Volumes processed: {status['volumes_processed']}")
            print(f"  Total chunks in DB: {status['total_chunks_in_db']}")
            print(f"  Collection: {status['collection_name']}")
            print("\nProcessed volumes:")
            for vol, info in status['volumes'].items():
                print(f"  - {vol}: {info['chunks']} chunks, {info['pages']} pages")
        
        elif sys.argv[1] == "--reingest":
            print("Clearing and re-ingesting all files...")
            results = clear_and_reingest()
            print(f"\nProcessed {len(results)} files")
        
        elif sys.argv[1] == "--file" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            result = ingest_file(file_path)
            print(f"Result: {result}")
        
        else:
            print("Usage:")
            print("  python ingest.py                    # Ingest all files in PDF_DIR")
            print("  python ingest.py --status           # Show ingestion status")
            print("  python ingest.py --reingest         # Clear and re-ingest all")
            print("  python ingest.py --file <path>      # Ingest a specific file")
    else:
        # Default: ingest all files
        results = ingest_directory()
        print(f"\nTotal: {len(results)} files processed")
