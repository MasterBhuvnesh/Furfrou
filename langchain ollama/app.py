"""
Light Novel AI Agent - Single File Version
==============================================

A simple RAG (Retrieval-Augmented Generation) chatbot that can answer
questions about light novels using LangChain and Ollama.

How it works:
1. Load PDF files from the .pdfs folder
2. Split them into smaller chunks
3. Convert chunks to embeddings and store in ChromaDB
4. When you ask a question, find relevant chunks
5. Send those chunks + your question to the LLM for an answer

Usage:
    python app.py                  # Start the chatbot
    python app.py --ingest         # Ingest PDFs first, then chat
    python app.py --ingest-only    # Only ingest PDFs, don't chat
    python app.py --status         # Show ingestion status
"""

import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.resolve()
PDF_DIR = BASE_DIR / ".pdfs"           # Put your PDFs here
CHROMA_DIR = BASE_DIR / ".chroma_db"   # Vector database storage
REGISTRY_FILE = BASE_DIR / "registry.json"  # Tracks processed files

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"          # Main chat model
EMBEDDING_MODEL = "mxbai-embed-large"  # Embedding model

# Chunking settings (how to split documents)
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks

# Retrieval settings
NUM_RESULTS = 5        # Number of chunks to retrieve per query

# =============================================================================
# IMPORTS (all LangChain components we need)
# =============================================================================

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage


# =============================================================================
# REGISTRY - Track which PDFs have been processed
# =============================================================================

def load_registry() -> dict:
    """Load the registry of processed files."""
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


def save_registry(registry: dict) -> None:
    """Save the registry to disk."""
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def is_processed(filename: str) -> bool:
    """Check if a file has already been processed."""
    registry = load_registry()
    return filename in registry


# =============================================================================
# EMBEDDINGS - Convert text to vectors
# =============================================================================

def get_embeddings():
    """
    Create an embedding model.
    This converts text into numerical vectors for similarity search.
    """
    return OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL,
    )


# =============================================================================
# VECTOR STORE - Store and search embeddings
# =============================================================================

def get_vectorstore():
    """
    Get or create the ChromaDB vector store.
    This is where we store all the document embeddings.
    """
    return Chroma(
        collection_name="light_novels",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# =============================================================================
# DOCUMENT LOADING & PROCESSING
# =============================================================================

def load_pdf(file_path: Path) -> list:
    """
    Load a PDF file and return its pages as documents.
    Each page becomes a separate document with metadata.
    """
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    
    # Add source filename to metadata
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
    
    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks.
    
    Why chunk? Because:
    - LLMs have token limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't lose context at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
    )
    return splitter.split_documents(documents)


def ingest_file(file_path: Path) -> dict:
    """
    Process a single PDF file:
    1. Load it
    2. Split into chunks
    3. Store in vector database
    4. Update registry
    """
    filename = file_path.name
    
    # Skip if already processed
    if is_processed(filename):
        return {"file": filename, "status": "skipped", "reason": "already processed"}
    
    try:
        print(f"Loading {filename}...")
        documents = load_pdf(file_path)
        
        print(f"Splitting into chunks...")
        chunks = split_documents(documents)
        
        print(f"Embedding and storing {len(chunks)} chunks...")
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        
        # Update registry
        registry = load_registry()
        registry[filename] = {
            "status": "embedded",
            "chunks": len(chunks),
            "pages": len(documents),
            "date": datetime.now().isoformat(),
        }
        save_registry(registry)
        
        print(f"Done! ({len(chunks)} chunks from {len(documents)} pages)")
        return {"file": filename, "status": "success", "chunks": len(chunks)}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"file": filename, "status": "error", "error": str(e)}


def ingest_all_pdfs():
    """Process all PDFs in the PDF directory."""
    PDF_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        print("Add your light novel PDFs there and run again!")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files\n")
    
    for pdf_file in sorted(pdf_files):
        ingest_file(pdf_file)
    
    print("\nIngestion complete!")


def show_status():
    """Show the current ingestion status."""
    registry = load_registry()
    vectorstore = get_vectorstore()
    
    print("\nIngestion Status")
    print("=" * 50)
    print(f"Total volumes: {len(registry)}")
    print(f"Total chunks in database: {vectorstore._collection.count()}")
    print()
    
    if registry:
        print("Processed files:")
        for filename, info in registry.items():
            print(f"{filename}")
            print(f"Chunks: {info.get('chunks', '?')}, Pages: {info.get('pages', '?')}")
    else:
        print("No files processed yet. Run with --ingest to start.")
    print()


# =============================================================================
# RETRIEVAL - Find relevant chunks for a query
# =============================================================================

def retrieve_context(query: str, k: int = NUM_RESULTS) -> str:
    """
    Find the most relevant chunks for a query.
    Returns them formatted as context for the LLM.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    
    if not results:
        return "No relevant information found in the database."
    
    # Format results with source info
    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LLM - The AI that generates answers
# =============================================================================

def get_llm():
    """Create the chat LLM."""
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=LLM_MODEL,
        temperature=0.7,
    )


# The system prompt - tells the AI how to behave
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about light novels.

IMPORTANT RULES:
1. Base your answers ONLY on the provided context from the novels
2. If the context doesn't contain the answer, say "I couldn't find this in the available volumes"
3. Cite which volume/page the information comes from when possible
4. Be accurate - don't make up information not in the context
5. Be friendly and engaging when discussing the story

You're chatting with a fan who wants to know more about the novels!"""


# =============================================================================
# CHAT - The main conversation loop
# =============================================================================

class NovelChatbot:
    """
    A simple chatbot that answers questions about light novels.
    Uses RAG (Retrieval-Augmented Generation) for accurate answers.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.history = []  # Store conversation history
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        
        Steps:
        1. Retrieve relevant context from the vector store
        2. Build a prompt with context + history + question
        3. Send to LLM and get response
        4. Save to history and return
        """
        # Step 1: Get relevant context
        context = retrieve_context(user_message)
        
        # Step 2: Build the prompt
        prompt = f"""{SYSTEM_PROMPT}

## Context from Light Novels:
{context}

## Conversation History:
{self._format_history()}

## User's Question:
{user_message}

Please provide a helpful answer based on the context above."""

        # Step 3: Get LLM response
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Step 4: Update history (keep last 10 exchanges)
        self.history.append({"user": user_message, "assistant": answer})
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        return answer
    
    def _format_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.history:
            return "No previous conversation."
        
        parts = []
        for exchange in self.history[-5:]:  # Last 5 exchanges
            parts.append(f"User: {exchange['user']}")
            parts.append(f"Assistant: {exchange['assistant'][:200]}...")
        return "\n".join(parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []


def run_chat():
    """Run the interactive chat loop."""
    print("\n" + "=" * 60)
    print("Light Novel AI Chatbot")
    print("=" * 60)
    print("Ask me anything about your light novels!")
    print("Commands: /clear (clear history), /status, /quit")
    print("=" * 60 + "\n")
    
    # Check if we have any data
    registry = load_registry()
    if not registry:
        print("No novels ingested yet!")
        print(f"Put PDFs in: {PDF_DIR}")
        print("Run: python app.py --ingest\n")
    
    chatbot = NovelChatbot()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break
            elif user_input.lower() == "/clear":
                chatbot.clear_history()
                print("History cleared!")
                continue
            elif user_input.lower() == "/status":
                show_status()
                continue
            
            # Get response
            print("\nThinking...\n")
            response = chatbot.chat(user_input)
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# =============================================================================
# MAIN - Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Ensure directories exist
    PDF_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == "--ingest":
            # Ingest then chat
            ingest_all_pdfs()
            run_chat()
        
        elif arg == "--ingest-only":
            # Just ingest, don't chat
            ingest_all_pdfs()
        
        elif arg == "--status":
            # Show status
            show_status()
        
        elif arg == "--help":
            print(__doc__)
        
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
    
    else:
        # Default: just run chat
        run_chat()
