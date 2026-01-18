"""
Central configuration for the Light Novel AI Agent.
Contains all paths, model settings, and parameters.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Data directories
PDF_DIR = BASE_DIR / ".pdfs"
DOCS_DIR = BASE_DIR / ".docs"
CHROMA_DIR = BASE_DIR / ".chroma_db"

# Registry file for tracking processed volumes
REGISTRY_FILE = BASE_DIR / "registry.json"

# Ensure directories exist
PDF_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Model (DeepSeek R1 via Ollama)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:8b")
LLM_TEMPERATURE = 0.7
LLM_NUM_CTX = 8192  # Context window size

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# =============================================================================
# TEXT SPLITTING CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# =============================================================================
# RETRIEVER CONFIGURATION
# =============================================================================

# Number of documents to retrieve
RETRIEVER_K = 5

# Search type: "similarity", "mmr" (Maximal Marginal Relevance)
SEARCH_TYPE = "similarity"

# MMR specific settings (if using MMR)
MMR_FETCH_K = 20
MMR_LAMBDA_MULT = 0.5

# =============================================================================
# CHROMA CONFIGURATION
# =============================================================================

CHROMA_COLLECTION_NAME = "light_novel_collection"

# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================

# Maximum number of messages to keep in memory
MEMORY_MAX_MESSAGES = 20

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Maximum iterations for the agent
AGENT_MAX_ITERATIONS = 10

# Verbose output
AGENT_VERBOSE = True
