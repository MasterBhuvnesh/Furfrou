# 📘 Light Novel AI Agent — System Design

## 🎯 Objective

Build a **LangChain + Ollama powered AI Agent** that can:

- Answer questions from a **10-volume light novel**
- Use **retrieval-augmented generation (RAG)**
- Support **dynamic volume uploads**
- Use **DeepSeek R1 via Ollama**
- Store embeddings in a **local vector database**

---

## 🧠 System Overview

The system will process novels, embed them, store them in a vector database, and allow an AI agent to retrieve relevant content when answering user queries.

---

## 🏗️ High-Level Architecture

```
Volumes (PDF/Text)
        ↓
Document Loaders
        ↓
Text Splitters
        ↓
Embedding Model
        ↓
Vector Store (ChromaDB)
        ↓
Retriever
        ↓
Agent (DeepSeek R1 via Ollama)
        ↓
Tools + Prompting + Memory
        ↓
Final Answer
```

---

## 📂 Project Structure

```
project/
│
├── agent.py            # Main agent logic
├── embedding.py        # Embedding model loader + logic
├── vectorstore.py      # Vector DB storage + loading
├── retriever.py        # Retrieval logic
├── tools.py            # Agent tools
├── prompting.py        # System prompt templates
├── memory.py           # Conversation memory
├── loaders.py          # Document loaders (PDF, TXT)
├── splitter.py         # Text splitting logic
├── ingest.py           # Data ingestion pipeline
├── config.py           # Central configuration
├── registry.json       # Tracks processed volumes
├── main.py             # User interaction interface
└── requirements.txt
```

---

## 🗄️ Vector Database Choice

**ChromaDB (Local)**

**Why:**

- Easy to use
- Persistent local storage
- Fully compatible with LangChain
- Efficient for novel-scale RAG

---

## 🔢 Embedding Model

Use one of the following (via Ollama):

| Model                | Purpose                             |
| -------------------- | ----------------------------------- |
| **nomic-embed-text** | Best general text embedding         |
| bge-large-en         | Alternative high-quality embeddings |
| mxbai-embed-large    | We are using this                   |

---

## 🤖 LLM Model

| Role     | Model                    |
| -------- | ------------------------ |
| Main LLM | **DeepSeek R1 (Ollama)** |

This powers reasoning, agent decisions, and response generation.

---

## 🧩 Data Ingestion Flow

### Purpose:

Convert novel volumes into vector embeddings.

### Steps:

```
1. Load document using loaders.py
2. Split into chunks using splitter.py
3. Generate embeddings using embedding.py
4. Store embeddings in ChromaDB via vectorstore.py
5. Update registry.json to mark processed volumes
```

---

## 📄 Registry Logic

A `registry.json` file tracks what volumes have already been embedded.

Example:

```json
{
  "volume_1.pdf": {
    "status": "embedded",
    "chunks": 245,
    "last_updated": "2026-01-18"
  }
}
```

### Upload Handling Logic

```
If uploaded file not in registry → Process + store + update registry
Else → Skip ingestion
```

---

## 🔍 Retrieval Flow

```
User Query → Embed Query → Similarity Search → Retrieve Relevant Chunks → Send to LLM
```

Handled by:

- `retriever.py`
- `vectorstore.py`
- `embedding.py`

---

## 🧰 Agent Tools

Defined in `tools.py`

| Tool           | Function                              |
| -------------- | ------------------------------------- |
| Retriever Tool | Fetch relevant novel context          |
| Character Tool | Extract character-related passages    |
| Volume Finder  | Identify which volume contains events |
| Summarizer     | Condense long retrieved text          |
| Timeline Tool  | Understand story progression          |

---

## 🧠 Agent Capabilities

The agent will:

- Use RAG to fetch relevant text
- Decide which tool to use for a question
- Maintain memory for conversations
- Answer contextually based on all stored volumes

---

## 💬 Prompting Strategy

Handled by `prompting.py`

Includes:

- Agent identity
- Novel context awareness
- Tool usage guidelines
- Response formatting instructions

---

## 🧠 Memory

Handled by `memory.py`

Supports:

- Chat history buffer
- Context retention across user sessions

---

## 🚀 Deployment Options

- CLI via `main.py`
- Web API via FastAPI (future upgrade)
- Extendable to serverless or cloud

---

## 🧩 Final Workflow

```
Load & Ingest Volumes → Store in Vector DB → Start Agent → Ask Questions → Agent Retrieves + Answers → Add New Volumes Anytime
```

---

## ✅ System Ready For Implementation

This design supports:

- Scalability
- Clean modular coding
- Future multi-agent or API upgrades
