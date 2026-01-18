"""
Agent tools for the Light Novel AI Agent.
Defines retrieval and analysis tools.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K
from retriever import retrieve_documents, retrieve_with_context
from prompting import (
    RETRIEVER_TOOL_DESCRIPTION,
    CHARACTER_TOOL_DESCRIPTION,
    VOLUME_FINDER_DESCRIPTION,
    SUMMARIZER_DESCRIPTION,
    TIMELINE_TOOL_DESCRIPTION,
)


@tool
def search_novels(query: str) -> str:
    """
    Search the light novel database for relevant passages.
    
    Use this tool to find information about any topic in the light novels.
    """
    documents = retrieve_documents(query, k=RETRIEVER_K)
    
    if not documents:
        return "No relevant passages found in the light novel database."
    
    results = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"[Result {i} - {source}, Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)


@tool
def search_character(character_query: str) -> str:
    """
    Search for character-specific information in the light novels.
    
    Use this tool when looking for information about specific characters,
    their descriptions, relationships, or development.
    """
    # Enhanced query for character-specific search
    enhanced_query = f"character {character_query} description appearance personality"
    documents = retrieve_documents(enhanced_query, k=RETRIEVER_K)
    
    if not documents:
        return f"No information found about '{character_query}' in the light novel database."
    
    results = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"[Character Info {i} - {source}, Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)


@tool
def find_volume(event_description: str) -> str:
    """
    Find which volume contains a specific event or information.
    
    Use this tool to identify which volume an event occurs in.
    """
    documents = retrieve_documents(event_description, k=3)
    
    if not documents:
        return "Could not locate this event in any of the available volumes."
    
    # Extract volume information
    volumes_found = set()
    results = []
    
    for doc in documents:
        source = doc.metadata.get("source_file", "Unknown")
        volumes_found.add(source)
        page = doc.metadata.get("page", "N/A")
        results.append(f"Found in: {source}, Page {page}")
        results.append(f"Context: {doc.page_content[:300]}...")
    
    header = f"This event appears in: {', '.join(volumes_found)}\n\n"
    return header + "\n\n".join(results)


@tool
def summarize_content(text_to_summarize: str) -> str:
    """
    Summarize a long passage of text into a concise summary.
    
    Use this tool when you need to condense retrieved content.
    """
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,  # Lower temperature for more focused summaries
    )
    
    summary_prompt = f"""Please summarize the following text concisely, 
    preserving the key information and events:

    {text_to_summarize}

    Summary:"""
    
    response = llm.invoke(summary_prompt)
    return response.content


@tool
def analyze_timeline(timeline_query: str) -> str:
    """
    Analyze the timeline and story progression for a specific topic.
    
    Use this tool to understand when events happen relative to each other.
    """
    # Search for timeline-related content
    documents = retrieve_documents(timeline_query, k=RETRIEVER_K)
    
    if not documents:
        return "No timeline information found for this query."
    
    # Collect all relevant passages with their sources
    timeline_info = []
    for doc in documents:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        timeline_info.append({
            "source": source,
            "page": page,
            "content": doc.page_content
        })
    
    # Format the response
    formatted = "Timeline Analysis:\n\n"
    for i, info in enumerate(timeline_info, 1):
        formatted += f"{i}. [{info['source']}, Page {info['page']}]\n"
        formatted += f"   {info['content'][:200]}...\n\n"
    
    return formatted


def get_tools() -> list:
    """
    Get all available tools for the agent.
    
    Returns:
        List of tool instances
    """
    return [
        search_novels,
        search_character,
        find_volume,
        summarize_content,
        analyze_timeline,
    ]


def get_tool_descriptions() -> dict:
    """
    Get descriptions for all tools.
    
    Returns:
        Dictionary mapping tool names to descriptions
    """
    return {
        "search_novels": RETRIEVER_TOOL_DESCRIPTION,
        "search_character": CHARACTER_TOOL_DESCRIPTION,
        "find_volume": VOLUME_FINDER_DESCRIPTION,
        "summarize_content": SUMMARIZER_DESCRIPTION,
        "analyze_timeline": TIMELINE_TOOL_DESCRIPTION,
    }


if __name__ == "__main__":
    # List available tools
    tools = get_tools()
    print("Available Tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
