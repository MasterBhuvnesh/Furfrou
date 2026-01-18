"""
Main agent logic for the Light Novel AI Agent.
Implements RAG-based question answering with tool usage.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import (
    OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    LLM_NUM_CTX, AGENT_VERBOSE,
)
from tools import get_tools
from memory import ConversationMemory, get_session_memory
from prompting import AGENT_SYSTEM_PROMPT
from retriever import retrieve_with_context


def get_llm() -> ChatOllama:
    """Initialize and return the Ollama LLM."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )


class LightNovelAgent:
    """High-level interface for the Light Novel AI Agent using tool calling."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.llm = get_llm()
        self.tools = get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = get_session_memory(session_id)
    
    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        # Build messages with history
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        messages.extend(self.memory.get_messages())
        messages.append(HumanMessage(content=message))
        
        # Get response (may include tool calls)
        response = self.llm_with_tools.invoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        result = tool.invoke(tool_args)
                        tool_results.append(f"[{tool_name}]: {result}")
                        break
            
            # Get final response with tool results
            context = "\n\n".join(tool_results)
            followup = f"Based on the retrieved information:\n{context}\n\nProvide a helpful answer."
            messages.append(response)
            messages.append(HumanMessage(content=followup))
            final_response = self.llm.invoke(messages)
            answer = final_response.content
        else:
            answer = response.content
        
        # Update memory
        self.memory.add_user_message(message)
        self.memory.add_ai_message(answer)
        
        return answer
    
    def ask(self, question: str, use_rag: bool = True) -> str:
        """Ask a question with optional RAG context."""
        if use_rag:
            context = retrieve_with_context(question)
            enhanced = f"Context:\n{context}\n\nQuestion: {question}"
            return self.chat(enhanced)
        return self.chat(question)
    
    def clear_history(self) -> None:
        self.memory.clear()


class SimpleRAGChain:
    """A simpler RAG chain without tool complexity."""
    
    def __init__(self):
        self.llm = get_llm()
        self.memory = ConversationMemory()
    
    def query(self, question: str) -> str:
        """Query with RAG context."""
        context = retrieve_with_context(question)
        history = self.memory.get_formatted_history()
        
        prompt = f"""{AGENT_SYSTEM_PROMPT}

Context from Light Novels:
{context}

Conversation History:
{history}

User Question: {question}

Please provide a helpful answer based on the context above."""
        
        response = self.llm.invoke(prompt)
        self.memory.add_user_message(question)
        self.memory.add_ai_message(response.content)
        return response.content
    
    def clear_history(self) -> None:
        self.memory.clear()


if __name__ == "__main__":
    # Quick test
    chain = SimpleRAGChain()
    print(chain.query("Hello! What can you help me with?"))
