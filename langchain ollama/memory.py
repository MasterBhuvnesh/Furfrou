"""
Conversation memory management.
Handles chat history and context retention.
"""

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import MEMORY_MAX_MESSAGES


class ConversationMemory:
    """
    Manages conversation history with a maximum message limit.
    """
    
    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to retain
        """
        self.max_messages = max_messages
        self._history: InMemoryChatMessageHistory = InMemoryChatMessageHistory()
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to history."""
        self._history.add_user_message(message)
        self._trim_history()
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to history."""
        self._history.add_ai_message(message)
        self._trim_history()
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history."""
        self._history.add_message(message)
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Trim history to maximum message count."""
        messages = self._history.messages
        if len(messages) > self.max_messages:
            # Keep only the most recent messages
            self._history.messages = messages[-self.max_messages:]
    
    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in history."""
        return self._history.messages
    
    def get_history(self) -> BaseChatMessageHistory:
        """Get the underlying chat history object."""
        return self._history
    
    def clear(self) -> None:
        """Clear all messages from history."""
        self._history.clear()
    
    def get_formatted_history(self) -> str:
        """
        Get history formatted as a string.
        
        Returns:
            Formatted conversation history
        """
        messages = self.get_messages()
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
            else:
                formatted.append(f"{msg.type}: {msg.content}")
        
        return "\n".join(formatted)
    
    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self._history.messages)


# Session storage for multiple conversations
_sessions: dict[str, ConversationMemory] = {}


def get_session_memory(session_id: str = "default") -> ConversationMemory:
    """
    Get or create a conversation memory for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        ConversationMemory instance for the session
    """
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def clear_session(session_id: str = "default") -> None:
    """
    Clear a session's memory.
    
    Args:
        session_id: Session to clear
    """
    if session_id in _sessions:
        _sessions[session_id].clear()


def clear_all_sessions() -> None:
    """Clear all session memories."""
    _sessions.clear()


if __name__ == "__main__":
    # Test memory
    memory = get_session_memory("test")
    
    memory.add_user_message("Hello, can you tell me about the protagonist?")
    memory.add_ai_message("The protagonist is a brave adventurer who...")
    memory.add_user_message("What happens in chapter 2?")
    memory.add_ai_message("In chapter 2, the protagonist embarks on...")
    
    print("Conversation history:")
    print(memory.get_formatted_history())
    print(f"\nTotal messages: {len(memory)}")
