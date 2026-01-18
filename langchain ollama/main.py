"""
User interaction interface for the Light Novel AI Agent.
Provides CLI-based interaction with the agent.
"""

import sys
from pathlib import Path

from config import PDF_DIR
from agent import LightNovelAgent, SimpleRAGChain
from ingest import ingest_directory, ingest_file, get_ingestion_status


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("📚 Light Novel AI Agent")
    print("=" * 60)
    print("Ask questions about your light novels!")
    print("Commands: /help, /status, /ingest, /clear, /quit")
    print("=" * 60 + "\n")


def print_help():
    """Print help information."""
    print("""
Available Commands:
  /help       - Show this help message
  /status     - Show ingestion status
  /ingest     - Ingest all PDFs from the .pdfs folder
  /ingest <path> - Ingest a specific file
  /clear      - Clear conversation history
  /simple     - Switch to simple RAG mode (no tools)
  /agent      - Switch to full agent mode (with tools)
  /quit       - Exit the application

Just type your question to chat with the agent!
""")


def handle_command(command: str, agent, mode: list) -> bool:
    """Handle special commands. Returns False to quit."""
    cmd = command.lower().strip()
    
    if cmd == "/quit" or cmd == "/exit":
        print("Goodbye! 👋")
        return False
    
    elif cmd == "/help":
        print_help()
    
    elif cmd == "/status":
        status = get_ingestion_status()
        print(f"\n📊 Ingestion Status:")
        print(f"   Volumes: {status['volumes_processed']}")
        print(f"   Chunks: {status['total_chunks_in_db']}")
        if status['volumes']:
            print("   Files:")
            for vol, info in status['volumes'].items():
                print(f"     - {vol}: {info['chunks']} chunks")
        print()
    
    elif cmd == "/ingest":
        print("📥 Ingesting files from .pdfs folder...")
        ingest_directory(PDF_DIR)
        print("Done!")
    
    elif cmd.startswith("/ingest "):
        file_path = command[8:].strip()
        if Path(file_path).exists():
            print(f"📥 Ingesting {file_path}...")
            result = ingest_file(file_path)
            print(f"Result: {result['status']}")
        else:
            print(f"File not found: {file_path}")
    
    elif cmd == "/clear":
        agent.clear_history()
        print("🧹 Conversation history cleared.")
    
    elif cmd == "/simple":
        mode[0] = "simple"
        print("🔄 Switched to simple RAG mode.")
    
    elif cmd == "/agent":
        mode[0] = "agent"
        print("🔄 Switched to full agent mode.")
    
    else:
        print(f"Unknown command: {command}")
        print("Type /help for available commands.")
    
    return True


def main():
    """Main entry point for the CLI."""
    print_banner()
    
    # Check ingestion status
    status = get_ingestion_status()
    if status['total_chunks_in_db'] == 0:
        print("⚠️  No documents ingested yet!")
        print(f"   Put PDFs in: {PDF_DIR}")
        print("   Then run: /ingest\n")
    
    # Initialize agents
    agent = LightNovelAgent()
    simple_chain = SimpleRAGChain()
    mode = ["agent"]  # Using list for mutability
    
    print("Ready! Type your question or /help for commands.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                if not handle_command(user_input, agent, mode):
                    break
                continue
            
            # Get response based on mode
            print("\n🤔 Thinking...\n")
            
            if mode[0] == "simple":
                response = simple_chain.query(user_input)
            else:
                response = agent.chat(user_input)
            
            print(f"Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
