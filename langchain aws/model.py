# Install required packages first:
# pip install langchain langchain-community boto3

# from langchain_community.chat_models import BedrockChat # Its deprecated
from langchain_aws import ChatBedrock
from langchain.messages  import HumanMessage, SystemMessage

# Initialize the Bedrock Chat model with Claude Sonnet 4
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",  # or your preferred region
    model_kwargs={
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

# Simple text generation
def simple_chat():
    print("=== Simple Chat Example ===")
    
    # Single message
    response = llm.invoke("What are the benefits of using AWS cloud services?")
    print("Response:", response.content)
    print()

# Chat with system message
def chat_with_system_message():
    print("=== Chat with System Message ===")
    
    messages = [
        SystemMessage(content="You are a helpful AWS cloud architect assistant."),
        HumanMessage(content="How would you design a scalable web application on AWS?")
    ]
    
    response = llm.invoke(messages)
    print("Response:", response.content)
    print()

# Interactive chat function
def interactive_chat():
    print("=== Interactive Chat (type 'quit' to exit) ===")
    
    conversation_history = [
        SystemMessage(content="You are a helpful AI assistant. Be concise and helpful.")
    ]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Get response
        response = llm.invoke(conversation_history)
        print(f"Assistant: {response.content}")
        
        # Add assistant response to history
        conversation_history.append(response)

# Run examples
if __name__ == "__main__":
    try:
        simple_chat()
        chat_with_system_message()
        interactive_chat()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Proper AWS credentials configured")
        print("2. Access to the Claude Sonnet 4 model in Bedrock")
        print("3. Correct IAM permissions")
