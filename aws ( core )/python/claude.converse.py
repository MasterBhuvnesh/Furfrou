# Use the Conversation API to send a text message to Anthropic Claude.

import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
region = os.getenv("AWS_REGION")
model_id = os.getenv("BEDROCK_MODEL_ID")

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name=region)

# Start a conversation with the user message.
user_message = "Whats Machine Learning?.Explain in 10 sentence."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

# Inference Configuration 
# maxTokens   → max tokens to generate      [ How Long ]
# temperature → randomness                  [ How Random ]
# topP        → probability cutoff            [ How wide the choices]

#  Factual / Q&A     : {"maxTokens": 300, "temperature": 0.2, "topP": 0.9}
#  Chat / Assistant  : {"maxTokens": 512, "temperature": 0.5, "topP": 0.9}
#  Creative writing  : {"maxTokens": 800, "temperature": 0.8, "topP": 0.95}


try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    # tokens_used = response["usage"]["totalTokens"]
    print(response_text)
    # print(f"Tokens used: {tokens_used}")

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Source : https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_Converse_AnthropicClaude_section.html