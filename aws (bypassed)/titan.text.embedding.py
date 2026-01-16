# Generate and print an embedding with Amazon Titan Text Embeddings V2.

import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
region = os.getenv("AWS_REGION")
model_id = os.getenv("BEDROCK_MODEL_ID")

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name=region)

# The text to convert to an embedding.
input_text = "Whats Machine Learning?.Explain in 1 sentence."

# Create the request for the model.
native_request = {"inputText": input_text}

# Convert the native request to JSON.
request = json.dumps(native_request)

# Invoke the model with the request.
response = client.invoke_model(modelId=model_id, body=request)

# Decode the model's native response body.
model_response = json.loads(response["body"].read())

# Extract and print the generated embedding and the input text token count.
embedding = model_response["embedding"]
input_token_count = model_response["inputTextTokenCount"]

print("\nYour input:")
print(input_text)
print(f"Number of input tokens: {input_token_count}")
print(f"Size of the generated embedding: {len(embedding)}")
print("Embedding:")
print(embedding)

# Source : https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanTextEmbeddings_section.html