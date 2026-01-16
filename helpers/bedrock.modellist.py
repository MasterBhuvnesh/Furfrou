import boto3

# Test Bedrock connectivity
bedrock = boto3.client('bedrock', region_name='us-east-1')

try:
    # List available models
    response = bedrock.list_foundation_models()
    print("Available models:", len(response['modelSummaries']))
    for model in response['modelSummaries']:
        print(f"- {model['modelId']}")
except Exception as e:
    print(f"Error: {e}")
