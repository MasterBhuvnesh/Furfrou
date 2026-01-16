import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";

const region = process.env.AWS_REGION || "us-east-1";
const modelId = process.env.BEDROCK_MODEL_ID;

if (!modelId) {
  console.error("ERROR: BEDROCK_MODEL_ID environment variable is not defined.");
  process.exit(1);
}

const client = new BedrockRuntimeClient({ region });

const userMessage = "Different types of activation functions used in neural networks";
const conversation = [
  {
    role: "user",
    content: [{ text: userMessage }],
  },
];

const command = new ConverseCommand({
  modelId,
  messages: conversation,
  inferenceConfig: { maxTokens: 512, temperature: 0.5, topP: 0.9 },
});

try {
  const response = await client.send(command);
  const responseText = response.output.message.content[0].text;
  console.log(responseText);
} catch (err) {
  console.error(`ERROR: Can't invoke '${modelId}' in region '${region}'.`);
  console.error(`Reason: ${err.message || err}`);
  process.exit(1);
}
