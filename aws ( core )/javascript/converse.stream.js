import {
  BedrockRuntimeClient,
  ConverseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import dotenv from "dotenv";

dotenv.config();

const client = new BedrockRuntimeClient({
  region: process.env.AWS_REGION,
  requestHandler: new NodeHttpHandler({
    requestTimeout: 30000,
    httpAgent: { maxSockets: 50 },
  }),
});

const modelId = process.env.BEDROCK_MODEL_ID;

const userMessage =
  "Whats Machine Learning?.Explain in 10 sentence.";
const conversation = [
  {
    role: "user",
    content: [{ text: userMessage }],
  },
];

const command = new ConverseStreamCommand({
  modelId,
  messages: conversation,
  inferenceConfig: { maxTokens: 512, temperature: 0.5, topP: 0.9 },
});

try {
  const response = await client.send(command);

  for await (const item of response.stream) {
    if (item.contentBlockDelta) {
      process.stdout.write(item.contentBlockDelta.delta?.text);
    }
  }
} catch (err) {
  console.log(`ERROR: Can't invoke '${modelId}'. Reason: ${err}`);
  process.exit(1);
}
