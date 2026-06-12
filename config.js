// AWS Bedrock configuration
//
// SECURITY WARNING: Hardcoding AWS credentials in source code is a bad
// practice. Anyone with access to this file (or this repo, if committed)
// gets full access to whatever permissions this IAM user/role has.
// Replace the placeholders below with your real values, and make sure
// this file is NEVER committed to a public repository (add it to .gitignore).

module.exports = {
  region: "us-east-1", // e.g. "us-east-1"
  knowledgeBaseId: "TOBEFILLED", // e.g. "ABCD1234EF"
  modelArn: "anthropic.claude-3-haiku-20240307-v1:0",

  credentials: {
    accessKeyId: "TOBEFILLED",
    secretAccessKey: "TOBEFILLED",
    // sessionToken: "YOUR_SESSION_TOKEN", // only needed for temporary credentials
  },

  // PDF chat tab: chunks are embedded with Bedrock Titan and stored/searched in Pinecone.
  embeddingModelId: "amazon.titan-embed-text-v2:0",
  chunking: {
    chunkSize: 1000, // characters per chunk
    chunkOverlap: 100, // characters of overlap between consecutive chunks
  },
  pinecone: {
    apiKey: "TOBEFILLED",
    indexName: "TOBEFILLED",
  },
};
