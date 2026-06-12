const crypto = require("node:crypto");
const express = require("express");
const multer = require("multer");
const { PDFParse } = require("pdf-parse");
const { Pinecone } = require("@pinecone-database/pinecone");
const {
  BedrockRuntimeClient,
  InvokeModelCommand,
  ConverseCommand,
} = require("@aws-sdk/client-bedrock-runtime");

const config = require("./config");

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const bedrockRuntime = new BedrockRuntimeClient({
  region: config.region,
  credentials: config.credentials,
});

const pinecone = new Pinecone({ apiKey: config.pinecone.apiKey });
const pineconeIndex = pinecone.index({ name: config.pinecone.indexName });

// Renders a detected table as a markdown table so it survives chunking
// and gives the LLM clear row/column structure instead of jumbled text.
function tableToMarkdown(table) {
  if (!table || table.length === 0) return "";

  const rows = table.map((row) => row.map((cell) => (cell || "").replaceAll("|", "/").trim()));
  const [header, ...body] = rows;
  const headerRow = `| ${header.join(" | ")} |`;
  const separatorRow = `| ${header.map(() => "---").join(" | ")} |`;
  const bodyRows = body.map((row) => `| ${row.join(" | ")} |`);

  return [headerRow, separatorRow, ...bodyRows].join("\n");
}

function chunkText(text, chunkSize, overlap) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end === text.length) break;
    start = end - overlap;
  }
  return chunks;
}

async function embedText(text) {
  const command = new InvokeModelCommand({
    modelId: config.embeddingModelId,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify({ inputText: text }),
  });
  const response = await bedrockRuntime.send(command);
  const body = JSON.parse(Buffer.from(response.body).toString("utf-8"));
  return body.embedding;
}

router.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "A PDF file is required" });
  }

  try {
    const parser = new PDFParse({ data: req.file.buffer });
    const textResult = await parser.getText();
    const tableResult = await parser.getTable();
    await parser.destroy();

    let combinedText = textResult.text;
    for (const pageTables of tableResult.pages) {
      for (const table of pageTables.tables) {
        const markdown = tableToMarkdown(table);
        if (markdown) {
          combinedText += `\n\n[Table from page ${pageTables.num}]\n${markdown}`;
        }
      }
    }

    const chunks = chunkText(
      combinedText,
      config.chunking.chunkSize,
      config.chunking.chunkOverlap
    );

    if (chunks.length === 0) {
      return res.status(400).json({ error: "No extractable text found in PDF" });
    }

    const vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const embedding = await embedText(chunks[i]);
      vectors.push({
        id: `${crypto.randomUUID()}`,
        values: embedding,
        metadata: {
          text: chunks[i],
          source: req.file.originalname,
          chunkIndex: i,
        },
      });
    }

    const batchSize = 100;
    for (let i = 0; i < vectors.length; i += batchSize) {
      await pineconeIndex.upsert({ records: vectors.slice(i, i + batchSize) });
    }

    res.json({
      message: `Indexed ${chunks.length} chunk(s) from ${req.file.originalname}`,
      chunkCount: chunks.length,
    });
  } catch (err) {
    console.error("PDF upload failed:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

router.post("/query", async (req, res) => {
  const { question } = req.body;

  if (!question?.trim()) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    const queryEmbedding = await embedText(question);

    const results = await pineconeIndex.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    const matches = results.matches || [];
    const context = matches
      .map((match, i) => `[${i + 1}] ${match.metadata?.text || ""}`)
      .join("\n\n");

    const prompt = `You are a helpful assistant answering questions about uploaded PDF documents.

Use the context below to answer the question. Format your response in markdown using:
- A short summary sentence first
- Headings or bold text for key points where helpful
- Bullet points for lists where helpful
- A "Sources" reference is not needed in the text - it is shown separately

If the context doesn't contain the answer, say so plainly instead of guessing.

Context:
${context}

Question: ${question}`;

    const command = new ConverseCommand({
      modelId: config.modelArn,
      messages: [{ role: "user", content: [{ text: prompt }] }],
    });

    const response = await bedrockRuntime.send(command);
    const answer = response.output?.message?.content?.[0]?.text || "";

    const sources = matches.map((match) => ({
      title: match.metadata?.source || "Source",
      text: match.metadata?.text,
      score: match.score,
    }));

    res.json({ answer, sources });
  } catch (err) {
    console.error("PDF query failed:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

// Catches multer errors (e.g. file too large) and returns JSON instead of
// falling through to Express's default HTML error page.
router.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({
        error: `File too large. Maximum size is ${MAX_UPLOAD_BYTES / (1024 * 1024)}MB.`,
      });
    }
    return res.status(400).json({ error: err.message });
  }
  next(err);
});

module.exports = router;
