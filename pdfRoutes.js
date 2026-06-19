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
const { keywordSearch: bm25KeywordSearch } = require("./bm25");

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

// Pinecone doesn't support full-text retrieval, so for keyword (BM25) search
// we pull all chunk text back out of Pinecone and index it in memory. The
// result is cached since listing/fetching every record is expensive.
let chunkCachePromise = null;

async function fetchAllChunksFromPinecone() {
  const chunks = [];
  let paginationToken;

  do {
    const listResult = await pineconeIndex.listPaginated({
      limit: 100,
      ...(paginationToken ? { paginationToken } : {}),
    });
    const ids = (listResult.vectors || []).map((vector) => vector.id).filter(Boolean);

    if (ids.length > 0) {
      const fetchResult = await pineconeIndex.fetch({ ids });
      for (const id of ids) {
        const metadata = fetchResult.records[id]?.metadata;
        if (!metadata) continue;
        chunks.push({
          id,
          text: metadata.text,
          source: metadata.source,
          chunkIndex: metadata.chunkIndex,
        });
      }
    }

    paginationToken = listResult.pagination?.next;
  } while (paginationToken);

  return chunks;
}

function getAllChunks() {
  if (!chunkCachePromise) {
    chunkCachePromise = fetchAllChunksFromPinecone().catch((err) => {
      chunkCachePromise = null;
      throw err;
    });
  }
  return chunkCachePromise;
}

function invalidateChunkCache() {
  chunkCachePromise = null;
}

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

// --- Retrieval ---

async function vectorSearch(question, topK) {
  const queryEmbedding = await embedText(question);

  const results = await pineconeIndex.query({
    vector: queryEmbedding,
    topK,
    includeMetadata: true,
  });

  return (results.matches || []).map((match) => ({
    id: match.id,
    text: match.metadata?.text,
    source: match.metadata?.source,
    chunkIndex: match.metadata?.chunkIndex,
    score: match.score,
  }));
}

async function keywordSearch(question, topK) {
  const chunks = await getAllChunks();
  return bm25KeywordSearch(chunks, question, topK, config.retrieval.bm25);
}

async function hybridSearch(question, topK) {
  const fetchK = Math.max(topK * 2, 10);
  const [vectorResults, keywordResults] = await Promise.all([
    vectorSearch(question, fetchK),
    keywordSearch(question, fetchK),
  ]);

  const rrfK = config.retrieval.rrf.k;
  const combined = new Map();

  const addResults = (results) => {
    results.forEach((result, idx) => {
      const rrfScore = 1 / (rrfK + idx + 1);
      const existing = combined.get(result.id);
      if (existing) {
        existing.score += rrfScore;
      } else {
        combined.set(result.id, { ...result, score: rrfScore });
      }
    });
  };

  addResults(vectorResults);
  addResults(keywordResults);

  return Array.from(combined.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

async function search(mode, question, topK) {
  if (mode === "vector") return vectorSearch(question, topK);
  if (mode === "keyword") return keywordSearch(question, topK);
  return hybridSearch(question, topK);
}

async function generateAnswer(question, matches) {
  const context = matches
    .map((match, i) => `[${i + 1}] ${match.text || ""}`)
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
    title: match.source || "Source",
    text: match.text,
    score: match.score,
  }));

  return { answer, sources };
}

// --- Advanced RAG pipeline (query expansion, self-query, filtered search, rerank) ---

function extractJson(text) {
  if (!text) return null;

  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : text;

  try {
    return JSON.parse(candidate.trim());
  } catch {
    return null;
  }
}

async function converseJson(prompt) {
  const command = new ConverseCommand({
    modelId: config.modelArn,
    messages: [{ role: "user", content: [{ text: prompt }] }],
  });
  const response = await bedrockRuntime.send(command);
  const text = response.output?.message?.content?.[0]?.text || "";
  return extractJson(text);
}

// Generates alternative phrasings of the question to widen recall, since a
// single phrasing may miss chunks that use different wording.
async function expandQueries(question) {
  const prompt = `Generate 3 alternative search queries that rephrase or expand the question below to improve retrieval recall (use synonyms, broader or narrower phrasing, different angles). Return ONLY a JSON array of 3 strings, with no other text or markdown.

Question: ${question}`;

  const parsed = await converseJson(prompt);
  const expansions = Array.isArray(parsed)
    ? parsed.filter((q) => typeof q === "string" && q.trim())
    : [];

  return [question, ...expansions];
}

// Detects whether the question targets a specific uploaded document and, if
// so, returns a metadata filter plus a query rewritten to drop the
// document reference (so embedding focuses on the actual information need).
async function selfQuery(question, availableSources) {
  if (availableSources.length === 0) {
    return { source: null, searchQuery: question };
  }

  const prompt = `Available documents: ${availableSources.join(", ")}

Given the user question below, determine whether the user is asking specifically about one of the documents listed above (by name). If so, respond with JSON {"source": "<exact filename from the list>", "searchQuery": "<the question, rewritten to focus on the information need without the document reference>"}. If the question isn't about a specific document, respond with JSON {"source": null, "searchQuery": "<the original question>"}. Return ONLY JSON, with no other text or markdown.

Question: ${question}`;

  const parsed = await converseJson(prompt);
  if (!parsed || typeof parsed !== "object") {
    return { source: null, searchQuery: question };
  }

  const source = availableSources.includes(parsed.source) ? parsed.source : null;
  const searchQuery =
    typeof parsed.searchQuery === "string" && parsed.searchQuery.trim()
      ? parsed.searchQuery
      : question;

  return { source, searchQuery };
}

// Runs vector search for each query variant, optionally filtered to a single
// source document, and merges the results by chunk id (keeping the best
// score seen for each chunk).
async function filteredVectorSearch(queries, filterSource, topK) {
  const filterArg = filterSource ? { source: { $eq: filterSource } } : undefined;

  const resultSets = await Promise.all(
    queries.map(async (q) => {
      const queryEmbedding = await embedText(q);
      const results = await pineconeIndex.query({
        vector: queryEmbedding,
        topK,
        includeMetadata: true,
        ...(filterArg ? { filter: filterArg } : {}),
      });
      return (results.matches || []).map((match) => ({
        id: match.id,
        text: match.metadata?.text,
        source: match.metadata?.source,
        chunkIndex: match.metadata?.chunkIndex,
        score: match.score,
      }));
    })
  );

  const merged = new Map();
  for (const results of resultSets) {
    for (const result of results) {
      const existing = merged.get(result.id);
      if (!existing || result.score > existing.score) {
        merged.set(result.id, result);
      }
    }
  }

  return Array.from(merged.values());
}

// Asks the LLM to score each candidate chunk's relevance to the question and
// keeps only the top-scoring ones, trimming noise the vector search let through.
async function rerank(question, candidates, topK) {
  if (candidates.length === 0) return [];

  const passages = candidates
    .map((c, i) => `[${i + 1}] ${(c.text || "").slice(0, 1000)}`)
    .join("\n\n");

  const prompt = `Question: ${question}

Rate how relevant each passage below is to answering the question, on a scale from 0 (irrelevant) to 10 (directly answers it). Return ONLY a JSON array of numbers, one per passage, in the same order, with no other text or markdown.

${passages}`;

  const parsed = await converseJson(prompt);
  const scores = Array.isArray(parsed) ? parsed : [];

  return candidates
    .map((c, i) => ({ ...c, rerankScore: typeof scores[i] === "number" ? scores[i] : 0 }))
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .slice(0, topK);
}

async function getAvailableSources() {
  const chunks = await getAllChunks();
  return Array.from(new Set(chunks.map((c) => c.source).filter(Boolean)));
}

// --- Routes ---

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
        id: crypto.randomUUID(),
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

    invalidateChunkCache();

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
  const { question, mode = "hybrid" } = req.body;

  if (!question?.trim()) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    const matches = await search(mode, question, 5);
    const { answer, sources } = await generateAnswer(question, matches);
    res.json({ answer, sources, mode });
  } catch (err) {
    console.error("PDF query failed:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

router.post("/compare", async (req, res) => {
  const { question } = req.body;

  if (!question?.trim()) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    const modes = ["vector", "keyword", "hybrid"];
    const results = await Promise.all(
      modes.map(async (mode) => {
        const matches = await search(mode, question, 5);
        const { answer, sources } = await generateAnswer(question, matches);
        return { mode, answer, sources };
      })
    );
    res.json({ results });
  } catch (err) {
    console.error("PDF compare failed:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

router.post("/advanced-query", async (req, res) => {
  const { question } = req.body;

  if (!question?.trim()) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    const availableSources = await getAvailableSources();

    const [expandedQueries, selfQueryResult] = await Promise.all([
      expandQueries(question),
      selfQuery(question, availableSources),
    ]);

    const searchQueries = Array.from(new Set([...expandedQueries, selfQueryResult.searchQuery]));

    let candidates = await filteredVectorSearch(searchQueries, selfQueryResult.source, 8);

    // Fall back to an unfiltered search if a detected filter returns nothing.
    if (candidates.length === 0 && selfQueryResult.source) {
      candidates = await filteredVectorSearch(searchQueries, null, 8);
    }

    const reranked = await rerank(question, candidates, 5);
    const { answer, sources } = await generateAnswer(question, reranked);

    res.json({
      answer,
      sources,
      pipeline: {
        expandedQueries,
        selfQuery: selfQueryResult,
        candidateCount: candidates.length,
        reranked: reranked.map((c) => ({
          source: c.source,
          chunkIndex: c.chunkIndex,
          vectorScore: c.score,
          rerankScore: c.rerankScore,
          text: c.text,
        })),
      },
    });
  } catch (err) {
    console.error("Advanced RAG query failed:", err);
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
