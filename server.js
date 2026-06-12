const express = require("express");
const path = require("path");
const {
  BedrockAgentRuntimeClient,
  RetrieveAndGenerateCommand,
} = require("@aws-sdk/client-bedrock-agent-runtime");

const config = require("./config");
const pdfRoutes = require("./pdfRoutes");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/api/pdf", pdfRoutes);

const client = new BedrockAgentRuntimeClient({
  region: config.region,
  credentials: config.credentials,
});

// Derives a human-readable title (e.g. an ADR file name) from a citation's
// retrieval location, so the UI can show "ADR-014-event-sourcing.md" instead
// of a raw S3/web URI.
function getSourceTitle(location) {
  if (!location) return "Source";

  const uri =
    location.s3Location?.uri ||
    location.webLocation?.url ||
    location.confluenceLocation?.url ||
    location.salesforceLocation?.url ||
    location.sharePointLocation?.url ||
    location.customDocumentLocation?.id;

  if (!uri) return "Source";

  const fileName = uri.split("/").findLast((part) => part.length > 0);
  try {
    return decodeURIComponent(fileName);
  } catch {
    return fileName;
  }
}

app.post("/api/query", async (req, res) => {
  const { question, sessionId } = req.body;

  if (!question || !question.trim()) {
    return res.status(400).json({ error: "Question is required" });
  }

  try {
    const command = new RetrieveAndGenerateCommand({
      input: { text: question },
      retrieveAndGenerateConfiguration: {
        type: "KNOWLEDGE_BASE",
        knowledgeBaseConfiguration: {
          knowledgeBaseId: config.knowledgeBaseId,
          modelArn: config.modelArn,
          generationConfiguration: {
            promptTemplate: {
              textPromptTemplate: `You are a helpful assistant answering questions about the team's Architecture Decision Records (ADRs) and other documentation.

Use the search results below to answer the question. Format your response in markdown using:
- A short summary sentence first
- Headings or bold text for key points where helpful
- Bullet points for lists of options, decisions, or consequences
- A "Sources" reference is not needed in the text - it is shown separately

If the search results don't contain the answer, say so plainly instead of guessing.

Search results:
$search_results$

Question: $query$`,
            },
          },
        },
      },
      ...(sessionId ? { sessionId } : {}),
    });

    const response = await client.send(command);

    const sources = (response.citations || [])
      .flatMap((citation) => citation.retrievedReferences || [])
      .map((ref) => ({
        text: ref.content?.text,
        title: getSourceTitle(ref.location),
        location: ref.location,
      }));

    res.json({
      answer: response.output?.text || "",
      sessionId: response.sessionId,
      sources,
    });
  } catch (err) {
    console.error("Bedrock query failed:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Bedrock RAG app listening on http://localhost:${PORT}`);
});
