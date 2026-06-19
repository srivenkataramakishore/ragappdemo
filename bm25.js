// Simple BM25 keyword search over an in-memory list of chunks.
// Each chunk is expected to have at least a `text` field.

function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

// Precomputes per-document term frequencies, lengths and document
// frequencies so scoring each query only needs to look up these stats.
function buildIndex(chunks) {
  const docTokens = chunks.map((chunk) => tokenize(chunk.text));
  const docLengths = docTokens.map((tokens) => tokens.length);
  const avgDocLength =
    docLengths.length > 0
      ? docLengths.reduce((sum, len) => sum + len, 0) / docLengths.length
      : 0;

  const docFreq = new Map();
  const docTermFreqs = docTokens.map((tokens) => {
    const termFreq = new Map();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
    }
    for (const term of termFreq.keys()) {
      docFreq.set(term, (docFreq.get(term) || 0) + 1);
    }
    return termFreq;
  });

  return {
    docTermFreqs,
    docLengths,
    avgDocLength,
    docFreq,
    docCount: chunks.length,
  };
}

function keywordSearch(chunks, question, topK, options = {}) {
  const { k1 = 1.5, b = 0.75 } = options;
  if (chunks.length === 0) return [];

  const index = buildIndex(chunks);
  const queryTokens = tokenize(question);
  if (queryTokens.length === 0) return [];

  const scores = chunks.map((chunk, i) => {
    const termFreq = index.docTermFreqs[i];
    const docLength = index.docLengths[i];
    let score = 0;

    for (const term of queryTokens) {
      const freq = termFreq.get(term) || 0;
      if (freq === 0) continue;

      const df = index.docFreq.get(term) || 0;
      const idf = Math.log((index.docCount - df + 0.5) / (df + 0.5) + 1);
      const denom =
        freq + k1 * (1 - b + (b * docLength) / (index.avgDocLength || 1));
      score += idf * ((freq * (k1 + 1)) / denom);
    }

    return { chunk, score };
  });

  return scores
    .filter((entry) => entry.score > 0)
    .sort((x, y) => y.score - x.score)
    .slice(0, topK)
    .map(({ chunk, score }) => ({ ...chunk, score }));
}

module.exports = { tokenize, keywordSearch };
