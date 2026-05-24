/**
 * index.js
 * Orchestrates the full pipeline:
 *
 *   loadTokenizer + loadModel  (parallel)
 *        ↓
 *   encodePrompt  →  tokensArr + encoded
 *        ↓
 *   genLogits     →  tensorArr + vocabSize
 *        ↓
 *   chopper       →  tensChuncksArr
 *        ↓
 *   softMax       →  probsChunksArr
 *        ↓
 *   computeEntropy →  entropiesArr          ← NEW
 *        ↓
 *   tokenProbMaper →  probData
 *        ↓
 *   mergeSubTokensToWords  →  merged
 *        ↓
 *   (annotate merged words with entropy)    ← NEW
 *        ↓
 *   compressAndJoin  →  result
 */

import { loadTokenizer, encodePrompt } from "./tokenizer.js";
import { loadModel, genLogits } from "./model.js";
import { chopper } from "./chopper.js";
import { softMax } from "./softmax.js";
import { computeEntropy } from "./entropy.js";
import { tokenProbMaper } from "./probability.js";
import { mergeSubTokensToWords } from "./merger.js";
import { compressAndJoin } from "./compressor.js";

const MODEL_ID = "Xenova/gpt2";
const prompt = "What is the capital city of France Bakuradze?";

// Probability: keep word if its probability is below this
const PROB_THRESHOLD = 0.005;

// Entropy: keep word if the model entropy before it exceeded this (bits).
// Set to -Infinity to disable entropy filtering (pure probability mode).
// GPT-2 typical range: ~2–8 bits. Start around 3.0 bits.
const ENTROPY_THRESHOLD = 3.0;

// ─── 1. Load tokenizer and model in parallel ──────────────────────────────────
const [tokenizer, model] = await Promise.all([
  loadTokenizer(MODEL_ID),
  loadModel(MODEL_ID),
]);

// ─── 2. Encode prompt ─────────────────────────────────────────────────────────
const { encoded, tokensArr } = await encodePrompt(tokenizer, prompt);

// ─── 3. Forward pass → raw logit buffer ──────────────────────────────────────
const { tensorArr, vocabSize } = await genLogits(model, encoded);

// ─── 4. Chop flat tensor into per-token chunks ────────────────────────────────
const tensChuncksArr = chopper(tokensArr, tensorArr, vocabSize);

// ─── 5. Softmax → probability views ──────────────────────────────────────────
const probsChunksArr = await softMax(tensChuncksArr);

// ─── 6. Shannon entropy per token position ────────────────────────────────────
const entropiesArr = computeEntropy(probsChunksArr);

// ─── 7. Map each token to its predicted probability ───────────────────────────
const probData = tokenProbMaper(tokensArr, tokenizer, probsChunksArr);

// ─── 8. Merge BPE sub-tokens into whole words ────────────────────────────────
const merged = mergeSubTokensToWords(probData);

// ─── 9. Annotate each merged word with its average position entropy ───────────
// Walk the merged words in order; each word's tokenCount tells us how many
// positions it consumed, so we can advance a cursor through entropiesArr.
// entropiesArr[i] = entropy before position i+1, so token at position i+1
// uses entropiesArr[i] — same causal offset as probability.
let cursor = 0;
for (const word of merged) {
  let sum = 0;
  for (let k = 0; k < word.tokenCount; k++) {
    // entropiesArr[cursor - 1] is the prior-position entropy for this token.
    // Position 0 has no prior, so clamp to 0.
    sum += cursor > 0 ? (entropiesArr[cursor - 1] ?? 0) : 0;
    cursor++;
  }
  word.entropy = sum / word.tokenCount;
}

// ─── 10. Filter by probability OR entropy, reconstruct string ─────────────────
const result = compressAndJoin(merged, PROB_THRESHOLD, ENTROPY_THRESHOLD);

// ─── Output ───────────────────────────────────────────────────────────────────
const originalTokenCount = tokensArr.length;
const compressedTokenCount = result.kept.reduce(
  (sum, w) => sum + w.tokenCount,
  0,
);

console.log(`Original:   "${prompt}" (${originalTokenCount} tokens)`);
console.log(
  `Compressed: "${result.text.trim()}" (${compressedTokenCount} tokens)`,
);
console.log();
console.log("kept:");
for (const w of result.kept)
  console.log(
    `  "${w.text}"  prob=${w.probability.toFixed(4)}  entropy=${w.entropy.toFixed(2)} bits`,
  );
console.log("removed:");
for (const w of result.removed)
  console.log(
    `  "${w.text}"  prob=${w.probability.toFixed(4)}  entropy=${w.entropy.toFixed(2)} bits`,
  );
