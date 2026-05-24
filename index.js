/**
 * index.js
 * Orchestrates the full pipeline in the same order as the original script:
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
 *   tokenProbMaper →  probData
 *        ↓
 *   mergeSubTokensToWords  →  merged
 *        ↓
 *   compressAndJoin  →  result
 */

import { loadTokenizer, encodePrompt } from "./tokenizer.js";
import { loadModel, genLogits } from "./model.js";
import { chopper } from "./chopper.js";
import { softMax } from "./softmax.js";
import { tokenProbMaper } from "./probability.js";
import { mergeSubTokensToWords } from "./merger.js";
import { compressAndJoin } from "./compressor.js";

const MODEL_ID = "Xenova/gpt2";
const prompt = "What is the capital city of France Bakuradze?";
const THRESHOLD = 0.005;

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

// ─── 6. Map each token to its predicted probability ───────────────────────────
const probData = tokenProbMaper(tokensArr, tokenizer, probsChunksArr);

// ─── 7. Merge BPE sub-tokens into whole words ────────────────────────────────
const merged = mergeSubTokensToWords(probData);

// ─── 8. Filter and reconstruct compressed string ─────────────────────────────
const result = compressAndJoin(merged, THRESHOLD);

// ─── Output ───────────────────────────────────────────────────────────────────
const originalTokenCount = tokensArr.length;
const compressedTokenCount = result.kept.reduce(
  (sum, word) => sum + word.tokenCount,
  0,
);

console.log(`Original:   "${prompt}" (${originalTokenCount} tokens)`);
console.log(
  `Compressed: "${result.text.trim()}" (${compressedTokenCount} tokens)`,
);
