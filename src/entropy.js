/**
 * entropy.js
 * Computes Shannon entropy for each token position from probsChunksArr.
 *
 * Sits between softMax and tokenProbMaper in the pipeline:
 *
 *   softMax  →  probsChunksArr
 *        ↓
 *   computeEntropy  →  entropiesArr
 *        ↓
 *   tokenProbMaper
 *
 * ─── What entropy tells us ───────────────────────────────────────────────────
 * probsChunksArr[i] is the model's probability distribution over the full
 * vocabulary at position i — its belief about what token comes NEXT.
 *
 * H(i) = -Σ p(x) · log₂(p(x))   for all x in vocabulary
 *
 *   High H(i) → model was UNCERTAIN → next token is informative → KEEP
 *   Low  H(i) → model was CONFIDENT → next token was predictable → candidate for removal
 *
 * Maximum entropy for GPT-2: log₂(50257) ≈ 15.6 bits  (uniform distribution)
 * Typical sentence range:    ~2 – 8 bits
 *
 * ─── Complexity ──────────────────────────────────────────────────────────────
 * Time : O(V·T) — must read every probability value (same cost class as softmax)
 * Space: O(T)   — one float per token position, no chunk data is duplicated
 */

const LN2_INV = 1 / Math.LN2; // convert nats → bits without calling Math.log2

/**
 * Computes Shannon entropy (bits) for a single probability chunk.
 * Zero-probability entries are skipped (p·log(p) → 0 as p → 0, by convention).
 *
 * @param   {Float32Array} probs  One token position's probability view
 * @returns {number}              Entropy in bits
 */
function entropyOfChunk(probs) {
  let h = 0;
  for (let j = 0; j < probs.length; j++) {
    const p = probs[j];
    if (p > 0) h -= p * Math.log(p);
  }
  return h * LN2_INV;
}

/**
 * Computes Shannon entropy for every token position in probsChunksArr.
 *
 * entropiesArr[i] answers: "how uncertain was the model at position i
 * about what token comes next?" — so the token at position i+1 is the
 * one described by entropiesArr[i]. Same causal offset as tokenProbMaper.
 *
 * @param   {Float32Array[]} probsChunksArr  Output of softMax() — T probability views
 * @returns {Float32Array}                   entropiesArr — T entropy scalars in bits
 */
export function computeEntropy(probsChunksArr) {
  const T = probsChunksArr.length;
  const entropiesArr = new Float32Array(T);

  for (let i = 0; i < T; i++) {
    entropiesArr[i] = entropyOfChunk(probsChunksArr[i]);
  }

  return entropiesArr;
}
