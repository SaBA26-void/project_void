/**
 * compressor.js
 * Filters words using probability and/or Shannon entropy, then reconstructs
 * the compressed string.
 *
 * Each word object arriving from mergeSubTokensToWords already has:
 *   word.probability  — joint probability of the word's sub-tokens
 *   word.entropy      — avg Shannon entropy across the word's positions (bits)
 *                       (annotated in index.js from entropiesArr)
 *
 * ─── Keep logic ──────────────────────────────────────────────────────────────
 * A word is KEPT if EITHER signal marks it as informative:
 *
 *   probability < probThreshold     → model assigned it low chance (surprising)
 *   entropy     > entropyThreshold  → model was uncertain before it (ambiguous)
 *
 * A word is REMOVED only when BOTH signals agree it was predictable.
 * entropyThreshold defaults to -Infinity so passing no entropy threshold
 * falls back to pure probability filtering — identical to the original behaviour.
 *
 * ─── Complexity ──────────────────────────────────────────────────────────────
 * Time : O(W)  |  Space: O(W)   (W = merged word count)
 */

const PUNCT_RE = /^\p{P}+$/u;

/**
 * Filters words by the probability+entropy criterion and joins survivors
 * into a compressed string.
 *
 * @param {{ text: string, probability: number, entropy: number, tokenCount: number }[]} mergedWords
 * @param {number}  [probThreshold=0.7]             Keep if probability < this
 * @param {number}  [entropyThreshold=-Infinity]     Keep if entropy > this (bits)
 * @param {boolean} [alwaysKeepFirst=true]           Pin the first word regardless
 * @returns {{ text: string, kept: object[], removed: object[] }}
 */
export function compressAndJoin(
  mergedWords,
  probThreshold = 0.7,
  entropyThreshold = -Infinity,
  alwaysKeepFirst = true,
) {
  const kept = [];
  const removed = [];

  for (let i = 0; i < mergedWords.length; i++) {
    const w = mergedWords[i];

    if (i === 0 && alwaysKeepFirst) {
      kept.push(w);
      continue;
    }

    const surprisingByProb = w.probability < probThreshold;
    const uncertainByEntropy = w.entropy > entropyThreshold;

    if (surprisingByProb || uncertainByEntropy) kept.push(w);
    else removed.push(w);
  }

  // String reconstruction: parts[] + single join = O(W)
  const parts = [];
  for (let i = 0; i < kept.length; i++) {
    const { text } = kept[i];
    if (i === 0 || PUNCT_RE.test(text)) parts.push(text);
    else parts.push(" " + text);
  }

  return { text: parts.join(""), kept, removed };
}
