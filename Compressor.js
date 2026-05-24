/**
 * compressor.js
 * Filters out high-probability (predictable) words and reconstructs
 * the compressed string.
 *
 * Original built the output string with += inside a loop — O(W²) in the
 * worst case because each concatenation may copy the growing string.
 * Optimised: collect parts into an array, join once at the end — O(W).
 *
 * Complexity: O(W) time  |  O(W) space  (W = word count)
 */

const PUNCT_RE = /^\p{P}+$/u;

/**
 * Filters words by probability threshold and joins them into a string.
 *
 * A word is KEPT when:
 *   - it is the first word and alwaysKeepFirst is true, OR
 *   - its joint probability is BELOW threshold (surprising → informative)
 *
 * A word is REMOVED when its probability is >= threshold (predictable → redundant).
 *
 * @param {{ text: string, probability: number, tokenCount: number }[]} mergedWords
 * @param {number}  [threshold=0.7]        Drop words at or above this probability
 * @param {boolean} [alwaysKeepFirst=true] Pin the first word regardless
 * @returns {{ text: string, kept: object[], removed: object[] }}
 */
export function compressAndJoin(
  mergedWords,
  threshold = 0.7,
  alwaysKeepFirst = true,
) {
  const keep = (prob) => prob < threshold;

  const kept = [];
  const removed = [];

  // --- Filter pass ---
  for (let i = 0; i < mergedWords.length; i++) {
    const w = mergedWords[i];

    if (i === 0 && alwaysKeepFirst) {
      kept.push(w);
      continue;
    }
    if (keep(w.probability)) kept.push(w);
    else removed.push(w);
  }

  // --- String reconstruction: parts[] + single join = O(W) ---
  const parts = [];
  for (let i = 0; i < kept.length; i++) {
    const { text } = kept[i];
    // No space before the first word or before punctuation
    if (i === 0 || PUNCT_RE.test(text)) parts.push(text);
    else parts.push(" " + text);
  }

  return { text: parts.join(""), kept, removed };
}
