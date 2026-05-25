/**
 * merger.js
 * Reassembles BPE/WordPiece sub-tokens into whole words and computes
 * a joint probability for each word.
 *
 * Original ran two passes: one accumulation loop then a secondary .map()
 * to finalize each group. Optimised to a single pass — finalizeWord() is
 * called inline as each group closes, so no second array scan is needed.
 *
 * Complexity: O(T) time  |  O(W) space  (W = word count ≤ T)
 */

/** Compiled once at module scope — regex construction is not free */
const PUNCT_RE = /^\p{P}+$/u;

/**
 * Converts an accumulated word group into its final output shape.
 * Computes log-sum joint probability and finds the best representative
 * token ID in a single loop.
 *
 * @param {{ ids: number[], texts: string[], probs: number[] }} group
 * @returns {{ id, ids, text, probability, tokenCount }}
 */
function finalizeWord(group) {
  const { ids, texts, probs } = group;
  const len = probs.length;

  let logSum = 0;
  let repIndex = 0;

  for (let i = 0; i < len; i++) {
    logSum += Math.log(probs[i] > 0 ? probs[i] : 1e-300);
    if (probs[i] > probs[repIndex]) repIndex = i;
  }

  return {
    id: ids[repIndex],
    ids,
    text: texts.join(""),
    probability: Math.exp(logSum),
    tokenCount: ids.length,
  };
}

/**
 * Merges a flat probData array into word-level objects.
 *
 * Rules:
 *  - Empty/whitespace tokens are skipped
 *  - Punctuation tokens are always emitted as standalone words
 *  - A new word starts when the raw token text begins with a space
 *    (GPT-2 encodes word boundaries as a leading space on the first sub-token)
 *
 * @param {{ id: number, text: string, probability: number }[]} probData
 * @returns {{ id, ids, text, probability, tokenCount }[]}  merged word array
 */
export function mergeSubTokensToWords(probData) {
  const words = [];
  let current = null; // accumulator for the word currently being built

  for (let i = 0; i < probData.length; i++) {
    const { id, text: raw, probability } = probData[i];
    const tokenTrim = raw.trim();

    if (tokenTrim === "") continue;

    // --- Punctuation: always a standalone word ---
    if (PUNCT_RE.test(tokenTrim)) {
      if (current) {
        words.push(finalizeWord(current));
        current = null;
      }
      words.push({
        id,
        ids: [id],
        text: tokenTrim,
        probability,
        tokenCount: 1,
      });
      continue;
    }

    // --- New word: leading space OR the very first non-empty token ---
    if (raw.startsWith(" ") || current === null) {
      if (current) words.push(finalizeWord(current));
      current = { ids: [id], texts: [tokenTrim], probs: [probability] };
    } else {
      // Sub-token continuation — append into the current group
      current.ids.push(id);
      current.texts.push(tokenTrim);
      current.probs.push(probability);
    }
  }

  if (current) words.push(finalizeWord(current));

  return words;
}
