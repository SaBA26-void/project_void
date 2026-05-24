/**
 * probability.js
 * Maps each token ID to the probability the model assigned to it,
 * given all tokens that came before it (causal / autoregressive lookup).
 *
 * The first token always gets probability 0 — there is no prior row to
 * condition on at position 0.
 *
 * Complexity: O(T) time  |  O(T) space
 */

/**
 * Builds a probability-annotated token array.
 *
 * For each token at position i, the probability comes from the softmax row
 * of the PREVIOUS position (i-1) — that's how causal next-token prediction works:
 * row i-1 holds the distribution over what token should come at position i.
 *
 * @param {number[]}       tokensArr      Token IDs from encodePrompt()
 * @param {object}         tokenizer      HuggingFace tokenizer (for decode)
 * @param {Float32Array[]} probsChunksArr Probability views from softMax()
 * @returns {{ id: number, text: string, probability: number }[]}  probData
 */
export function tokenProbMaper(tokensArr, tokenizer, probsChunksArr) {
  const n = tokensArr.length;
  const probData = new Array(n); // pre-allocate — avoids repeated V8 array growth

  for (let index = 0; index < n; index++) {
    const id = tokensArr[index];
    const tokenText = tokenizer.decode([id]);

    if (index === 0) {
      probData[index] = { id, text: tokenText, probability: 0 };
      continue;
    }

    const priorProbs = probsChunksArr[index - 1];
    const probability = priorProbs ? (priorProbs[id] ?? 0) : 0;

    probData[index] = { id, text: tokenText, probability };
  }

  return probData;
}
