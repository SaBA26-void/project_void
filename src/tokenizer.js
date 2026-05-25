/**
 * tokenizer.js
 * Loads the HuggingFace tokenizer and encodes a prompt into token IDs.
 *
 * Complexity: O(T) time and space  (T = number of tokens in the prompt)
 */

import { AutoTokenizer } from "@huggingface/transformers";

/**
 * Loads and returns the tokenizer for the given model ID.
 *
 * @param {string} modelId
 * @returns {Promise<object>} tokenizer instance
 */
export async function loadTokenizer(modelId = "Xenova/gpt2") {
  return AutoTokenizer.from_pretrained(modelId);
}

/**
 * Encodes a prompt into:
 *  - encoded  : the tokenizer output object (passed directly to the model)
 *  - tokensArr: flat number[] of token IDs (used for chopper + probability map)
 *
 * Two calls are needed because the HuggingFace API returns different shapes
 * from tokenizer(prompt) vs tokenizer.encode(prompt).
 *
 * @param {object} tokenizer
 * @param {string} prompt
 * @returns {Promise<{ encoded: object, tokensArr: number[] }>}
 */
export async function encodePrompt(tokenizer, prompt) {
  const [encoded, tokensArr] = await Promise.all([
    tokenizer(prompt),
    tokenizer.encode(prompt),
  ]);
  return { encoded, tokensArr };
}
