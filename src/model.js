/**
 * model.js
 * Loads the causal language model and runs a single forward pass
 * to produce raw logits (modelGen).
 *
 * Complexity: O(V·T) time and space — the full logit tensor must be produced.
 *             V = vocab size (~50,257 for GPT-2), T = token count.
 */

import { AutoModelForCausalLM } from "@huggingface/transformers";

const VALID_DTYPES = new Set(["q4", "q5", "q8", "fp16", "fp32"]);

/**
 * Loads and returns the causal LM for the given model ID.
 *
 * @param {string} modelId
 * @param {{dtype?: "q4"|"q5"|"q8"|"fp16"|"fp32"}} [quantization]
 * @returns {Promise<object>} model instance
 */
export async function loadModel(
  modelId = "Xenova/gpt2",
  { dtype = "fp16" } = {},
) {
  if (!VALID_DTYPES.has(dtype)) {
    throw new Error(
      `Invalid dtype "${dtype}". Must be one of: q4, q5, q8, fp16, fp32`,
    );
  }

  return AutoModelForCausalLM.from_pretrained(modelId, { dtype });
}

/**
 * Runs a forward pass and extracts the three values needed downstream:
 *  - tensorArr : Float32Array — the raw flat logit buffer [T × V]
 *  - vocabSize : number       — vocabulary dimension V (logits.dims[2])
 *
 * `encoded` comes from encodePrompt() in tokenizer.js.
 *
 * @param {object} model
 * @param {object} encoded  tokenizer output object
 * @returns {Promise<{ tensorArr: Float32Array, vocabSize: number }>}
 */
export async function genLogits(model, encoded) {
  const modelGen = await model(encoded);

  // .data is the underlying Float32Array buffer — no copy, just a reference
  const tensorArr = modelGen.logits.data;
  const vocabSize = modelGen.logits.dims[2];

  return { tensorArr, vocabSize };
}
