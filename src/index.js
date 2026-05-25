import { loadTokenizer, encodePrompt } from "./tokenizer.js";
import { loadModel, genLogits } from "./model.js";
import { chopper } from "./chopper.js";
import { softMax } from "./softmax.js";
import { computeEntropy } from "./entropy.js";
import { tokenProbMaper } from "./probability.js";
import { mergeSubTokensToWords } from "./merger.js";
import { compressAndJoin } from "./compressor.js";

/**
 * compressPrompt(modelId, prompt, options)
 *
 * options:
 *  - probability: number (mode A)
 *  - entropy: number (mode B)
 */
export async function compressPrompt(
  modelId,
  quantization,
  prompt,
  options = {},
) {
  const { probability, entropy } = options;

  // enforce mutual exclusivity
  if (probability != null && entropy != null) {
    throw new Error("Use either probability OR entropy mode, not both");
  }

  if (probability == null && entropy == null) {
    throw new Error("You must provide either probability or entropy");
  }

  const mode = probability != null ? "prob" : "entropy";

  // 1. load model + tokenizer
  const [tokenizer, model] = await Promise.all([
    loadTokenizer(modelId),
    loadModel(modelId, quantization),
  ]);

  // 2. encode prompt
  const { encoded, tokensArr } = await encodePrompt(tokenizer, prompt);

  // 3. forward pass
  const { tensorArr, vocabSize } = await genLogits(model, encoded);

  // 4. chunk logits
  const chunks = chopper(tokensArr, tensorArr, vocabSize);

  // 5. softmax
  const probs = await softMax(chunks);

  // 6. entropy
  const entropies = computeEntropy(probs);

  // 7. map token probabilities
  const probData = tokenProbMaper(tokensArr, tokenizer, probs);

  // 8. merge tokens → words
  const merged = mergeSubTokensToWords(probData);

  // 9. attach entropy per word
  let cursor = 0;

  for (const word of merged) {
    let sum = 0;

    for (let k = 0; k < word.tokenCount; k++) {
      sum += cursor > 0 ? (entropies[cursor - 1] ?? 0) : 0;
      cursor++;
    }

    word.entropy = sum / word.tokenCount;
  }

  // 10. compress using selected mode
  if (mode === "prob") {
    return compressAndJoin(merged, probability, -Infinity);
  }

  return compressAndJoin(merged, Infinity, entropy);
}
