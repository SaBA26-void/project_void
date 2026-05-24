import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

// Pure Lambda: Slices flat Float32Array into token chunks
const chunkTensor = (tensorData, vocabSize) =>
  Array.from({ length: tensorData.length / vocabSize }, (_, i) =>
    tensorData.subarray(i * vocabSize, (i + 1) * vocabSize),
  );

// Pure Lambda: Stable Softmax
const computeSoftmax = (logits) => {
  const maxLogit = Math.max(...logits); // Note: For massive vocabularies, a standard loop is faster, but this is highly declarative
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sumExp);
};

/**
 * Compresses a prompt by stripping tokens that cross a predictability threshold.
 */
const compressPrompt = async (promptText, threshold = 0.15) => {
  const modelName = "Xenova/gpt2";

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(modelName);
    const model = await AutoModelForCausalLM.from_pretrained(modelName);

    const encoded = await tokenizer(promptText);
    const inputIds = Array.from(encoded.input_ids.data); // Convert Int32Array to standard JS Array

    const modelGen = await model(encoded);
    const tensorData = modelGen.logits.data;
    const vocabSize = modelGen.logits.dims[2];

    // 1. Chunk the logits per sequence position
    const logitChunks = chunkTensor(tensorData, vocabSize);

    // 2. Map every input token to an object containing its metadata and calculated probability
    const tokenAnalysisPipeline = inputIds.map((id, index) => {
      const tokenText = tokenizer.decode([id]);

      // The first token has no prior context in this loop sequence setup, assign 0 probability baseline
      if (index === 0) {
        return { id, text: tokenText, probability: 0 };
      }

      // Get the distribution from the step BEFORE this token appeared
      const priorDistribution = computeSoftmax(logitChunks[index - 1]);
      const probability = priorDistribution[id];

      return { id, text: tokenText, probability };
    });

    console.log("--- Token Probability Analysis ---");
    tokenAnalysisPipeline.forEach((t) =>
      console.log(
        `Token: "${t.text.trim()}" | Prob: ${(t.probability * 100).toFixed(2)}%`,
      ),
    );

    // 3. Filter out tokens that exceed our redundancy threshold
    const compressedTokens = tokenAnalysisPipeline.filter((token) => {
      const isRedundant = token.probability > threshold;
      return !isRedundant; // Keep only if NOT redundant
    });

    // 4. Reconstruct text from the surviving token IDs
    const compressedIds = compressedTokens.map((t) => t.id);
    const compressedText = tokenizer.decode(compressedIds);

    console.log("\n--- Compression Results ---");
    console.log(`Original:   "${promptText}" (${inputIds.length} tokens)`);
    console.log(
      `Compressed: "${compressedText.trim()}" (${compressedIds.length} tokens)`,
    );
  } catch (error) {
    console.error("Compression Pipeline Error:", error);
  }
};

// Run with a 15% predictability limit
compressPrompt(
  "Tbilisi is Georgia’s capital and the second‑largest city in the region after Baku, the capital of Azerbaijan.",
  0.1,
);
