import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

// 1. Pure Function: Slice 1D flat tensor array into a 2D matrix (Tokens x Vocabulary)
const chunkTensor = (tensorData, vocabSize) => {
    const chunks = [];
    const numTokens = tensorData.length / vocabSize;

    for (let i = 0; i < numTokens; i++) {
        const start = i * vocabSize;
        const end = start + vocabSize;
        chunks.push(tensorData.subarray(start, end));
    }
    return chunks;
};

// 2. Pure Function: Numerically stable Softmax for a single token slice
const computeSoftmax = (logits) => {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    const probabilities = new Float32Array(logits.length);
    let sumExp = 0;

    for (let i = 0; i < logits.length; i++) {
        const expVal = Math.exp(logits[i] - maxLogit);
        probabilities[i] = expVal;
        sumExp += expVal;
    }

    for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sumExp;
    }

    return probabilities;
};

// 3. Main Execution Function
const analyzePromptProbabilities = async (promptText) => {
    const modelName = "Xenova/gpt2";

    try {
        const tokenizer = await AutoTokenizer.from_pretrained(modelName);
        const model = await AutoModelForCausalLM.from_pretrained(modelName);

        // Tokenize prompt to get sequence input IDs
        const encoded = await tokenizer(promptText);
        const inputIds = encoded.input_ids.data; // Int32Array of token IDs

        // Run the model forward pass
        const modelGen = await model(encoded);
        const tensorData = modelGen.logits.data;
        const vocabSize = modelGen.logits.dims[2];

        // Break the raw logits down by sequence positions
        const logitChunks = chunkTensor(tensorData, vocabSize);

        console.log(`\nAnalyzing sequence probabilities for: "${promptText}"\n`);
        console.log(`------------------------------------------------------------`);

        // We loop up to logitChunks.length - 1 because the very last chunk 
        // predicts a future token outside our current prompt.
        for (let i = 0; i < logitChunks.length - 1; i++) {
            const currentTokenId = inputIds[i];
            const nextTokenId = inputIds[i + 1];

            const currentTokenText = tokenizer.decode([currentTokenId]);
            const nextTokenText = tokenizer.decode([nextTokenId]);

            // Transform raw logits at step i to a valid probability distribution
            const probabilities = computeSoftmax(logitChunks[i]);

            // Look up the probability assigned to the token that actually followed
            const actualTokenProbability = probabilities[nextTokenId];
            const percentage = (actualTokenProbability * 100).toFixed(4);

            console.log(`Given: [${currentTokenText.trim()}] -> Probability of next token [${nextTokenText.trim()}]: ${percentage}%`);
        }
        console.log(`------------------------------------------------------------`);

    } catch (error) {
        console.error("Error analyzing sequence:", error);
    }
};

analyzePromptProbabilities("What is the capital city of France?");