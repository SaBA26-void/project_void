import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

const prompt = "What is the capital city of France Bakuradze?";

async function run() {
  // model tokenizer and model pipe
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
  const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2");

  // tokenizer
  const encoded = await tokenizer(prompt);
  const tokensArr = await tokenizer.encode(prompt);

  // model gen of logits
  const modelGen = await model(encoded);

  // getting logits arr
  const tensorArr = modelGen.logits.data;

  // getting  model vocabulary size aka  total token model have
  const vocabSize = modelGen.logits.dims[2];

  return [tokensArr, tensorArr, vocabSize, tokenizer];
}

const [tokensArr, tensorArr, vocabSize, tokenizer] = await run();

// chopping raw tensor Arrayes for each token the size of vocabSize
async function chopper(tokensArr, tensorArr, vocabSize) {
  const tensChuncksArr = [];

  tokensArr.map((item, index) => {
    let arr = [];
    arr = tensorArr.slice(index * vocabSize, index * vocabSize + vocabSize);
    tensChuncksArr.push(arr);
  });
  return tensChuncksArr;
}

// execute tensor of chuncks of arr
const tensChuncksArr = await chopper(tokensArr, tensorArr, vocabSize);

// softmax for probabilities
async function softMax(arr) {
  const e = Math.E;
  const probsChunksArr = [];

  arr.map((subArr) => {
    // get max Num for each sub Arr
    let max = -Infinity;
    for (let i = 0; i < subArr.length; i++) {
      if (subArr[i] > max) max = subArr[i];
    }

    // get exps arr from raw data
    const exps = new Float32Array(subArr.length);
    for (let i = 0; i < subArr.length; i++) {
      exps[i] = e ** (subArr[i] - max);
    }

    // sum of exps arr
    let sum = 0;
    for (let i = 0; i < exps.length; i++) {
      sum += exps[i];
    }

    // normalize exps to get probs
    const probs = new Float32Array(subArr.length);
    for (let i = 0; i < exps.length; i++) {
      probs[i] = exps[i] / sum;
    }
    probsChunksArr.push(probs);
  });

  return probsChunksArr;
}

// execute probabilities
const probsChunksArr = await softMax(tensChuncksArr);

// probability map for each tokens
function tokenProbMaper(tokensArr, tokenizer, probsChunksArr) {
  return tokensArr.map((id, index) => {
    const tokenText = tokenizer.decode([id]);

    if (index === 0) {
      return { id, text: tokenText, probability: 0 };
    }

    const priorProbs = probsChunksArr[index - 1];
    const probability = priorProbs ? (priorProbs[id] ?? 0) : 0;

    return { id, text: tokenText, probability };
  });
}

// execute probability map
const probData = tokenProbMaper(tokensArr, tokenizer, probsChunksArr);

//Merge subtokens into words
function mergeSubTokensToWords(probData) {
  const words = [];
  let current = null;

  for (const t of probData) {
    const raw = t.text;
    const tokenTrim = raw.trim();
    if (tokenTrim === "") continue;

    const isPunctuation = /^\p{P}+$/u.test(tokenTrim);

    if (isPunctuation) {
      if (current) {
        words.push(current);
        current = null;
      }
      words.push({
        ids: [t.id],
        text: tokenTrim,
        probs: [t.probability],
      });
      continue;
    }

    const startsNew = raw.startsWith(" ") || current === null;

    if (startsNew) {
      if (current) words.push(current);
      current = {
        ids: [t.id],
        texts: [tokenTrim],
        probs: [t.probability],
      };
    } else {
      current.ids.push(t.id);
      current.texts.push(tokenTrim);
      current.probs.push(t.probability);
    }
  }

  if (current) words.push(current);

  return words.map((w) => {
    const logSum = w.probs.reduce(
      (s, p) => s + Math.log(Math.max(p, 1e-300)),
      0,
    );
    const jointProb = Math.exp(logSum);

    const text = w.texts ? w.texts.join("") : w.text || "";

    let repIndex = 0;
    for (let i = 1; i < w.probs.length; i++) {
      if (w.probs[i] > w.probs[repIndex]) repIndex = i;
    }
    const repId = w.ids[repIndex];

    return {
      id: repId,
      ids: w.ids,
      text,
      probability: jointProb,
      tokenCount: w.ids.length, // Track how many tokens make up this word
    };
  });
}

// Compression / final string builder
function compressAndJoin(mergedWords, threshold = 0.7, alwaysKeepFirst = true) {
  const keep = (prob) => prob < threshold;

  const kept = [];
  const removed = [];

  for (let i = 0; i < mergedWords.length; i++) {
    const w = mergedWords[i];

    if (i === 0 && alwaysKeepFirst) {
      kept.push(w);
      continue;
    }

    if (keep(w.probability)) kept.push(w);
    else removed.push(w);
  }

  let out = "";
  for (const item of kept) {
    const p = item.text;
    if (out === "") {
      out = p;
      continue;
    }
    if (/^\p{P}+$/u.test(p)) out += p;
    else out += " " + p;
  }

  return { text: out, kept, removed };
}

// -----------------
// Execution block
// -----------------

const merged = mergeSubTokensToWords(probData);

const result = compressAndJoin(merged, 0.005);

// Calculate token counts dynamically
const originalTokenCount = tokensArr.length;
const compressedTokenCount = result.kept.reduce(
  (sum, word) => sum + word.tokenCount,
  0,
);

// Final output
console.log(`Original:   "${prompt}" (${originalTokenCount} tokens)`);
console.log(
  `Compressed: "${result.text.trim()}" (${compressedTokenCount} tokens)`,
);
