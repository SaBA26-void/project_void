import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

const prompt = "What is the capital city of France?";

async function run() {
  // model tokenizer and model pipe
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
  const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2");

  // tokenizer
  const encoded = await tokenizer(prompt);
  const tokensArr = await tokenizer.encode(prompt);
  console.log("encoded input_ids data: ", encoded.input_ids.data, "\n \n \n");
  // console.log("token Arr: ", tokensArr);

  // model gen of logits
  const modelGen = await model(encoded);
  // console.log("model Gen", modelGen);

  // this part is important for rest
  // getting logits arr
  const tensorArr = modelGen.logits.data;
  // console.log("tensorArr", tensorArr);
  // getting  model vocabulary size aka  total token model have
  const vocabSize = modelGen.logits.dims[2];
  // console.log("vocabSize", vocabSize);

  return await [tokensArr, tensorArr, vocabSize, tokenizer];
}

//
const [tokensArr, tensorArr, vocabSize, tokenizer] = await run();

// chooping raw tensor Arrayes for each token the size of vocabSize
async function chopper(tokensArr, tensorArr, vocabSize) {
  const tensChuncksArr = [];

  tokensArr.map((item, index) => {
    let arr = [];
    arr = tensorArr.slice(index * vocabSize, index * vocabSize + vocabSize);
    tensChuncksArr.push(arr);
  });
  return tensChuncksArr;
}

//
const tensChuncksArr = await chopper(tokensArr, tensorArr, vocabSize);

async function softMax(arr) {
  const e = Math.E;
  const probsChunksArr = [];

  arr.map((subArr) => {
    // get max Num for each sub Arr
    let max = -Infinity;
    for (let i = 0; i < subArr.length; i++) {
      if (subArr[i] > max) max = subArr[i];
    }
    console.log("MAx num", max);

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

    // normilze exps to get probs
    const probs = new Float32Array(subArr.length);
    for (let i = 0; i < exps.length; i++) {
      probs[i] = exps[i] / sum;
    }
    probsChunksArr.push(probs);
  });
  console.log("Probs", probsChunksArr);

  return probsChunksArr;
}

const probsChunksArr = await softMax(tensChuncksArr);

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

const probData = tokenProbMaper(tokensArr, tokenizer, probsChunksArr);

console.log("---------- token Probabilities ------------");
probData.forEach((t) =>
  console.log(
    `Token: "${t.text.trim()}" | Prob: ${(t.probability * 100).toFixed(2)}%`,
  ),
);
