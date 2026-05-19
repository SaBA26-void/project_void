import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

const prompt = "What it the captial city of France?";

async function run() {
  // model tokenizer and model pipe
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
  const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2");

  // tokenizer
  const encoded = await tokenizer(prompt);
  const tokesArr = await tokenizer.encode(prompt);
  console.log("encoded", encoded);
  console.log("token Arrs", tokesArr);

  // model gen of logits
  const modelGen = await model(encoded);
  console.log("model Gen", modelGen);

  // getting logits arr
  const tensorArr = modelGen.logits.data;
  console.log("tensorArr", tensorArr);
  // getting  model vocabulary size aka  total token model have
  const vocabSize = modelGen.logits.dims[2];
  console.log("vocabSize", vocabSize);
}

run();
