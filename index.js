import { pipeline } from "@huggingface/transformers";

const classifier = await pipeline("sentiment-analysis");
const result = await classifier("your text");

console.log(result);