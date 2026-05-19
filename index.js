import {
  pipeline,
  AutoTokenizer,
  AutoModelForCausalLM,
} from "@huggingface/transformers";

const prompt = "What is the capital city of France?";

async function run() {
  // Load tokenizer + model
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
  const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2");

  // Tokenize properly
  const encoded = await tokenizer(prompt);

  console.log("Full  data:", encoded, "\n \n \n \n");
  console.log("Input IDs:", encoded.input_ids.tolist());
  console.log("Tokens:", encoded.tokens);
  console.log("Attention mask:", encoded.attention_mask.tolist());

  // Run model forward
  const output = await model(encoded);
  const keyTensor = output["present.0.key"];
  console.log("Model output:", output);

  // Pipeline for generation (optional)
  const generator = await pipeline("text-generation", "Xenova/gpt2");
  const result = await generator(prompt, {
    max_new_tokens: 150,
    do_sample: true,
    temperature: 0.7,
    top_k: 50,
    top_p: 0.8,
    repetition_penalty: 1.2,
  });
  console.log("Generated:", result[0].generated_text);
  console.log("სრული ტენზორის ობიექტი: ", output["present.0.key"]);
  console.log("ტენზორის განზომილებები (dims):", keyTensor.ort_tensor.dims);
}

run();
