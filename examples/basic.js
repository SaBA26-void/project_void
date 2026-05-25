import { compressPrompt } from "project_void";

const prompt = "What is the capital city of France?";

// --------------------
// PROBABILITY MODE
// --------------------
const result = await compressPrompt("Xenova/gpt2", { dtype: "q4" }, prompt, {
  probability: 0.1,
});
// OR entropy mode:
// const result = await compressPrompt(
//   "Xenova/gpt2",
//   prompt,
//   { entropy: 7 }
// );

// --------------------
// OUTPUT
// --------------------
const originalTokenCount =
  result.kept.reduce((s, w) => s + w.tokenCount, 0) +
  result.removed.reduce((s, w) => s + w.tokenCount, 0);

const compressedTokenCount = result.kept.reduce((s, w) => s + w.tokenCount, 0);

console.log(`Original:   "${prompt}" (${originalTokenCount} tokens)`);

console.log(
  `Compressed: "${result.text.trim()}" (${compressedTokenCount} tokens)`,
);

console.log("\nkept:");

for (const w of result.kept) {
  console.log(
    `  "${w.text}"  prob=${w.probability.toFixed(
      4,
    )}  entropy=${w.entropy.toFixed(2)} bits`,
  );
}

console.log("\nremoved:");

for (const w of result.removed) {
  console.log(
    `  "${w.text}"  prob=${w.probability.toFixed(
      4,
    )}  entropy=${w.entropy.toFixed(2)} bits`,
  );
}
