# Project Void

**Repository:** [github.com/SaBA26-void/project_void](https://github.com/SaBA26-void/project_void) · **Author:** [@SaBA26-void](https://github.com/SaBA26-void)

Entropy-aware prompt compression for large language models.

Project Void analyzes token probability and entropy to remove low-information tokens while preserving semantic structure.

Designed for:

- Prompt optimization
- Token-cost reduction
- LLM preprocessing pipelines
- Entropy-based NLP experiments

---

## Installation

Requires **Node.js 18+**.

```bash
npm install project_void
```

On first use, the Hugging Face model (e.g. `Xenova/gpt2`) is downloaded from the network and cached locally.

### Install size

The published `project_void` package is small (source only). `npm install` also pulls in `@huggingface/transformers` (ONNX Runtime, native addons), which is typically **hundreds of MB** on disk. That footprint comes from running a local causal LM, not from this library’s compression code.

---

## Usage

### Probability mode

```javascript
import { compressPrompt } from "project_void";

const result = await compressPrompt(
  "Xenova/gpt2",
  "What is the capital city of France?",
  {
    probability: 0.1,
  },
);

console.log(result.text);
```

---

### Entropy mode

```javascript
import { compressPrompt } from "project_void";

const result = await compressPrompt(
  "Xenova/gpt2",
  "What is the capital city of France?",
  {
    entropy: 7,
  },
);

console.log(result.text);
```

---

## Full Example

```javascript
import { compressPrompt } from "project_void";

const prompt = "What is the capital city of France?";

const result = await compressPrompt("Xenova/gpt2", prompt, {
  probability: 0.1,
});

const originalTokenCount =
  result.kept.reduce((s, w) => s + w.tokenCount, 0) +
  result.removed.reduce((s, w) => s + w.tokenCount, 0);

const compressedTokenCount = result.kept.reduce((s, w) => s + w.tokenCount, 0);

console.log(`Original: "${prompt}" (${originalTokenCount} tokens)`);

console.log(
  `Compressed: "${result.text.trim()}" (${compressedTokenCount} tokens)`,
);

console.log("\nKept:");

for (const w of result.kept) {
  console.log(
    `"${w.text}" prob=${w.probability.toFixed(4)} entropy=${w.entropy.toFixed(2)} bits`,
  );
}

console.log("\nRemoved:");

for (const w of result.removed) {
  console.log(
    `"${w.text}" prob=${w.probability.toFixed(4)} entropy=${w.entropy.toFixed(2)} bits`,
  );
}
```

---

## API

## `compressPrompt(modelId, prompt, options)`

Compresses text using token-level probability or entropy filtering.

### Parameters

| Name                  | Type     | Description                                        |
| --------------------- | -------- | -------------------------------------------------- |
| `modelId`             | `string` | Hugging Face model identifier (e.g. `Xenova/gpt2`) |
| `prompt`              | `string` | Input text                                         |
| `options.probability` | `number` | Keep tokens below probability threshold            |
| `options.entropy`     | `number` | Keep tokens above entropy threshold                |

Use **either**:

- `probability`

or

- `entropy`

---

## Returns

```javascript
{
  text: string,
  kept: Word[],
  removed: Word[]
}
```

Each word contains:

```javascript
{
  text: string,
  probability: number,
  entropy: number,
  tokenCount: number
}
```

---

## Compression Strategy

### Probability mode

Keeps low-probability (informative) tokens.

---

### Entropy mode

Keeps high-entropy (uncertain / information-rich) tokens.

---

## Example Output

```javascript
{
  text: "capital France",
  kept: [...],
  removed: [...]
}
```

---

## Features

- Token probability pruning
- Entropy-based filtering
- Claude Shannon information theory
- Hugging Face transformer support (`@huggingface/transformers`)
- ESM package with a single public export: `compressPrompt`
- Token-level analysis output
- Research-friendly compression pipeline

---

## Use Cases

- Reduce LLM prompt cost
- Compress retrieval context
- Token saliency analysis
- Prompt engineering experiments
- Information-density filtering

---

## License

MIT

---

## Author

**Saba Bakuradze** — [@SaBA26-void](https://github.com/SaBA26-void)

- Project repository: [SaBA26-void/project_void](https://github.com/SaBA26-void/project_void)
- GitHub profile: [github.com/SaBA26-void](https://github.com/SaBA26-void)
