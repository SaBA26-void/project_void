# project-void

Entropy-aware prompt compression for LLMs.

## Install

```bash
npm install project-void
```

## Usage

```js
import { compressPrompt } from "project-void";

const result = await compressPrompt(
  "Xenova/gpt2",
  "What is the capital city of France Bakuradze?",
  0.005,
  3,
);

console.log(result.text);
```

## API

### compressPrompt(model, prompt, probability, entropy?)

- `model`: HF model string
- `prompt`: input text
- `probability`: compression threshold
- `entropy`: optional entropy threshold
