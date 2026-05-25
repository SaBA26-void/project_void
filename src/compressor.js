const PUNCT_RE = /^\p{P}+$/u;

/**
 * Two modes:
 *  - probability mode: use probThreshold
 *  - entropy mode: use entropyThreshold
 */
export function compressAndJoin(
  mergedWords,
  probThreshold = 0.7,
  entropyThreshold = -Infinity,
  alwaysKeepFirst = true,
) {
  const kept = [];
  const removed = [];

  const useProbabilityMode = entropyThreshold === -Infinity;

  for (let i = 0; i < mergedWords.length; i++) {
    const w = mergedWords[i];

    if (i === 0 && alwaysKeepFirst) {
      kept.push(w);
      continue;
    }

    let keep = false;

    if (useProbabilityMode) {
      keep = w.probability < probThreshold;
    } else {
      keep = w.entropy > entropyThreshold;
    }

    if (keep) kept.push(w);
    else removed.push(w);
  }

  const parts = [];

  for (let i = 0; i < kept.length; i++) {
    const { text } = kept[i];

    if (i === 0 || PUNCT_RE.test(text)) {
      parts.push(text);
    } else {
      parts.push(" " + text);
    }
  }

  return {
    text: parts.join(""),
    kept,
    removed,
  };
}
