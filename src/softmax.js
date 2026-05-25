/**
 * softmax.js
 * Applies numerically-stable softmax to every chunk produced by chopper().
 *
 * Original allocated TWO extra Float32Arrays per token (exps + probs) and
 * used Math.E ** x. Optimised version:
 *   • Mutates each chunk VIEW in place  → 0 extra Float32Array allocations
 *   • Math.exp(x) instead of e**x       → JIT engine intrinsic, faster
 *   • invSum multiply instead of divide → one division, V multiplies vs V divides
 *
 * Complexity: O(V·T) time  |  O(1) extra space per chunk (in-place on views)
 */

/**
 * Applies softmax IN PLACE on a single Float32Array chunk.
 * Three passes: find max → exp+sum → normalise.
 *
 * @param {Float32Array} subArr  One token's logit view (from chopper)
 */
function softmaxChunk(subArr) {
  const len = subArr.length;

  // Pass 1 — max for numerical stability (prevents exp() overflow)
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (subArr[i] > max) max = subArr[i];
  }

  // Pass 2 — exp(x - max) written back into the same view + accumulate sum
  let sum = 0;
  for (let i = 0; i < len; i++) {
    subArr[i] = Math.exp(subArr[i] - max);
    sum += subArr[i];
  }

  // Pass 3 — normalise: one division, V multiplications
  const invSum = 1 / sum;
  for (let i = 0; i < len; i++) {
    subArr[i] *= invSum;
  }
}

/**
 * Applies softmax to every chunk in tensChuncksArr (output of chopper).
 * Mutates each view in place — no new arrays allocated.
 *
 * @param {Float32Array[]} arr  tensChuncksArr from chopper()
 * @returns {Float32Array[]}    probsChunksArr — same array, now holding probabilities
 */
export async function softMax(arr) {
  const probsChunksArr = arr; // same reference — views are mutated in place

  for (let i = 0; i < arr.length; i++) {
    softmaxChunk(arr[i]);
  }

  return probsChunksArr;
}
