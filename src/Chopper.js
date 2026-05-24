/**
 * chopper.js
 * Slices the flat [T × V] logit tensor into T per-token chunks.
 *
 * Original used tensorArr.slice() which COPIES V floats per token
 * → O(V·T) extra memory allocation on top of the tensor itself.
 *
 * Optimised: subarray() returns a VIEW into the same buffer — O(1) per slice,
 * O(T) total extra space (just T pointer references, no data duplicated).
 *
 * Complexity: O(T) time  |  O(T) extra space
 */

/**
 * Chops the flat logit tensor into an array of per-token typed-array views.
 * Each view covers exactly vocabSize values for that token position.
 *
 * IMPORTANT: the returned views share memory with tensorArr.
 * Mutating a view mutates the original buffer — softMax.js relies on this.
 *
 * @param {number[]}     tokensArr  Token ID array — its length drives the loop
 * @param {Float32Array} tensorArr  Raw flat logit buffer from modelGen
 * @param {number}       vocabSize  V — vocabulary size from modelGen.logits.dims[2]
 * @returns {Float32Array[]}        tensChuncksArr — T zero-copy views, one per token
 */
export function chopper(tokensArr, tensorArr, vocabSize) {
  const tensChuncksArr = new Array(tokensArr.length); // pre-allocate, no resizing

  for (let index = 0; index < tokensArr.length; index++) {
    // subarray() is O(1): moves a pointer, copies zero bytes
    tensChuncksArr[index] = tensorArr.subarray(
      index * vocabSize,
      index * vocabSize + vocabSize,
    );
  }

  return tensChuncksArr;
}
