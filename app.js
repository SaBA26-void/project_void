import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

const promt = "What is the capital city of France?";

async function run() {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/Qwen1.5-0.5B-Chat");
    const model = await AutoModelForCausalLM.from_pretrained("Xenova/Qwen1.5-0.5B-Chat")

    const encoded = await tokenizer(promt);
    const keyTensor = encoded['input_ids']
    const attention = encoded['attention_mask']

    // console.log("full data: \n", encoded, "\n \n");
    // console.log("keyTensor \n", keyTensor, "\n \n")
    // console.log("attention \n", attention, "\n \n")

    const modelGen = await model(encoded);
    const logitsData = await modelGen['logits']//['ort_tensor']['cpuData'];
    const logitsarr = [...modelGen['logits']['ort_tensor']['cpuData']]
    // const data1 = logitsData['ort_tensor']
    // const data = data1['cpuData']
    console.log("logitsData \n", logitsData, "\n");
    // console.log("data", data)


}

run();
const promtarr = promt.match(/\w+|[^\w\s]/g);

for (let i = 0; i < promtarr.length; i++) {
    console.log(promtarr[i]);
}