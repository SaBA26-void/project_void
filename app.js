import { AutoTokenizer, AutoModelForCausalLM } from "@huggingface/transformers";

const promt = "What is the capital city of France?";

async function run() {
    const tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");
    const model = await AutoModelForCausalLM.from_pretrained("Xenova/gpt2");
    // Xenova/Qwen1.5-0.5B-Chat

    const encoded = await tokenizer(promt);
    const keyTensorsArr = await encoded['input_ids']['ort_tensor']['cpuData'];
    const attention = encoded['attention_mask'];

    // console.log("full data: \n", encoded, "\n \n");
    // console.log("keyTensor \n", keyTensorsArr, "\n \n")
    // console.log("attention \n", attention, "\n \n")

    const modelGen = await model(encoded);
    const logitsData = await modelGen['logits']//['ort_tensor']['cpuData'];
    const logitsArr = [...modelGen['logits']['ort_tensor']['cpuData']];
    const dimSize = modelGen['logits']['ort_tensor']['dims'][2];
    // const data1 = logitsData['ort_tensor']
    // const data = data1['cpuData']
    // console.log("logitsArr \n", logitsArr.length, "\n");
    // console.log("data", data)

    const chunks = [];
    function chopper() {
        chunks;
        for (let i = 0; i < keyTensorsArr.length; i++) {
            chunks.push(logitsArr.slice(i * dimSize, ((i * dimSize) + dimSize)));
            console.log("start: ", i * dimSize, " ", "end: ", ((i * dimSize) + dimSize));
        }
        // console.log(chunks.map(c => c.length));
        return chunks;
    };
    chopper();
    // console.log("chunks", chunks);

    function softMax(chunks) {
        let e = Math.E;
        const probChunks = [];
        const maxProbChunks = [];
        const onesChuncks = [];

        for (const arr of chunks) {

            const max = Math.max(...arr);
            const expsArr = arr.map(item => e ** (item - max));
            const sum = expsArr.reduce((acc, curr) => acc + curr, 0);

            if (sum === 0) {
                const n = expsArr.length;
                probChunks.push(Array(n).fill(1 / n));
                continue;
            }

            const probs = expsArr.map(p => p / sum);
            const maxP = Math.max(...probs);
            probChunks.push(probs);
            maxProbChunks.push(maxP);
            const isOne = probs.reduce((acc, curr) => acc + curr, 0);
            onesChuncks.push(isOne);
        }

        // return maxProbChunks;
        return probChunks;
        // return onesChuncks;
    }

    console.log(softMax(chunks));
    const probs = softMax(chunks);
    const probChunks = softMax(chunks);



}

run();
