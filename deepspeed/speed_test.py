import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

dataset = load_dataset("ChaiML/user_model_inputs")
model_id = "hakurei/litv2-6B-rev2"
model_id = "gpt2"

NUM_SAMPLES = 1

GENERATION_KWARGS = {
    "max_new_tokens": 32,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}
model = AutoModelForCausalLM.from_pretrained(model_id).half().eval().to(0)

INPUT_EXAMPLES = dataset["train"]["text"][:NUM_SAMPLES]

INPUT_EXAMPLES = [example[-500:] for example in INPUT_EXAMPLES]

# load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)


def call_model(model, input_text, batch_size, desc="", verbose=False):
    if verbose:
        print(desc)
        print(f"Batch size: {batch_size}")
    inputs = tokenizer([input_text] * batch_size, return_tensors='pt').to(0)
    results = model.generate(**inputs, **GENERATION_KWARGS)
    for i, result in enumerate(results):
        text_output = tokenizer.decode(result)
        result = text_output[len(input_text):]
        if verbose:
            print(f"#{i}: {result}")
    return result


torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch single batch"):
    print("#" * 10, "INPUT", "#" * 10)
    print(example)
    print("-" * 30)

    result = call_model(
        model=model,
        input_text=example,
        batch_size=1,
        desc="Torch",
        verbose=True
    )
    _ = call_model(
        model=model,
        input_text=example,
        batch_size=4,
        desc="Torch",
        verbose=True
    )
    # text_output = tokenizer.decode(result[0])
    # result = text_output[len(example):]
    torch_outputs.append(result)

model.to("cpu")
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated single batch"):
    print("#" * 10, "INPUT", "#" * 10)
    print(example)
    print("-" * 30)

    result = call_model(
        model=ds_model,
        input_text=example,
        batch_size=1,
        desc="Deepspeed",
        verbose=True
    )
    _ = call_model(
        model=ds_model,
        input_text=example,
        batch_size=4,
        desc="Deepspeed",
        verbose=True
    )

    # text_output = tokenizer.decode(result[0])
    # result = text_output[len(example):]
    accelerated_outputs.append(result)

num_matches = 0
for torch_output, accelerated_output in zip(torch_outputs, accelerated_outputs):
    num_matches += (torch_output == accelerated_output)

    # print("#" * 100)
    # print(torch_output)
    # print("-" * 100)
    # print(accelerated_output)

print(f"Accuracy: {num_matches / len(INPUT_EXAMPLES)}")
print(f"Matches: {num_matches}/{len(INPUT_EXAMPLES)}")
