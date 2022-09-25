import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

dataset = load_dataset("ChaiML/user_model_inputs")
# model_id = "hakurei/litv2-6B-rev2"
model_id = "gpt2"

NUM_SAMPLES = 1
VERBOSE = True
BATCH_SIZE = 4
MAX_TOKENS = 2048

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
torch_model = AutoModelForCausalLM.from_pretrained(model_id).half().eval().to(0)

INPUT_EXAMPLES = dataset["train"]["text"][:NUM_SAMPLES]

# load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)


def prepare_input(example):
    tokens = tokenizer(example, return_tensors='pt').input_ids[0]
    # print(len(tokens))
    tokens = tokens[
             -(MAX_TOKENS - GENERATION_KWARGS["max_new_tokens"]):
             ]
    # print(len(tokens))
    return tokenizer.decode(
        tokens
    )


INPUT_EXAMPLES = [
    prepare_input(example) for example in INPUT_EXAMPLES
]


def call_model(model, input_text, batch_size, desc="", verbose=False):
    assert batch_size > 0
    if verbose:
        print(desc)
        print(f"Batch size: {batch_size}")
    inputs = tokenizer([input_text] * batch_size, return_tensors='pt').to(0)
    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    outputs_set = list()
    output = None
    for i, output in enumerate(outputs):
        text_output = tokenizer.decode(output)
        output = text_output[len(input_text):]
        outputs_set.append(output)
        if verbose:
            print(f"#{i}: {output}")
    assert len(set(outputs_set)) == 1
    return output


torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch"):
    if VERBOSE:
        print("#" * 10, "INPUT", "#" * 10)
        print(example)
        print("-" * 30)

    result = call_model(
        model=torch_model,
        input_text=example,
        batch_size=BATCH_SIZE,
        desc="Torch",
        verbose=VERBOSE
    )

    torch_outputs.append(result)

torch_model.to("cpu")
ds_model = deepspeed.init_inference(
    model=torch_model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated"):
    if VERBOSE:
        print("#" * 10, "INPUT", "#" * 10)
        print(example)
        print("-" * 30)

    result = call_model(
        model=ds_model,
        input_text=example,
        batch_size=BATCH_SIZE,
        desc="Deepspeed",
        verbose=VERBOSE
    )

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
