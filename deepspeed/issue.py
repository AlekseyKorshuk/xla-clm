import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from datasets import load_dataset

dataset = load_dataset("ChaiML/user_model_inputs")

model_id = "hakurei/litv2-6B-rev2"

VERBOSE = True
BATCH_SIZE = 4

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

torch_model = AutoModelForCausalLM.from_pretrained(model_id).half().eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = 198


def call_model(model, input_texts, desc="", verbose=False):
    inputs = tokenizer(input_texts, return_tensors='pt', padding="longest", padding_side="left").to(0)

    if verbose:
        print(desc)
        print(f"Batch size: {len(input_texts)}")
        print(f"Input size: {inputs.input_ids.size()}")

    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    outputs_set = list()
    output = None
    for i, output in enumerate(outputs):
        text_output = tokenizer.decode(output[len(inputs.input_ids[i]):], skip_special_tokens=True)
        output = text_output
        outputs_set.append(output)
        if verbose:
            print(f"#{i}: {output}")
    return output


ds_model = deepspeed.init_inference(
    model=torch_model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

call_model(
    model=ds_model,
    input_texts=dataset["train"]["text"][:4],
    desc="Deepspeed",
    verbose=VERBOSE
)
