import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from datasets import load_dataset

dataset = load_dataset("ChaiML/user_model_inputs")

model_id = "hakurei/litv2-6B-rev2"
# model_id = "gpt2"

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

torch_model = AutoModelForCausalLM.from_pretrained(model_id).half().eval().to(0)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token_id = 198


def call_model(model, input_texts, desc="", verbose=False):
    inputs = tokenizer(input_texts, return_tensors='pt', padding="longest").to(0)

    if verbose:
        print(desc)
        print(f"Batch size: {len(input_texts)}")
        print(f"Input size: {inputs.input_ids.size()}")

    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    outputs_set = list()
    output = None
    for i, output in enumerate(outputs):
        text_output = tokenizer.decode(output[len(inputs.input_ids[i]):], skip_special_tokens=False)
        output = text_output
        outputs_set.append(output)
        if verbose:
            print(f"#{i}: {output.strip()}")
    return output


LONG_EXAMPLE = "User: How are you?\nBot: This is demo response, it is quite long to check everything.\n" * 30 + \
               "User: How are you?\nBot:"
print(f"Long example length: {len(LONG_EXAMPLE)}")
input_texts = [LONG_EXAMPLE] * BATCH_SIZE

call_model(
    model=torch_model,
    input_texts=input_texts,
    desc="Torch",
    verbose=VERBOSE
)

torch_model.cpu()

ds_model = deepspeed.init_inference(
    model=torch_model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

call_model(
    model=ds_model,
    input_texts=input_texts,
    desc="Deepspeed",
    verbose=VERBOSE
)
