import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
import deepspeed
import tqdm

model_id = "gpt2"

VERBOSE = True
BATCH_SIZE = 4

EXAMPLE = "Model Implementations for Inference (MII) is an open-sourced repository" \
          " for making low-latency and high-throughput inference accessible to all " \
          "data scientists by alleviating the need to apply complex system optimization " \
          "techniques themselves. Out-of-box, MII offers support for thousands of " \
          "widely used DL models, optimized using DeepSpeed-Inference, that can be " \
          "deployed with a few lines of code, while achieving significant latency " \
          "reduction compared to their vanilla open-sourced versions."

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

tokenizer = AutoTokenizer.from_pretrained(model_id)


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


call_model(
    model=torch_model,
    input_text=EXAMPLE,
    batch_size=BATCH_SIZE,
    desc="Torch",
    verbose=VERBOSE
)

torch_model.to("cpu")

ds_model = deepspeed.init_inference(
    model=torch_model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

call_model(
    model=ds_model,
    input_text=EXAMPLE,
    batch_size=BATCH_SIZE,
    desc="Deepspeed",
    verbose=VERBOSE
)
