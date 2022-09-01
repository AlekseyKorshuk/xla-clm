import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
dataset = load_dataset("ChaiML/user_model_inputs")
# Model Repository on huggingface.co
model_id = "hakurei/litv2-6B-rev2"
# model_id = "gpt2"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Test pipeline
GENERATION_KWARGS = {
    "max_new_tokens": 64,
    # "min_new_tokens": 8,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

INPUT_EXAMPLES = dataset["train"]["text"][:100]

torch_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
print("Pytorch")
torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch"):
    inputs = torch_pipe.tokenizer(example, return_tensors='pt').to(0)
    result = torch_pipe.model.generate(**inputs, **GENERATION_KWARGS)
    # torch_output = torch_pipe(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    # torch_outputs.append(torch_output)
# print(torch_output)
# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
ds_clf = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated")
accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated"):
    inputs = ds_clf.tokenizer(example, return_tensors='pt').to(0)
    result = ds_clf.model.generate(**inputs, **GENERATION_KWARGS)
    # accelerated_output = ds_clf(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    # accelerated_outputs.append(accelerated_output)

difference = list(set(torch_outputs) - set(accelerated_outputs))
print(len(difference))
