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
model_id = "KoboldAI/OPT-6B-nerys-v2"
# model_id = "gpt2"

stats = {}

# load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Test pipeline
GENERATION_KWARGS = {
    "max_new_tokens": 32,
    # "min_new_tokens": 8,
    'eos_token_id': 198,
    'do_sample': True,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

INPUT_EXAMPLES = dataset["train"]["text"][:100]

example = INPUT_EXAMPLES[0]

import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_checkpoint = "KoboldAI/OPT-6B-nerys-v2"
save_directory = "onnx/"
file_name = "model.onnx"
onnx_path = os.path.join(save_directory, "model.onnx")

# Load a model from transformers and export it through the ONNX format
# model = ORTModelForCausalLM.from_pretrained(model_checkpoint, from_transformers=True).to(0)

max_batch_size = 1
for i in range(1, 5):
    try:
        inputs = tokenizer([example] * i, return_tensors='pt').to(0)
        # result = model.generate(**inputs, **GENERATION_KWARGS)
        print(f"Batch size: {i}")
        max_batch_size = i
    except Exception as ex:
        print(ex)
        break

# torch_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
print("ONNX single batch")
torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES[:20], desc="ONNX single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    # result = model.generate(**inputs, **GENERATION_KWARGS)
    # torch_output = torch_pipe(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    # torch_outputs.append(torch_output)
print("ONNX batch size")
torch_outputs = []
try:
    for example in tqdm.tqdm(INPUT_EXAMPLES[:10], desc="ONNX batch size"):
        inputs = tokenizer([example] * max_batch_size, return_tensors='pt').to(0)
        # result = model.generate(**inputs, **GENERATION_KWARGS)
except Exception as ex:
    print(ex)
