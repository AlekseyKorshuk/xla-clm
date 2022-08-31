from optimum.onnxruntime import ORTModelForCausalLM

# load ORT Model
model = ORTModelForCausalLM.from_pretrained("distilgpt2", from_transformers=True)

"""defaultly placed on cpu"""

print(model.device)

"""check if GPU is available an move to GPU"""

import torch

if torch.cuda.is_available():
    model.to(torch.device("cuda"))

print(model.device)

"""load pipeline on GPU"""

from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
pip = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # device 0 for gpu
print(pip.device)
