import os

import tqdm
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

model_checkpoint = "hakurei/litv2-6B-rev2"
save_directory = "tmp/onnx/"
file_name = "model.onnx"
onnx_path = os.path.join(save_directory, "model.onnx")

# Load a model from transformers and export it through the ONNX format
model = ORTModelForCausalLM.from_pretrained(save_directory, file_name=file_name)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""defaultly placed on cpu"""

print(model.device)

"""check if GPU is available an move to GPU"""

import torch

if torch.cuda.is_available():
    model.to(torch.device("cuda"))

print(model.device)

# Save the onnx model and tokenizer
# model.save_pretrained(save_directory, file_name=file_name)
# tokenizer.save_pretrained(save_directory)


cls_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

results = cls_pipeline("I love burritos!")
print(results)

inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="pt")

print("ONNX")
for _ in tqdm.trange(10):
    for _ in tqdm.trange(10):
        model(**inputs)
        # outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))

print("Pytorch")
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(0)
inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="pt").to(0)
for _ in tqdm.trange(10):
    for _ in tqdm.trange(10):
        outputs = model(**inputs)