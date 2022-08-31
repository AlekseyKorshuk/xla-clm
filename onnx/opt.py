import os

import tqdm
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

model_checkpoint = "hakurei/litv2-6B-rev2"
save_directory = "tmp/onnx/"
file_name = "model.onnx"
onnx_path = os.path.join(save_directory, "model.onnx")

# Load a model from transformers and export it through the ONNX format
opt_model = ORTModelForCausalLM.from_pretrained(model_checkpoint, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""defaultly placed on cpu"""

print(opt_model.device)

"""check if GPU is available an move to GPU"""

import torch

if torch.cuda.is_available():
    opt_model.to(torch.device("cuda"))

print(opt_model.device)

# Save the onnx model and tokenizer
# model.save_pretrained(save_directory, file_name=file_name)
# tokenizer.save_pretrained(save_directory)


opt_pipeline = pipeline("text-generation", model=opt_model, tokenizer=tokenizer, device=0)

results = opt_pipeline("I love burritos!")
print(results)

inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="pt").to(0)

print("ONNX")
for _ in range(1):
    for _ in tqdm.trange(10):
        # opt_pipeline("I love burritos!", do_sample=False)
        opt_pipeline.model.generate(**inputs)
        # outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
for _ in range(1):
    for _ in tqdm.trange(1000):
        opt_pipeline.model(**inputs)
        # cls_pipeline("I love burritos!", do_sample=False)

print("Pytorch")
# model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(0)
torch_pipeline = pipeline("text-generation", model_checkpoint, device=0)

for _ in range(1):
    for _ in tqdm.trange(10):
        torch_pipeline.model.generate(**inputs)
        # torch_pipeline("I love burritos!", do_sample=False)
for _ in range(1):
    for _ in tqdm.trange(1000):
        torch_pipeline.model(**inputs)
