import os
from test.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_checkpoint = "hakurei/litv2-6B-rev2"
save_directory = "onnx/"
file_name = "model.onnx"
onnx_path = os.path.join(save_directory, "model.onnx")

# Load a model from transformers and export it through the ONNX format
model = ORTModelForCausalLM.from_pretrained(model_checkpoint, file_name=onnx_path)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


cls_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

results = cls_pipeline("I love burritos!")

# # Save the onnx model and tokenizer
# model.save_pretrained(save_directory, file_name=file_name)
# tokenizer.save_pretrained(save_directory)