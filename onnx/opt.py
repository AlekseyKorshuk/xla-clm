import os

import tqdm
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_checkpoint = "gpt2"
save_directory = "tmp/onnx/"
file_name = "model.onnx"
onnx_path = os.path.join(save_directory, "model.onnx")

# Load a model from transformers and export it through the ONNX format
model = ORTModelForCausalLM.from_pretrained(model_checkpoint, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the onnx model and tokenizer
model.save_pretrained(save_directory, file_name=file_name)
tokenizer.save_pretrained(save_directory)


cls_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

results = cls_pipeline("I love burritos!")
print(results)

inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")

print("ONNX")
for _ in tqdm.trange(10):
    for _ in tqdm.trange(10):
        model(**inputs)
        # outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))