import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
import deepspeed

model_id = "hakurei/litv2-6B-rev2"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Test pipeline
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

ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
pipe = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

result = pipe("Chai is")
print(result)

inputs = pipe.tokenizer("Chai is", return_tensors='pt')
result = pipe.model.generate(**inputs, **GENERATION_KWARGS)
print(result)
