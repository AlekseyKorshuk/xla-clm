import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

generation_kwargs = {
    "max_new_tokens": 64,
    "eos_token_id": 198,
    "do_sample": True,
    "temperature": 0.72,
    "top_k": 0,
    "top_p": 0.725,
    "repetition_penalty": 1.13,
}

padding_kwargs = {
    "pad_to_multiple_of": 8,
    "padding": True,
    "padding_side": "left",
    "pad_token": "</s>",
}

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", **padding_kwargs
)
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
input_text = "repetition_penalty error"

xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_text, return_tensors="tf")

print("model.generate")
model.generate(**tokenized_input, **generation_kwargs)

print("xla_generate")
xla_generate(**tokenized_input, **generation_kwargs)
