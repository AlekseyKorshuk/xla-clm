# Stand-alone TF XLA generate example for Decoder-Only Models.

# Note: execution times are deeply dependent on hardware.
# If you have a machine with a powerful GPU, I highly recommend you to try this example there!
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# 1. Load model and tokenizer
model_name = "gpt2" # hakurei/litv2-6B-rev2
# remember: decoder-only models need left-padding
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)

# 2. Prepare tokenization and generation arguments -- don't forget padding to avoid retracing!
tokenization_kwargs = {"pad_to_multiple_of": 64, "padding": True, "return_tensors": "tf"}
generation_kwargs = {
    'temperature': 0.72,
    'repetition_penalty': 1.13125,
    'max_new_tokens': 64,
    'top_p': 0.725,
    'top_k': 0,
    'do_sample': True,
    'eos_token_id': 198,
    # 'bad_words_ids': self.bad_words_ids
}
# generation_kwargs = {"num_beams": 4, "max_new_tokens": 32}


# 3. Create your XLA generate function a̶n̶d̶ ̶m̶a̶k̶e̶ ̶P̶y̶T̶o̶r̶c̶h̶ ̶e̶a̶t̶ ̶d̶u̶s̶t̶
# This is the only change with respect to original generate workflow!
xla_generate = tf.function(model.generate, jit_compile=True)

# print("ORIGINAL GENERATE:")
# # 4. Generate! Remember -- the first call will be slow, but all subsequent calls will be fast if you've done things right.
# input_prompts = [f"The best thing about {country} is" for country in ["Spain", "Japan", "Angola"]]
# for input_prompt in input_prompts:
#     tokenized_inputs = tokenizer([input_prompt], **tokenization_kwargs)
#     start = time.time_ns()
#     generated_text = model.generate(**tokenized_inputs, **generation_kwargs)
#     end = time.time_ns()
#     decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
#     print(f"Original prompt -- {input_prompt}")
#     print(f"Generated -- {decoded_text}")
#     print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# print("#" * 80)

print("XLA GENERATE:")
input_prompts = [f"The best thing about {country} is" for country in ["Spain", "Japan", "Angola"]]
for input_prompt in input_prompts:
    tokenized_inputs = tokenizer([input_prompt], **tokenization_kwargs)
    start = time.time_ns()
    generated_text = xla_generate(**tokenized_inputs, **generation_kwargs)
    end = time.time_ns()
    decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(f"Original prompt -- {input_prompt}")
    print(f"Generated -- {decoded_text}")
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
print("#" * 80)
