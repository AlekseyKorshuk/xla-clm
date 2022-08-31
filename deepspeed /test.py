import os
import deepspeed
import torch
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("ChaiML/user_model_inputs")

INPUT_EXAMPLES = dataset["train"]["text"][:10]


local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = pipeline('text-generation', model='gpt2',
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)

for example in INPUT_EXAMPLES:
    print("#"*100)
    string = generator(example, do_sample=False, max_new_tokens=50)[0]["generated_text"] #[len(example):]
    print(string)
