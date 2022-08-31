# Filename: gpt-neo-2.7b-generation.py
import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B',
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=False, min_length=50)
print(string)
