import os
import deepspeed
import torch
from transformers import pipeline

prompt = """Eliza is a therapist and she loves to listen and help.

Eliza: Hi, my name is Eliza. What is weighing on your mind?
Me: Hey Eliza
Eliza: Hi. I'm a therapist. How are you feeling?
User: I am feeling lonely
Eliza: This is dump dialogue
Me: Yes yes
Eliza: This is interesting dialogue
User: Are you sure?
Eliza: This is dump dialogue
Me: Yes yes
Eliza: This is interesting dialogue
User: Are you sure?
Eliza: This is dump dialogue
Me: Yes yes
Eliza: This is interesting dialogue
User: Are you sure?
Eliza:"""

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B',
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)

generated_text = generator("Deepspeed is", do_sample=False, max_new_tokens=50, eos_token_id=198)[0]["generated_text"]
print(generated_text)

generated_text = generator(prompt, do_sample=False, max_new_tokens=50, eos_token_id=198)[0]["generated_text"]
print(generated_text)
