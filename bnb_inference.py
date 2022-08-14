import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': True,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

# MAX_NEW_TOKENS = 128
model_name = 'hakurei/litv2-6B-rev2'  # hakurei/litv2-6B-rev2 facebook/opt-66b
text = """
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. 
How many punches did he throw?\n
A: Let’s think step by step.\n"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids
free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
max_memory = f'{free_in_GB - 2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_8bit=True,
    max_memory=max_memory
)

for i in tqdm.trange(10):
    with torch.autocast(device_type='cuda'):
        generated_ids = model.generate(input_ids, **GENERATION_KWARGS)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
