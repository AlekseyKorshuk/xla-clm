import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
dataset = load_dataset("ChaiML/user_model_inputs")
# Model Repository on huggingface.co
model_id = "ChaiML/reward_48m_gpt2_target_2"
tokenizer_id = "gptw"
token = "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"
# model_id = "gpt2"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token).to(0)

INPUT_EXAMPLES = dataset["train"]["reward_input"][:100]

torch_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

print("Pytorch")
torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch"):
    # inputs = tokenizer(example, return_tensors='pt').to(0)
    # result = model.generate(**inputs, **GENERATION_KWARGS)
    torch_output = torch_pipe(example)[0]
    print(torch_output)
    # torch_outputs.append(torch_output)
# print(torch_output)
# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
ds_clf = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated")
accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated"):
    accelerated_output = ds_clf(example)[0]
    # accelerated_outputs.append(accelerated_output)

difference = list(set(torch_outputs) - set(accelerated_outputs))
print(len(difference))
