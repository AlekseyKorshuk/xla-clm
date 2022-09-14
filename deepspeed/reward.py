import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings
from datasets import load_dataset
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, pipeline

warnings.filterwarnings("ignore")
data = load_dataset("ChaiML/rm_train_dataset", split="validation").shuffle(seed=42)
# Model Repository on huggingface.co
model_id = "ChaiML/gpt2-xl_num_messages_5"
tokenizer_id = "gpt2"
token = "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"
# model_id = "gpt2"

task_evaluator = evaluator("text-classification")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_auth_token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_id, use_auth_token=token).to(0)

torch_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

label_mapping = {"LABEL_0": 0, "LABEL_1": 1}
print("Pytorch")
torch_eval_results = task_evaluator.compute(
    model_or_pipeline=torch_pipe,
    data=data,
    label_mapping=label_mapping
)
print(torch_eval_results)

# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
ds_clf = pipeline("text-classification", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated")
eval_results = task_evaluator.compute(
    model_or_pipeline=ds_clf,
    data=data,
    label_mapping=label_mapping
)
print(eval_results)

print("Pytorch")
print(torch_eval_results)
