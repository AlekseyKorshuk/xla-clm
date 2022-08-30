from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import tqdm

tokenizer = AutoTokenizer.from_pretrained("hakurei/litv2-6B-rev2")
session = InferenceSession("onnx/model.onnx", providers=["CUDAExecutionProvider"])
# ONNX Runtime expects NumPy arrays as input
inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")

for _ in tqdm.trange(10):
    for _ in tqdm.trange(10):
        outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
