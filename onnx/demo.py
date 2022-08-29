from optimum_transformers import pipeline

# Initialize a pipeline by passing the task name and
# set onnx to True (default value is also True)
nlp = pipeline("text-generation", "gpt2", use_onnx=True)
nlp("Transformers and onnx runtime is an awesome")
# [{'label': 'POSITIVE', 'score': 0.999721109867096}]
