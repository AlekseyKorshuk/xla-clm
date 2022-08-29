# from optimum_transformers import pipeline
#
# # Initialize a pipeline by passing the task name and
# # set onnx to True (default value is also True)
# nlp = pipeline("text-generation", "gpt2", use_onnx=True)
# nlp("Transformers and onnx runtime is an awesome")
# # [{'label': 'POSITIVE', 'score': 0.999721109867096}]

from optimum_transformers import Benchmark

task = "text-generation"
model_name = "gpt2"
num_tests = 3

benchmark = Benchmark(task, model_name)
results = benchmark(num_tests, plot=True)
print(results)
